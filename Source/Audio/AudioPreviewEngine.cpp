// AudioPreviewEngine implementation. See the header for the contract:
// render() is real-time safe, resamples by linear interpolation and
// never allocates, locks, performs shared_ptr reference counting or
// touches the filesystem.

#include "AudioPreviewEngine.h"

#include <algorithm>
#include <thread>

namespace
{
class RenderActivityGuard
{
public:
    explicit RenderActivityGuard (std::atomic<int>& counterToUse)
        : counter (counterToUse)
    {
        counter.fetch_add (1, std::memory_order_seq_cst);
    }

    ~RenderActivityGuard()
    {
        counter.fetch_sub (1, std::memory_order_seq_cst);
    }

private:
    std::atomic<int>& counter;
};
} // namespace

AudioPreviewEngine::~AudioPreviewEngine()
{
    setClip (nullptr);
}

void AudioPreviewEngine::setClip (std::shared_ptr<const AudioClip> newClip)
{
    playing.store (false, std::memory_order_release);
    transportGeneration.fetch_add (1, std::memory_order_acq_rel);
    clipForAudio.store (nullptr, std::memory_order_seq_cst);
    waitForRendererToFinish();

    ownedClip = std::move (newClip);
    positionSamples.store (0.0, std::memory_order_release);
    endedFlag.store (false, std::memory_order_release);
    clipForAudio.store (ownedClip.get(), std::memory_order_seq_cst);
}

bool AudioPreviewEngine::start()
{
    auto c = getClip();
    if (c == nullptr || c->isEmpty())
        return false;

    endedFlag.store (false, std::memory_order_release);
    transportGeneration.fetch_add (1, std::memory_order_acq_rel);
    playing.store (true, std::memory_order_release);
    return true;
}

void AudioPreviewEngine::stop()
{
    playing.store (false, std::memory_order_release);
    transportGeneration.fetch_add (1, std::memory_order_acq_rel);
    positionSamples.store (0.0, std::memory_order_release);
    endedFlag.store (false, std::memory_order_release);
}

void AudioPreviewEngine::pause()
{
    playing.store (false, std::memory_order_release);
    transportGeneration.fetch_add (1, std::memory_order_acq_rel);
    endedFlag.store (false, std::memory_order_release);
}

double AudioPreviewEngine::getPositionSeconds() const
{
    auto c = getClip();
    if (c == nullptr || c->sourceSampleRate <= 0.0)
        return 0.0;

    return positionSamples.load (std::memory_order_acquire) / c->sourceSampleRate;
}

double AudioPreviewEngine::getDurationSeconds() const
{
    auto c = getClip();
    return c != nullptr ? c->durationSeconds : 0.0;
}

void AudioPreviewEngine::render (juce::AudioBuffer<float>& buffer, double hostSampleRate)
{
    if (! playing.load (std::memory_order_acquire) || hostSampleRate <= 0.0)
        return;

    RenderActivityGuard renderGuard (renderersInFlight);
    const std::uint64_t generation = transportGeneration.load (std::memory_order_acquire);

    // A stop or clip replacement may have arrived between the fast-path
    // check and our in-flight announcement.
    if (! playing.load (std::memory_order_acquire))
        return;

    const AudioClip* c = clipForAudio.load (std::memory_order_seq_cst);
    if (c == nullptr || c->isEmpty() || c->sourceSampleRate <= 0.0)
    {
        if (transportGeneration.load (std::memory_order_acquire) == generation)
        {
            playing.store (false, std::memory_order_release);
            positionSamples.store (0.0, std::memory_order_release);
        }
        return;
    }

    const int numSamples = buffer.getNumSamples();
    const int numOutputChannels = buffer.getNumChannels();
    const int clipChannels = c->previewBuffer.getNumChannels();
    const juce::int64 totalSamples = c->totalSamples;
    const double ratio = c->sourceSampleRate / hostSampleRate;

    float* const* output = buffer.getArrayOfWritePointers();
    const float* const* clipData = c->previewBuffer.getArrayOfReadPointers();

    double position = positionSamples.load (std::memory_order_acquire);

    for (int n = 0; n < numSamples; ++n)
    {
        if (position >= static_cast<double> (totalSamples))
        {
            // Clip ended mid-block: silence the rest of the block.
            for (int ch = 0; ch < numOutputChannels; ++ch)
                std::fill (output[ch] + n, output[ch] + numSamples, 0.0f);

            // A manual Stop/Pause or clip replacement must win over a
            // natural-end notification from this older render block.
            if (transportGeneration.load (std::memory_order_acquire) == generation
                && playing.exchange (false, std::memory_order_acq_rel))
            {
                positionSamples.store (0.0, std::memory_order_release);
                endedFlag.store (true, std::memory_order_release);
            }
            return;
        }

        const juce::int64 index0 = static_cast<juce::int64> (position);
        const float frac = static_cast<float> (position - static_cast<double> (index0));
        const juce::int64 index1 = juce::jmin (index0 + 1, totalSamples - 1);

        for (int ch = 0; ch < numOutputChannels; ++ch)
        {
            // Mono clip -> both outputs; stereo clip -> matching
            // channel; extra output channels reuse the last source.
            const int sourceChannel = juce::jmin (ch, clipChannels - 1);
            const float* data = clipData[sourceChannel];
            output[ch][n] = data[index0] + (data[index1] - data[index0]) * frac;
        }

        position += ratio;
    }

    // Only publish the new position if this is still the current
    // transport generation. This makes Stop/Pause win even when it
    // arrives after the render block's first playing check.
    if (playing.load (std::memory_order_acquire)
        && transportGeneration.load (std::memory_order_acquire) == generation)
        positionSamples.store (position, std::memory_order_release);
}

void AudioPreviewEngine::waitForRendererToFinish()
{
    while (renderersInFlight.load (std::memory_order_seq_cst) != 0)
        std::this_thread::yield();
}
