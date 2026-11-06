// AudioPreviewEngine implementation. See the header for the contract:
// render() is real-time safe, resamples by linear interpolation and
// never allocates, locks or touches the filesystem.

#include "AudioPreviewEngine.h"

#include <algorithm>

void AudioPreviewEngine::setClip (std::shared_ptr<const AudioClip> newClip)
{
    playing.store (false, std::memory_order_release);
    positionSamples.store (0.0, std::memory_order_release);
    endedFlag.store (false, std::memory_order_release);
    clip.store (std::move (newClip), std::memory_order_release);
}

bool AudioPreviewEngine::start()
{
    auto c = getClip();
    if (c == nullptr || c->isEmpty())
        return false;

    endedFlag.store (false, std::memory_order_release);
    playing.store (true, std::memory_order_release);
    return true;
}

void AudioPreviewEngine::stop()
{
    playing.store (false, std::memory_order_release);
    positionSamples.store (0.0, std::memory_order_release);
}

void AudioPreviewEngine::pause()
{
    playing.store (false, std::memory_order_release);
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

    auto c = getClip();
    if (c == nullptr || c->isEmpty() || c->sourceSampleRate <= 0.0)
    {
        playing.store (false, std::memory_order_release);
        positionSamples.store (0.0, std::memory_order_release);
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

            playing.store (false, std::memory_order_release);
            positionSamples.store (0.0, std::memory_order_release);
            endedFlag.store (true, std::memory_order_release);
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

    // Only publish the new position if playback is still running:
    // a stop() from the UI during this block must win.
    if (playing.load (std::memory_order_acquire))
        positionSamples.store (position, std::memory_order_release);
}
