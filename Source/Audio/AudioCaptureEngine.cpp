// AudioCaptureEngine implementation. See the header for the contract:
// the capture buffer is preallocated in prepare(); processInput() is
// real-time safe and can never write past the 60 s limit.

#include "AudioCaptureEngine.h"

#include <cmath>

void AudioCaptureEngine::prepare (double newSampleRate, int newNumChannels)
{
    // Any take in progress is discarded: prepareToPlay() runs while
    // the audio callback is suspended by the host.
    state.store (static_cast<int> (State::Idle), std::memory_order_release);
    writePosition.store (0, std::memory_order_release);
    limitReachedFlag.store (false, std::memory_order_release);

    sampleRate = newSampleRate > 0.0 ? newSampleRate : 44100.0;
    numChannels = juce::jlimit (1, 2, newNumChannels);
    maxSamples = static_cast<int> (std::ceil (kMaxDurationSeconds * sampleRate));

    // Single allocation for the whole session (cleared).
    captureBuffer.setSize (numChannels, maxSamples, false, true, true);
}

bool AudioCaptureEngine::startRecording()
{
    int expected = static_cast<int> (State::Idle);
    if (! state.compare_exchange_strong (expected, static_cast<int> (State::Recording),
                                        std::memory_order_acq_rel))
        return false;

    writePosition.store (0, std::memory_order_release);
    limitReachedFlag.store (false, std::memory_order_release);
    // Clear so no stale data from an earlier take can leak in.
    // No allocation: the buffer already exists.
    captureBuffer.clear();
    return true;
}

int AudioCaptureEngine::stopRecording()
{
    const int current = state.load (std::memory_order_acquire);
    if (current == static_cast<int> (State::Idle))
        return -1;

    state.store (static_cast<int> (State::Idle), std::memory_order_release);
    return writePosition.load (std::memory_order_acquire);
}

void AudioCaptureEngine::processInput (const juce::AudioBuffer<float>& buffer)
{
    if (state.load (std::memory_order_acquire) != static_cast<int> (State::Recording))
        return;

    const int writePos = writePosition.load (std::memory_order_acquire);
    const int remaining = maxSamples - writePos;
    if (remaining <= 0)
    {
        state.store (static_cast<int> (State::LimitReached), std::memory_order_release);
        limitReachedFlag.store (true, std::memory_order_release);
        return;
    }

    const int numSamples = buffer.getNumSamples();
    const int toCopy = juce::jmin (remaining, numSamples);
    const int channelsToCopy = juce::jmin (numChannels, buffer.getNumChannels());

    for (int ch = 0; ch < channelsToCopy; ++ch)
        captureBuffer.copyFrom (ch, writePos, buffer, ch, 0, toCopy);

    const int newWritePos = writePos + toCopy;
    writePosition.store (newWritePos, std::memory_order_release);

    if (newWritePos >= maxSamples)
    {
        state.store (static_cast<int> (State::LimitReached), std::memory_order_release);
        limitReachedFlag.store (true, std::memory_order_release);
    }
}

std::shared_ptr<const AudioClip> AudioCaptureEngine::createClip (const juce::String& name) const
{
    const int samples = writePosition.load (std::memory_order_acquire);
    juce::AudioBuffer<float> region (numChannels, samples);
    for (int ch = 0; ch < numChannels; ++ch)
        region.copyFrom (ch, 0, captureBuffer, ch, 0, samples);

    return AudioClip::fromBuffer (region, sampleRate, name, false);
}
