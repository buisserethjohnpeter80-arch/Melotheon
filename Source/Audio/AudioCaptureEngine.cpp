// AudioCaptureEngine implementation. See the header for the contract:
// the capture buffer is preallocated in prepare(); processInput() is
// real-time safe and can never write past the 60 s limit.

#include "AudioCaptureEngine.h"

#include <cmath>
#include <thread>

void AudioCaptureEngine::prepare (double newSampleRate, int newNumChannels)
{
    // Any take in progress is discarded: prepareToPlay() runs while
    // the audio callback is suspended by the host.
    state.store (static_cast<int> (State::Idle), std::memory_order_release);
    writePosition.store (0, std::memory_order_release);
    limitReachedFlag.store (false, std::memory_order_release);
    audioWritersInFlight.store (0, std::memory_order_release);

    sampleRate = newSampleRate > 0.0 ? newSampleRate : 44100.0;
    numChannels = juce::jlimit (1, 2, newNumChannels);
    maxSamples = static_cast<int> (std::ceil (kMaxDurationSeconds * sampleRate));

    // Single allocation for the whole session (cleared).
    captureBuffer.setSize (numChannels, maxSamples, false, true, true);
}

bool AudioCaptureEngine::startRecording()
{
    int expected = static_cast<int> (State::Idle);
    if (! state.compare_exchange_strong (expected, static_cast<int> (State::Arming),
                                        std::memory_order_acq_rel))
        return false;

    // A rapid Stop -> Record can arrive while the previous callback is
    // completing its last copy. Wait on the message thread before the
    // new take reuses the buffer.
    waitForAudioWriterToFinish();
    writePosition.store (0, std::memory_order_release);
    limitReachedFlag.store (false, std::memory_order_release);

    expected = static_cast<int> (State::Arming);
    return state.compare_exchange_strong (expected, static_cast<int> (State::Recording),
                                          std::memory_order_acq_rel);
}

int AudioCaptureEngine::stopRecording()
{
    const int current = state.exchange (static_cast<int> (State::Idle),
                                        std::memory_order_acq_rel);
    if (current == static_cast<int> (State::Idle))
        return -1;

    // Once Idle is published, a callback that has not entered the copy
    // will stand down. If one is already copying, wait until it publishes
    // the final write position before the message thread creates a clip.
    waitForAudioWriterToFinish();
    return writePosition.load (std::memory_order_acquire);
}

void AudioCaptureEngine::processInput (const juce::AudioBuffer<float>& buffer)
{
    if (state.load (std::memory_order_acquire) != static_cast<int> (State::Recording))
        return;

    audioWritersInFlight.fetch_add (1, std::memory_order_acq_rel);

    // stopRecording() may have published Idle between the first state
    // check and the in-flight announcement. Recheck before touching the
    // capture buffer.
    if (state.load (std::memory_order_acquire) != static_cast<int> (State::Recording))
    {
        audioWritersInFlight.fetch_sub (1, std::memory_order_release);
        return;
    }

    const int writePos = writePosition.load (std::memory_order_acquire);
    const int remaining = maxSamples - writePos;
    if (remaining <= 0)
    {
        int expected = static_cast<int> (State::Recording);
        if (state.compare_exchange_strong (expected, static_cast<int> (State::LimitReached),
                                           std::memory_order_acq_rel))
            limitReachedFlag.store (true, std::memory_order_release);

        audioWritersInFlight.fetch_sub (1, std::memory_order_release);
        return;
    }

    const int numSamples = buffer.getNumSamples();
    const int toCopy = juce::jmin (remaining, numSamples);
    const int channelsToCopy = juce::jmin (numChannels, buffer.getNumChannels());

    for (int ch = 0; ch < channelsToCopy; ++ch)
        captureBuffer.copyFrom (ch, writePos, buffer, ch, 0, toCopy);

    // Defensive channel handling: if a host ever supplies fewer channels
    // than it announced in prepareToPlay(), publish silence instead of
    // stale samples from an older take.
    for (int ch = channelsToCopy; ch < numChannels; ++ch)
        captureBuffer.clear (ch, writePos, toCopy);

    const int newWritePos = writePos + toCopy;
    writePosition.store (newWritePos, std::memory_order_release);

    if (newWritePos >= maxSamples)
    {
        int expected = static_cast<int> (State::Recording);
        if (state.compare_exchange_strong (expected, static_cast<int> (State::LimitReached),
                                           std::memory_order_acq_rel))
            limitReachedFlag.store (true, std::memory_order_release);
    }

    audioWritersInFlight.fetch_sub (1, std::memory_order_release);
}

std::shared_ptr<const AudioClip> AudioCaptureEngine::createClip (const juce::String& name) const
{
    const int samples = writePosition.load (std::memory_order_acquire);
    if (samples <= 0 || numChannels <= 0 || sampleRate <= 0.0)
        return nullptr;

    juce::AudioBuffer<float> region (numChannels, samples);
    for (int ch = 0; ch < numChannels; ++ch)
        region.copyFrom (ch, 0, captureBuffer, ch, 0, samples);

    return AudioClip::fromBuffer (region, sampleRate, name, false);
}

void AudioCaptureEngine::waitForAudioWriterToFinish()
{
    while (audioWritersInFlight.load (std::memory_order_acquire) != 0)
        std::this_thread::yield();
}
