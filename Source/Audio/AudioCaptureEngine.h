#pragma once

// AudioCaptureEngine: records host audio from processBlock() into a
// preallocated buffer (Phase 2).
//
// The capture buffer is allocated ONCE in prepare() (called from the
// processor's prepareToPlay, never on the audio thread). processInput()
// copies samples with no allocation, no locking and no file I/O.
//
// Maximum recording duration is 60 seconds. When the buffer fills,
// recording stops safely inside processBlock() and the engine reports
// LimitReached; the message thread finalizes the clip (see
// MelotheonAudioProcessor::finalizeRecordingFromLimit). Recording can
// never overwrite memory past the buffer end.
//
// State machine: Idle -> Recording -> Idle (stop) or LimitReached
// (60 s reached, awaiting finalization). startRecording() only works
// from Idle; stopRecording() returns the recorded sample count, or -1
// when nothing was recording.

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <atomic>
#include <memory>

#include "AudioClip.h"

class AudioCaptureEngine
{
public:
    static constexpr double kMaxDurationSeconds = 60.0;

    enum class State
    {
        Idle = 0,
        Recording = 1,
        LimitReached = 2
    };

    AudioCaptureEngine() = default;

    // Allocate the capture buffer for the given host rate / channel
    // count. NOT real-time safe: call only from prepareToPlay().
    // Any recording in progress is discarded.
    void prepare (double sampleRate, int numChannels);

    // Begin a take: reset the write position and clear the capture
    // area. Returns false unless the engine was idle.
    bool startRecording();

    // End a take. Returns the recorded sample count, or -1 when
    // nothing was recording. Message thread only.
    int stopRecording();

    // Copy one block of host input into the capture buffer.
    // Real-time safe: no allocation, no locks, no I/O. Silently
    // stops at the 60 s limit.
    void processInput (const juce::AudioBuffer<float>& buffer);

    // Build the finished clip from the recorded region. Message
    // thread only, after stopRecording().
    std::shared_ptr<const AudioClip> createClip (const juce::String& name) const;

    bool isRecording() const
    {
        return state.load (std::memory_order_acquire) == static_cast<int> (State::Recording);
    }

    bool limitReached() const { return limitReachedFlag.load (std::memory_order_acquire); }
    void clearLimitFlag() { limitReachedFlag.store (false, std::memory_order_release); }

    int getRecordedSamples() const { return writePosition.load (std::memory_order_acquire); }
    double getSampleRate() const { return sampleRate; }
    int getNumChannels() const { return numChannels; }

private:
    juce::AudioBuffer<float> captureBuffer; // preallocated in prepare()
    std::atomic<int> state { static_cast<int> (State::Idle) };
    std::atomic<int> writePosition { 0 };
    std::atomic<bool> limitReachedFlag { false };
    double sampleRate = 44100.0;
    int numChannels = 2;
    int maxSamples = 0;
};
