#pragma once

// AudioCaptureEngine: records host audio from processBlock() into a
// preallocated buffer (Phase 2).
//
// The capture buffer is allocated ONCE in prepare() (called from the
// processor's prepareToPlay, never on the audio thread). processInput()
// copies samples with no allocation, no locking and no file I/O.
//
// The message thread and audio thread use a small lock-free hand-off:
// processInput() announces when it is touching the capture buffer and
// stopRecording() waits for that single in-flight copy to finish before
// the message thread reads the finished take. This prevents a clip from
// being copied while the audio callback is still writing its last block.
//
// Maximum recording duration is 60 seconds. When the buffer fills,
// recording stops safely inside processBlock() and the engine reports
// LimitReached; the message thread finalizes the clip (see
// MelotheonAudioProcessor::finalizeRecordingFromLimit). Recording can
// never overwrite memory past the buffer end.
//
// State machine: Idle -> Arming -> Recording -> Idle (stop) or
// LimitReached (60 s reached, awaiting finalization). Arming is a short
// message-thread-only transition that keeps the audio callback out while
// a new take is reset. startRecording() only works from Idle;
// stopRecording() returns the recorded sample count, or -1 when nothing
// was recording.

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
        LimitReached = 2,
        Arming = 3
    };

    AudioCaptureEngine() = default;

    // Allocate the capture buffer for the given host rate / channel
    // count. NOT real-time safe: call only from prepareToPlay().
    // Any recording in progress is discarded.
    void prepare (double sampleRate, int numChannels);

    // Begin a take and reset the write position. Returns false unless
    // the engine was idle. The old buffer does not need a full clear:
    // every recorded region is overwritten before it can be published.
    bool startRecording();

    // End a take. Returns the recorded sample count, or -1 when
    // nothing was recording. Message thread only.
    int stopRecording();

    // Copy one block of host input into the capture buffer.
    // Real-time safe: no allocation, no locks, no I/O. Silently
    // stops at the 60 s limit.
    void processInput (const juce::AudioBuffer<float>& buffer);

    // Build the finished clip from the recorded region. Message
    // thread only, after stopRecording(). Returns null for an empty take.
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
    // Message-thread helper. It may wait for at most the current audio
    // callback's capture copy. The audio thread never waits on it.
    void waitForAudioWriterToFinish();

    juce::AudioBuffer<float> captureBuffer; // preallocated in prepare()
    std::atomic<int> state { static_cast<int> (State::Idle) };
    std::atomic<int> writePosition { 0 };
    std::atomic<bool> limitReachedFlag { false };
    std::atomic<int> audioWritersInFlight { 0 };
    double sampleRate = 44100.0;
    int numChannels = 2;
    int maxSamples = 0;
};
