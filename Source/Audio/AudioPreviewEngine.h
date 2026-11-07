#pragma once

// AudioPreviewEngine: plays the active AudioClip back through the
// host's Mixer Track (Phase 2).
//
// The engine is driven from processBlock(): render() overwrites the
// output buffer with the clip audio, resampled to the host sample
// rate by linear interpolation, so a file recorded at a different
// rate plays at the correct speed and pitch. No second audio device
// is created and host routing is never bypassed. When preview is
// inactive the processor passes host input through unchanged and
// render() is never called.
//
// Thread safety: the message thread owns the immutable AudioClip. The
// audio thread receives a non-owning atomic pointer and announces each
// render that is in flight. setClip() first removes that pointer, then
// waits on the message thread before releasing the old owner. This keeps
// shared_ptr reference counting and destruction off the audio thread.
// Playback position is an atomic double (in clip samples) updated once
// per block. No allocation, no locks and no I/O occur in render().

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "AudioClip.h"

class AudioPreviewEngine
{
public:
    AudioPreviewEngine() = default;
    ~AudioPreviewEngine();

    // Install a new clip (message thread). Stops any playback in
    // progress and resets the position to the start.
    void setClip (std::shared_ptr<const AudioClip> clip);

    // Message thread only. The audio callback never copies a shared_ptr.
    std::shared_ptr<const AudioClip> getClip() const { return ownedClip; }

    bool hasClip() const
    {
        auto c = getClip();
        return c != nullptr && ! c->isEmpty();
    }

    // Start playback from the current internal position. Returns
    // false when there is no clip to play.
    bool start();

    // Stop and return the cursor to the beginning.
    void stop();

    // Stop but keep the cursor where it is (resume with start()).
    void pause();

    bool isPlaying() const { return playing.load (std::memory_order_acquire); }

    double getPositionSeconds() const;
    double getDurationSeconds() const;

    // True once after playback reached the end of the clip. The
    // message thread consumes it to update the status display.
    bool consumePlaybackEnded() { return endedFlag.exchange (false, std::memory_order_acq_rel); }

    // Render one block of preview audio into the output buffer,
    // overwriting it. Real-time safe. When the clip ends mid-block
    // the remainder of the block is silence and playback stops.
    void render (juce::AudioBuffer<float>& buffer, double hostSampleRate);

private:
    // Wait only on the message thread while replacing/destroying a clip.
    void waitForRendererToFinish();

    std::shared_ptr<const AudioClip> ownedClip; // message-thread owner
    std::atomic<const AudioClip*> clipForAudio { nullptr };
    std::atomic<int> renderersInFlight { 0 };
    std::atomic<bool> playing { false };
    std::atomic<double> positionSamples { 0.0 };
    std::atomic<bool> endedFlag { false };
    // Stop, pause and clip replacement increment this token. render()
    // publishes its final cursor only if its token is still current.
    std::atomic<std::uint64_t> transportGeneration { 0 };
};
