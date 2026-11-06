#pragma once

// MelotheonAudioProcessor - the VST3 audio processor (Phase 2).
//
// Phase 2 adds the AUDIO FOUNDATION only:
//   * Live capture: host audio from the FL Studio Mixer Track is
//     recorded into a preallocated 60 s buffer (AudioCaptureEngine).
//   * Import: WAV / AIFF / FLAC (+ MP3 when JUCE enables it) files
//     are decoded on a worker thread (AudioFileLoader).
//   * Preview: the active AudioClip plays back through the Mixer
//     Track, resampled to the host rate (AudioPreviewEngine).
//   * Input meter: real RMS / peak levels from the incoming signal.
//   * Status: READY / LOADING / RECORDING / AUDIO READY / PLAYING /
//     STOPPED / NO INPUT / ERROR / LIMIT REACHED / EMPTY SOURCE.
//
// What is still NOT here (Phase 3+): pitch detection, note
// segmentation, MIDI generation, instrument playback, voice
// multiplication, choir generation, AI. ANALYZE stays non-functional.
//
// Thread safety: processBlock() never allocates, never locks, never
// touches files or the UI, and never changes shared_ptr ownership. The
// immutable active clip is owned on the message thread. AudioPreviewEngine
// hands a stable raw pointer to the audio callback and defers ownership
// changes until any in-flight render finishes. Recording, playback and
// file loading use atomics or explicit message-thread handoffs.
//
// The AudioProcessorValueTreeState holds the interface parameters so
// the editor state persists in the host project. The audio recording
// itself is NOT serialized into plugin state: a project reload
// without its temporary clip returns safely to READY / EMPTY SOURCE.

#include <JuceHeader.h>

#include <atomic>
#include <memory>

#include "Audio/AudioCaptureEngine.h"
#include "Audio/AudioFileLoader.h"
#include "Audio/AudioPreviewEngine.h"

class MelotheonAudioProcessor : public juce::AudioProcessor
{
public:
    enum class Status
    {
        Ready = 0,
        Loading,
        Recording,
        AudioReady,
        Playing,
        Stopped,
        NoInput,
        Error,
        LimitReached,
        EmptySource
    };

    enum class InputSource
    {
        Live = 0,      // capture host audio with Record
        Imported = 1   // play the loaded file / recorded clip
    };

    MelotheonAudioProcessor();
    ~MelotheonAudioProcessor() override;

    // Un-hide the base-class double-precision processBlock overload
    // (this class overrides the float version only; without this GCC
    // warns -Woverloaded-virtual).
    using juce::AudioProcessor::processBlock;

    // AudioProcessor boilerplate (JUCE 8).
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Melotheon"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    // ---- Phase 2 public API (editor + tests) ----

    InputSource getInputSource() const;

    Status getStatus() const
    {
        return static_cast<Status> (status.load (std::memory_order_acquire));
    }

    void setStatus (Status newStatus)
    {
        status.store (static_cast<int> (newStatus), std::memory_order_release);
    }

    // Short LCD text for the current status. Message thread only
    // (reads lastErrorMessage). For Error this is the error message.
    juce::String getStatusText() const;

    // Message thread only. The audio callback does not access this owner.
    std::shared_ptr<const AudioClip> getActiveClip() const { return activeClip; }

    bool hasActiveClip() const
    {
        auto clip = getActiveClip();
        return clip != nullptr && ! clip->isEmpty();
    }

    // Increments every time a new clip becomes active. The editor
    // polls it to know when to refresh the waveform.
    int getClipVersion() const { return clipVersion.load (std::memory_order_acquire); }

    // Smoothed input levels in dB, range -60..0. The editor reads
    // them ~30x/second for the meter.
    float getInputPeakDb() const { return inputPeakDb.load (std::memory_order_acquire); }
    float getInputRmsDb() const { return inputRmsDb.load (std::memory_order_acquire); }

    // Recording (message thread). startRecording() returns false
    // when the input source is not LIVE INPUT (status becomes Error
    // with "USE LIVE INPUT") or a take is already running.
    bool startRecording();
    // Stop and finalize the take into the active clip (AUDIO READY).
    // No-op when nothing is recording.
    void stopRecording();
    // Finalize a take that hit the 60 s limit (status LIMIT REACHED).
    // No-op when no limit was reached. Message thread only.
    void finalizeRecordingFromLimit();
    bool isRecording() const { return captureEngine.isRecording(); }

    // Preview (message thread). startPreview() returns false when
    // there is no clip (status becomes EMPTY SOURCE). Play starts
    // from the current internal preview position.
    bool startPreview();
    void stopPreview();  // stop + cursor to the beginning
    void pausePreview(); // stop + keep the cursor
    bool isPreviewPlaying() const { return previewEngine.isPlaying(); }
    // True once after playback reached the end of the clip.
    bool consumePlaybackEnded() { return previewEngine.consumePlaybackEnded(); }
    double getPreviewPositionSeconds() const { return previewEngine.getPositionSeconds(); }
    double getPreviewDurationSeconds() const { return previewEngine.getDurationSeconds(); }

    // Import a file (message thread): sets LOADING and decodes on a
    // worker thread; the result installs the clip or sets ERROR.
    void startFileImport (const juce::File& file);

    // Keep the PLAY ORIGINAL toggle parameter in sync with preview
    // state. Message thread only.
    void setPlayOriginalParameter (bool shouldPlay);

    // Interface parameters (editor state; persisted in the host).
    juce::AudioProcessorValueTreeState apvts;

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Install a finished clip as the active one (message thread).
    void installClip (std::shared_ptr<const AudioClip> clip, Status statusAfter);

    // Stop capture, wait for the last audio-thread copy and publish a
    // non-empty take. Returns false for no active take or zero samples.
    bool finalizeCapture (Status statusAfter);

    // Audio-thread helpers (processBlock only).
    void updateInputMeter (const juce::AudioBuffer<float>& buffer);
    void updateSilenceStatus (int numSamples);

    AudioCaptureEngine captureEngine;
    AudioPreviewEngine previewEngine;
    AudioFileLoader fileLoader;

    std::shared_ptr<const AudioClip> activeClip; // message-thread owner
    std::atomic<int> status { static_cast<int> (Status::Ready) };
    std::atomic<int> clipVersion { 0 };
    std::atomic<float> inputPeakDb { -60.0f };
    std::atomic<float> inputRmsDb { -60.0f };
    std::atomic<double> hostSampleRate { 44100.0 };

    // Set to false in the destructor so late file-load callbacks
    // never touch a destroyed processor.
    std::shared_ptr<std::atomic<bool>> aliveFlag
        = std::make_shared<std::atomic<bool>> (true);

    // Message thread only: short error text for the Error status.
    juce::String lastErrorMessage;

    // Audio-thread-only meter state (written in processBlock only,
    // reset in prepareToPlay while the callback is suspended).
    float meterPeakLinear = 0.0f;
    float meterRmsLinear = 0.0f;
    int silentSamples = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MelotheonAudioProcessor)
};
