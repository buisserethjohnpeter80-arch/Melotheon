// MelotheonAudioProcessor implementation (Phase 3B/3C).
// The Phase 2 audio foundation remains real-time safe. Pitch analysis
// and sample-instrument rendering run on workers; MIDI export runs only
// from an explicit message-thread button action.

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "Analysis/NoteSegmenter.h"
#include "Midi/MidiExporter.h"
#include "Project/ProjectState.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

namespace
{
// In-flight announcement for the audition voice. The message thread
// waits for the counter to drain before replacing the sample buffer.
// Same pattern as AudioPreviewEngine's RenderActivityGuard.
class ActivityGuard
{
public:
    explicit ActivityGuard (std::atomic<int>& counterToUse)
        : counter (counterToUse)
    {
        counter.fetch_add (1, std::memory_order_seq_cst);
    }

    ~ActivityGuard()
    {
        counter.fetch_sub (1, std::memory_order_seq_cst);
    }

private:
    std::atomic<int>& counter;
};

// The MIDI snapshot reader count is claimed before the slot is inspected,
// so this guard only has to release an already-acquired reader on every
// return path from renderMidiOut().
class ActivityReleaseGuard
{
public:
    explicit ActivityReleaseGuard (std::atomic<int>& counterToUse)
        : counter (counterToUse)
    {
    }

    ~ActivityReleaseGuard()
    {
        counter.fetch_sub (1, std::memory_order_seq_cst);
    }

private:
    std::atomic<int>& counter;
};

juce::String midiNoteName (int midiNote)
{
    static const char* names[] = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    midiNote = juce::jlimit (0, 127, midiNote);
    return juce::String (names[midiNote % 12]) + juce::String (midiNote / 12 - 1);
}
} // namespace

MelotheonAudioProcessor::MelotheonAudioProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "MelotheonState", createParameterLayout())
{
    // Cache the MIDI-out parameter atoms for the audio thread (see
    // PluginProcessor.h): getRawParameterValue() does a map lookup,
    // so it must never run inside processBlock().
    midiOutParam = apvts.getRawParameterValue ("midiOut");
    sendToChannelParam = apvts.getRawParameterValue ("sendToChannel");
    midiChannelParam = apvts.getRawParameterValue ("midiChannel");
    noiseGateParam = apvts.getRawParameterValue ("noiseGate");

    // P2-8: turning SEND TO CHANNEL on auto-enables the MIDI OUT
    // master (see parameterChanged).
    apvts.addParameterListener ("sendToChannel", this);
    // Exclusive input source: changing the source clears the old
    // clip so the two sources can never mix (see parameterChanged).
    apvts.addParameterListener ("inputSource", this);
    lastAppliedInputSource = getInputSource();
}

MelotheonAudioProcessor::~MelotheonAudioProcessor()
{
    apvts.removeParameterListener ("sendToChannel", this);
    apvts.removeParameterListener ("inputSource", this);
    aliveFlag->store (false, std::memory_order_release);
    pitchAnalysisEngine.cancel();
    sampleRenderEngine.cancel();
}

juce::AudioProcessorValueTreeState::ParameterLayout
MelotheonAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    // Bottom correction strip knobs (0.0 - 1.0).
    layout.add (std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "center", 1 }, "Center",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.5f));
    layout.add (std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "variation", 1 }, "Variation",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.5f));
    layout.add (std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "transition", 1 }, "Transition",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.5f));
    layout.add (std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "sensitivity", 1 }, "Sensitivity",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.65f));
    layout.add (std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "quantize", 1 }, "Quantize",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.8f));

    // Left capture panel: noise gate knob + mono toggle. Pitch analysis uses
    // the gate as the offline pitch detector's silence threshold; it
    // never gates or modifies the preview audio.
    layout.add (std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "noiseGate", 1 }, "Noise Gate",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.3f));
    layout.add (std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "mono", 1 }, "Mono", false));

    // Right instrument panel: MIDI out toggle, play buttons.
    layout.add (std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "midiOut", 1 }, "MIDI Out", false));
    // SEND TO CHANNEL: the per-melody switch for live MIDI out.
    // The MIDI OUT toggle is the master switch; both must be on
    // for the melody to stream to the host (see renderMidiOut).
    layout.add (std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "sendToChannel", 1 }, "Send To Channel", false));
    layout.add (std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "playOriginal", 1 }, "Play Original", false));
    layout.add (std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { "playInstrument", 1 }, "Play Instrument", false));

    // Key / scale (shown in the bottom strip as "KEY: C MINOR").
    layout.add (std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { "key", 1 }, "Key",
        juce::StringArray { "C", "C#", "D", "D#", "E", "F",
                            "F#", "G", "G#", "A", "A#", "B" },
        0));
    layout.add (std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { "scale", 1 }, "Scale",
        juce::StringArray { "Major", "Minor", "Dorian", "Mixolydian",
                            "Phrygian", "Lydian" },
        1));

    // Header input source: Phase 2 has exactly two sources.
    // LIVE INPUT captures host audio; IMPORTED AUDIO plays the loaded
    // file / recorded clip. (Phase 1 had VOICE / GUITAR / PIANO /
    // LINE here; those were labels only, never audio routings.)
    // Default is IMPORTED AUDIO: on an empty track Live Input is
    // not active by default — the user imports, switches to LIVE
    // INPUT to record, or lets voice auto-detection lock LIVE when
    // a voice is actually heard on the track (see pollVoiceDetection).
    // Old saved projects keep their saved value (APVTS state restore).
    layout.add (std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { "inputSource", 1 }, "Input Source",
        juce::StringArray { "LIVE INPUT", "IMPORTED AUDIO" },
        1));
    layout.add (std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { "midiChannel", 1 }, "MIDI Channel",
        juce::StringArray { "CHANNEL 1", "CHANNEL 2", "CHANNEL 3",
                            "CHANNEL 4", "CHANNEL 5", "CHANNEL 6",
                            "CHANNEL 7", "CHANNEL 8", "CHANNEL 9",
                            "CHANNEL 10", "CHANNEL 11", "CHANNEL 12",
                            "CHANNEL 13", "CHANNEL 14", "CHANNEL 15",
                            "CHANNEL 16" },
        0));

    return layout;
}

void MelotheonAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused (samplesPerBlock);

    hostSampleRate.store (sampleRate > 0.0 ? sampleRate : 44100.0,
                          std::memory_order_release);

    const int numChannels = juce::jlimit (1, 2, getTotalNumInputChannels());
    captureEngine.prepare (hostSampleRate.load (std::memory_order_acquire), numChannels);
    previewEngine.stop();
    instrumentPreviewEngine.stop();

    // Fresh audition voice state (the callback is suspended here).
    auditionNote.store (-1, std::memory_order_release);
    auditionGainAtomic.store (0.0f, std::memory_order_release);
    auditionPhase = 0.0;
    auditionSamplePos = 0.0;
    auditionCurrentNote = -1;
    auditionLastGeneration = 0;
    auditionLatchedSampleData = nullptr;
    auditionLatchedSampleLength = 0;
    auditionLatchedStep = 1.0;

    // Fresh live MIDI-out state (the callback is suspended here).
    resetMidiOutState();

    meterPeakLinear = 0.0f;
    meterRmsLinear = 0.0f;
    silentSamples = 0;

    // Fresh voice-detection accounting for the new stream (the
    // detected flag itself is sticky for the session: once a voice
    // was heard, it stays heard — only the lock application can
    // still be pending, and it re-detects within 0.5 s of voice
    // if a re-prepare ever dropped the accumulator mid-detection).
    voiceDetectVoicedSeconds = 0.0;
    voiceDetectSilentSeconds = 0.0;

    // A re-prepare aborts any take in progress (its partial capture
    // is discarded by captureEngine.prepare()).
    if (getStatus() == Status::Recording)
        setStatus (Status::Ready);
}

void MelotheonAudioProcessor::releaseResources()
{
    stopAudition();
    previewEngine.stop();
    instrumentPreviewEngine.stop();
    captureEngine.stopRecording(); // discard any unfinalized take

    // The host owns all-notes-off on suspend; only the local state
    // resets here, so a later resume re-syncs cleanly.
    resetMidiOutState();
}

bool MelotheonAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& mainOut = layouts.getMainOutputChannelSet();
    if (mainOut != juce::AudioChannelSet::mono() && mainOut != juce::AudioChannelSet::stereo())
        return false;
    if (const auto& mainIn = layouts.getMainInputChannelSet();
        mainIn != juce::AudioChannelSet::mono() && mainIn != juce::AudioChannelSet::stereo())
        return false;
    return true;
}

void MelotheonAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    midiMessages.clear();

    // Host tempo (BPM / time signature / PPQ) for the piano-roll
    // grid and MIDI export. Reading the playhead here is the
    // standard RT-safe pattern (small value copy, no allocation);
    // refreshHostTempo only stores atomics and bumps the version
    // when the tempo actually changed.
    refreshHostTempo();

    // The preview position this block starts at, captured before any
    // render advances it: the live MIDI-out scheduler below needs the
    // block's exact time range.
    const double midiBlockStartSeconds = getDisplayPreviewPositionSeconds();

    // Live MIDI out needs the transport state as it was at block
    // start: whether the switches were on, whether a preview was
    // playing, and when the active clip ends. The preview render
    // below may stop playback mid-block (natural end), and
    // renderMidiOut must still emit the events up to the clip end
    // instead of cutting them at sample 0.
    const bool midiSwitchesOn = midiOutEnabled() && sendToChannelEnabled();
    const bool midiInstrumentWasPlaying = instrumentPreviewEngine.isPlaying();
    const bool midiWasPlayingAtBlockStart = midiInstrumentWasPlaying
        || previewEngine.isPlaying();
    const double midiActiveClipEndSeconds = midiInstrumentWasPlaying
        ? instrumentPreviewEngine.getDurationSeconds()
        : previewEngine.getDurationSeconds();

    // 1. Input meter reads the incoming host audio (before any
    // preview overwrites the buffer).
    updateInputMeter (buffer);

    // 2. Live capture (copies input, no allocation).
    captureEngine.processInput (buffer);
    if (captureEngine.limitReached())
        setStatus (Status::LimitReached);

    // 3. One preview owns the output at a time. Instrument playback has
    // priority only while explicitly active; otherwise original preview or
    // normal Mixer Track pass-through remains unchanged.
    if (instrumentPreviewEngine.isPlaying())
        instrumentPreviewEngine.render (
            buffer, hostSampleRate.load (std::memory_order_acquire));
    else if (previewEngine.isPlaying())
        previewEngine.render (buffer, hostSampleRate.load (std::memory_order_acquire));

    // 4. Keyboard audition mixes the held piano key on top (additive:
    // preview keeps playing under it). No-op when no key is held.
    renderAudition (buffer, hostSampleRate.load (std::memory_order_acquire));

    // 5. Live MIDI out (SEND TO CHANNEL): when MIDI OUT + SEND TO
    // CHANNEL are on and a preview is playing, the current melody
    // streams as note-on/note-off to the host, in time with the
    // playhead. No-op otherwise (and it sends note-offs for every
    // sounding note when sending stops). The transport state was
    // captured before the preview render above, so a preview that
    // ended naturally inside this block still streams its events
    // up to the exact clip-end sample.
    renderMidiOut (midiMessages, midiBlockStartSeconds, buffer.getNumSamples(),
                   hostSampleRate.load (std::memory_order_acquire),
                   midiSwitchesOn, midiWasPlayingAtBlockStart, midiActiveClipEndSeconds);

    // 6. READY <-> NO INPUT transitions (only from those states).
    updateSilenceStatus (buffer.getNumSamples());
}

// ---- Audio + melody public API ----

MelotheonAudioProcessor::InputSource MelotheonAudioProcessor::getInputSource() const
{
    const float value = apvts.getRawParameterValue ("inputSource")->load();
    return static_cast<int> (value + 0.5f) == 0 ? InputSource::Live : InputSource::Imported;
}

juce::String MelotheonAudioProcessor::getStatusText() const
{
    switch (getStatus())
    {
        case Status::Ready:        return "READY";
        case Status::Loading:      return "LOADING";
        case Status::Recording:    return "RECORDING";
        case Status::AudioReady:   return "AUDIO READY";
        case Status::Playing:      return "PLAYING";
        case Status::Stopped:      return "STOPPED";
        case Status::NoInput:      return "NO INPUT";
        case Status::Error:        return lastErrorMessage.isNotEmpty() ? lastErrorMessage : "ERROR";
        case Status::LimitReached: return "LIMIT REACHED";
        case Status::EmptySource:  return "EMPTY SOURCE";
        case Status::Analyzing:
            return "ANALYZING " + juce::String (juce::roundToInt (
                pitchAnalysisEngine.getProgress() * 100.0f)) + "%";
        case Status::AnalysisReady: return "PITCH READY";
        case Status::NoPitch:       return "NO STABLE PITCH";
        case Status::LoadingSound:  return "LOADING SOUND";
        case Status::SoundReady:    return "SOUND READY";
        case Status::RenderingInstrument:
            return "RENDERING " + juce::String (juce::roundToInt (
                sampleRenderEngine.getProgress() * 100.0f)) + "%";
        case Status::InstrumentReady:   return "INSTRUMENT READY";
        case Status::PlayingInstrument:return "PLAYING INSTRUMENT";
        case Status::MidiExported:      return "MIDI EXPORTED";
    }
    return "READY";
}

bool MelotheonAudioProcessor::startRecording()
{
    if (getInputSource() != InputSource::Live)
    {
        lastErrorMessage = "USE LIVE INPUT";
        setStatus (Status::Error);
        return false;
    }

    // A new live take owns the transport. Invalidate any pending file
    // import and stop preview before arming capture.
    fileLoader.cancelPendingLoad();
    previewEngine.stop();
    instrumentPreviewEngine.stop();
    setPlayOriginalParameter (false);
    setPlayInstrumentParameter (false);
    clearPitchAnalysis();

    if (! captureEngine.startRecording())
        return false;

    // Pressing REC is a source decision: voice auto-detection must
    // never override it later.
    userChoseSource = true;

    setStatus (Status::Recording);
    return true;
}

void MelotheonAudioProcessor::stopRecording()
{
    finalizeCapture (Status::AudioReady);
}

void MelotheonAudioProcessor::finalizeRecordingFromLimit()
{
    if (! captureEngine.limitReached())
        return;

    finalizeCapture (Status::LimitReached);
}

bool MelotheonAudioProcessor::startPreview()
{
    // Keep transport state deterministic: preview does not interrupt
    // an analysis that the user explicitly started.
    if (pitchAnalysisEngine.isRunning())
        return false;

    // Play and Record are mutually exclusive. Pressing Play while a
    // take is active finalizes that take first, then previews it.
    if (captureEngine.isRecording() || captureEngine.limitReached())
    {
        if (! finalizeCapture (Status::AudioReady))
            return false;
    }

    // Do not let an older asynchronous import replace the clip while
    // the user is previewing it.
    fileLoader.cancelPendingLoad();
    instrumentPreviewEngine.stop();
    setPlayInstrumentParameter (false);

    if (! hasActiveClip())
    {
        setStatus (Status::EmptySource);
        return false;
    }
    if (previewEngine.start())
    {
        setStatus (Status::Playing);
        return true;
    }

    setStatus (Status::EmptySource);
    return false;
}

void MelotheonAudioProcessor::stopPreview()
{
    previewEngine.stop();
    instrumentPreviewEngine.stop();
    setPlayOriginalParameter (false);
    setPlayInstrumentParameter (false);
    setStatus (Status::Stopped);
}

void MelotheonAudioProcessor::pausePreview()
{
    previewEngine.pause();
    setStatus (Status::Stopped);
}

void MelotheonAudioProcessor::setPreviewPositionSeconds (double seconds)
{
    previewEngine.setPositionSeconds (seconds);
    if (instrumentPreviewEngine.isPlaying())
        instrumentPreviewEngine.setPositionSeconds (seconds);
}

juce::File MelotheonAudioProcessor::prepareMidiDragFile()
{
    juce::File tempFile = juce::File::getSpecialLocation (
        juce::File::tempDirectory).getChildFile ("Melotheon Melody.mid");
    if (! exportMidiToFile (tempFile))
        return {};
    return tempFile;
}

// ---- Keyboard audition ----

void MelotheonAudioProcessor::startAudition (int midiNote)
{
    const int note = juce::jlimit (0, 127, midiNote);
    const double hostRate = hostSampleRate.load (std::memory_order_acquire);
    const double sourceRate = auditionSampleRate.load (std::memory_order_acquire);
    const int root = instrumentRootMidi.load (std::memory_order_acquire);
    const bool hasSample = auditionSampleData.load (std::memory_order_acquire) != nullptr
                        && auditionSampleLength.load (std::memory_order_acquire) > 1;
    // Pitch the one-shot from its detected root; the sine fallback
    // needs no step (its frequency comes from the note itself).
    const double step = hasSample
        ? std::pow (2.0, static_cast<double> (note - root) / 12.0)
            * (sourceRate / (hostRate > 0.0 ? hostRate : 44100.0))
        : 1.0;
    auditionPlaybackStep.store (step, std::memory_order_release);
    auditionGeneration.fetch_add (1, std::memory_order_acq_rel);
    auditionNote.store (note, std::memory_order_release);
}

void MelotheonAudioProcessor::stopAudition()
{
    auditionNote.store (-1, std::memory_order_release);
}

void MelotheonAudioProcessor::installAuditionSample (
    std::shared_ptr<const AudioClip> clip, int rootMidi)
{
    stopAudition();
    waitForAuditionToFinish();

    instrumentRootMidi.store (
        juce::jlimit (0, 127, rootMidi), std::memory_order_release);
    if (clip == nullptr || clip->isEmpty())
    {
        auditionSampleBuffer.setSize (1, 0, false, false, true);
        auditionSampleRate.store (44100.0, std::memory_order_release);
        auditionSampleData.store (nullptr, std::memory_order_release);
        auditionSampleLength.store (0, std::memory_order_release);
    }
    else
    {
        const auto& mono = clip->analysisMonoBuffer;
        auditionSampleBuffer.setSize (1, mono.getNumSamples(), false, false, true);
        auditionSampleBuffer.copyFrom (0, 0, mono, 0, 0, mono.getNumSamples());
        auditionSampleRate.store (
            clip->sourceSampleRate > 0.0 ? clip->sourceSampleRate : 44100.0,
            std::memory_order_release);
        auditionSampleData.store (
            auditionSampleBuffer.getReadPointer (0), std::memory_order_release);
        auditionSampleLength.store (mono.getNumSamples(), std::memory_order_release);
    }
    auditionGeneration.fetch_add (1, std::memory_order_acq_rel);
}

void MelotheonAudioProcessor::waitForAuditionToFinish()
{
    // Bounded: the audio thread may be stopped (host idle), in which
    // case nothing can read the buffer and waiting is pointless.
    for (int i = 0; i < 2000; ++i)
    {
        if (auditionRenderersInFlight.load (std::memory_order_seq_cst) == 0
            && auditionGainAtomic.load (std::memory_order_acquire) < 0.0005f)
            return;
        std::this_thread::sleep_for (std::chrono::microseconds (50));
    }
}

void MelotheonAudioProcessor::renderAudition (juce::AudioBuffer<float>& buffer,
                                              double hostRate)
{
    const int note = auditionNote.load (std::memory_order_acquire);
    float gain = auditionGainAtomic.load (std::memory_order_acquire);
    if (note < 0 && gain < 0.0005f)
        return; // idle: no key held and the release tail is silent
    if (hostRate <= 0.0)
        return;

    ActivityGuard inFlightGuard (auditionRenderersInFlight);

    // A new generation (new sample or new note) re-latches the source
    // data, step and sustain-loop points and restarts the voice.
    const std::uint64_t generation = auditionGeneration.load (std::memory_order_acquire);
    if (generation != auditionLastGeneration)
    {
        auditionLastGeneration = generation;
        auditionLatchedSampleData = auditionSampleData.load (std::memory_order_acquire);
        auditionLatchedSampleLength = auditionSampleLength.load (std::memory_order_acquire);
        auditionLatchedStep = auditionPlaybackStep.load (std::memory_order_acquire);
        const int length = auditionLatchedSampleLength;
        const double sourceRate = auditionSampleRate.load (std::memory_order_acquire);
        auditionLatchedLoopStart = juce::jlimit (
            0, std::max (0, length - 2),
            static_cast<int> (sourceRate * 0.05));
        const int loopEnd = std::max (auditionLatchedLoopStart + 1,
            length - std::min (length / 8,
                               static_cast<int> (sourceRate * 0.03)));
        auditionLatchedLoopLength = std::max (1, loopEnd - auditionLatchedLoopStart);
        auditionSamplePos = 0.0;
        auditionPhase = 0.0;
    }
    if (note != auditionCurrentNote)
    {
        auditionCurrentNote = note;
        auditionSamplePos = 0.0;
        auditionPhase = 0.0;
    }

    const bool useSample = auditionLatchedSampleData != nullptr
                        && auditionLatchedSampleLength > 1;
    const float targetGain = note >= 0 ? (useSample ? 0.64f : 0.40f) : 0.0f;
    // One-pole envelope: ~5 ms attack, ~40 ms release.
    const double tau = targetGain > gain ? 0.005 : 0.040;
    const float coeff = static_cast<float> (
        1.0 - std::exp (-1.0 / (tau * hostRate)));

    const double frequencyHz = note >= 0
        ? 440.0 * std::pow (2.0, (static_cast<double> (note) - 69.0) / 12.0) : 0.0;
    const double phaseStep = 2.0 * juce::MathConstants<double>::pi
                           * frequencyHz / hostRate;
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    const float* sampleData = auditionLatchedSampleData;
    const int sampleLength = auditionLatchedSampleLength;

    for (int n = 0; n < numSamples; ++n)
    {
        gain += (targetGain - gain) * coeff;
        if (gain < 0.0005f && targetGain <= 0.0f)
            break; // release tail finished; the rest stays silent

        float sample = 0.0f;
        if (note >= 0)
        {
            if (useSample)
            {
                // Linear-interpolated resample; past the end, loop the
                // body (same sustain loop as the offline renderer) so
                // a held key keeps sounding.
                double position = auditionSamplePos;
                if (position >= static_cast<double> (sampleLength - 1))
                {
                    position = static_cast<double> (auditionLatchedLoopStart)
                        + std::fmod (position - auditionLatchedLoopStart,
                                     static_cast<double> (auditionLatchedLoopLength));
                }
                const int index0 = static_cast<int> (position);
                const int index1 = std::min (index0 + 1, sampleLength - 1);
                const float fraction = static_cast<float> (
                    position - static_cast<double> (index0));
                sample = sampleData[index0]
                       + (sampleData[index1] - sampleData[index0]) * fraction;
                auditionSamplePos = position + auditionLatchedStep;
            }
            else
            {
                auditionPhase += phaseStep;
                if (auditionPhase >= 2.0 * juce::MathConstants<double>::pi)
                    auditionPhase -= 2.0 * juce::MathConstants<double>::pi;
                sample = static_cast<float> (std::sin (auditionPhase));
            }
        }

        const float out = sample * gain;
        for (int ch = 0; ch < numChannels; ++ch)
            buffer.addSample (ch, n, out);
    }

    auditionGainAtomic.store (gain, std::memory_order_release);
}

// ---- Live MIDI out (SEND TO CHANNEL) ----

void MelotheonAudioProcessor::publishMelodyNotes (
    std::shared_ptr<const MelodyNoteSequence> notes)
{
    melodyNotes = std::move (notes);
    publishMidiOutSequence();
    melodyNotesVersion.fetch_add (1, std::memory_order_acq_rel);
}

size_t MelotheonAudioProcessor::midiOutCapacityBytesForSequence (size_t numNotes)
{
    // Worst case per block, derived from renderMidiOut:
    //  * one effective all-notes-off pass: at most one note-off per
    //    MIDI note number (128). Later all-notes-off calls in the
    //    same block emit nothing — the first one clears the sounding
    //    set.
    //  * per note in the sequence: at most one note-on, one note-off,
    //    and one defensive re-trigger note-off (noteOnMidiOut stops
    //    a note that is already sounding before re-triggering it).
    //    The defensive off is unreachable while notes are sorted and
    //    non-overlapping, but the bound does not rely on that.
    // Each event costs 3 bytes of MIDI data + 4 bytes of sample
    // position + 2 bytes of length (see juce_MidiBuffer.cpp:
    // addEvent stores numBytes + sizeof (int32) + sizeof (uint16)).
    // Practical ceiling: a 60 s clip with the shortest editable note
    // (0.03 s) holds ~2000 notes, so the bound stays near 55 KB.
    return (3 * numNotes + static_cast<size_t> (midiOutMaxAllNotesOffEvents))
        * static_cast<size_t> (midiOutBytesPerEvent);
}

int MelotheonAudioProcessor::findFreeMidiOutSlot (int publishedSlot, const int* readerCounts)
{
    for (int slot = 0; slot < midiOutSnapshotSlotCount; ++slot)
    {
        if (slot == publishedSlot)
            continue; // never write the slot the audio thread reads
        if (readerCounts[slot] == 0)
            return slot;
    }
    return -1; // no free slot (unreachable with 3 slots — see the invariant)
}

void MelotheonAudioProcessor::publishMidiOutSequence()
{
    // Three-slot publication, never blocking (P3-14): the audio
    // thread holds at most one slot at a time and a non-published
    // slot's reader count can only fall (only the published slot is
    // ever claimed), so among three slots there is always one that
    // is neither published nor held — and a slot read as free stays
    // free until this publish writes it, because the audio thread
    // can only claim the published slot. No wait, no spin, no sleep.
    const int currentSlot = midiOutPublishedSnapshot.load (std::memory_order_acquire);
    int readerCounts[midiOutSnapshotSlotCount] = {};
    for (int slot = 0; slot < midiOutSnapshotSlotCount; ++slot)
        readerCounts[slot] = midiOutSnapshots[slot].readers.load (std::memory_order_seq_cst);

    const int freeSlot = findFreeMidiOutSlot (currentSlot, readerCounts);
    if (freeSlot < 0)
    {
        // Unreachable by the invariant above. Dropping this publish
        // keeps the previous melody streaming (no spin, no torn
        // read); the next publish succeeds once the held slot drains,
        // which takes at most one audio block.
        jassertfalse;
        return;
    }

    auto& snapshot = midiOutSnapshots[freeSlot];
    snapshot.owner = melodyNotes;
    snapshot.generation = ++midiOutNextGeneration;

    // Publishes both owner and generation as one immutable snapshot.
    midiOutPublishedSnapshot.store (freeSlot, std::memory_order_release);
}

void MelotheonAudioProcessor::resetMidiOutState()
{
    midiOutLatchedGeneration = 0;
    midiOutLastChannel = -1;
    midiOutLastPosition = -1.0;
    midiOutStatus.store (static_cast<int> (MidiOutStatus::Off),
                         std::memory_order_release);
    for (int i = 0; i < 128; ++i)
    {
        midiOutNoteOn[i] = false;
        midiOutNoteChannel[i] = 0;
    }
}

void MelotheonAudioProcessor::noteOnMidiOut (juce::MidiBuffer& midiMessages,
                                            int sampleOffset,
                                            const MelodyNote& note, int channel)
{
    const int midiNote = juce::jlimit (0, 127, note.midiNote);
    // Same velocity mapping as the exported .mid file, so the live
    // stream and the file sound identical.
    const auto velocity = static_cast<juce::uint8> (std::clamp (
        static_cast<int> (std::lround (note.velocity * 126.0f)), 1, 127));

    // A note already sounding (e.g. re-triggered on another channel)
    // is stopped first, so a channel change mid-note can never leave
    // a stuck note.
    if (midiOutNoteOn[midiNote])
        noteOffMidiOut (midiMessages, sampleOffset, note);

    midiMessages.addEvent (juce::MidiMessage::noteOn (
        channel + 1, midiNote, velocity), sampleOffset);
    midiOutNoteOn[midiNote] = true;
    midiOutNoteChannel[midiNote] = channel;
}

void MelotheonAudioProcessor::noteOffMidiOut (juce::MidiBuffer& midiMessages,
                                             int sampleOffset,
                                             const MelodyNote& note)
{
    const int midiNote = juce::jlimit (0, 127, note.midiNote);
    if (! midiOutNoteOn[midiNote])
        return; // never send a note-off for a note that is not sounding

    // The note-off goes to the channel the note-on was sent on, even
    // if the user changed the channel dropdown since.
    midiMessages.addEvent (juce::MidiMessage::noteOff (
        midiOutNoteChannel[midiNote] + 1, midiNote), sampleOffset);
    midiOutNoteOn[midiNote] = false;
}

void MelotheonAudioProcessor::allNotesOffMidiOut (juce::MidiBuffer& midiMessages,
                                                 int sampleOffset)
{
    for (int midiNote = 0; midiNote < 128; ++midiNote)
    {
        if (! midiOutNoteOn[midiNote])
            continue;
        midiMessages.addEvent (juce::MidiMessage::noteOff (
            midiOutNoteChannel[midiNote] + 1, midiNote), sampleOffset);
        midiOutNoteOn[midiNote] = false;
    }
}

void MelotheonAudioProcessor::renderMidiOut (juce::MidiBuffer& midiMessages,
                                            double blockStartSeconds,
                                            int numSamples, double hostRate,
                                            bool midiSwitchesOn, bool wasPlayingAtBlockStart,
                                            double activeClipEndSeconds)
{
    const double blockEndSeconds = blockStartSeconds
        + static_cast<double> (numSamples) / hostRate;

    // The MIDI OUT toggle is the master switch and SEND TO CHANNEL is
    // the per-melody switch: both must be on, and a preview (or
    // instrument preview) must be playing, for any MIDI to leave.
    // wasPlayingAtBlockStart / activeClipEndSeconds were captured in
    // processBlock() before the preview render. sendingNow reads the
    // switches live, so a toggle takes effect in the same block as
    // before; wasSending uses the block-start state, so a natural
    // end is never confused with a toggle.
    const bool playingNow = previewEngine.isPlaying() || instrumentPreviewEngine.isPlaying();
    const bool sendingNow = midiOutEnabled() && sendToChannelEnabled() && playingNow;
    const bool wasSending = midiSwitchesOn && wasPlayingAtBlockStart;

    // The preview ended naturally during this block: it was playing
    // at block start, is stopped now, and the clip end falls inside
    // this block's time range. The MIDI range this block covers is
    // [blockStart, clipEnd]. A manual stop before the block does not
    // take this path: the position was already reset, so the clip
    // end is not inside the range (and wasPlayingAtBlockStart is
    // false).
    const bool endedNaturally = wasSending && ! sendingNow
        && activeClipEndSeconds > blockStartSeconds
        && activeClipEndSeconds <= blockEndSeconds;

    // Live MIDI-out status for the MIDI panel (P2-10): Off when the
    // path is off (master off, or SEND off), Waiting when both
    // switches are on but nothing is playing, Streaming while the
    // melody is actually streaming. Written every block, before any
    // early return, so the panel never shows a stale state. The Off
    // decision reads the switches live (like sendingNow), not the
    // block-start snapshot: a toggle mid-block must show Off in
    // this block, not one block late as Waiting.
    midiOutStatus.store (static_cast<int> (
        ! (midiOutEnabled() && sendToChannelEnabled()) ? MidiOutStatus::Off
        : (sendingNow ? MidiOutStatus::Streaming : MidiOutStatus::Waiting)),
        std::memory_order_release);

    // Claim a stable immutable snapshot FIRST, before any MIDI event
    // is written: the claim tells us the sequence size, so one
    // ensureSize() call below can pre-allocate every event this
    // block can emit (P3-13). A publication may happen between the
    // first index load and the reader increment; the second load
    // detects that race and retries before any slot data is
    // inspected. Once claimed, the message thread cannot reuse this
    // slot until the guard releases it.
    int snapshotIndex = 0;
    for (;;)
    {
        snapshotIndex = midiOutPublishedSnapshot.load (std::memory_order_acquire);
        auto& readers = midiOutSnapshots[snapshotIndex].readers;
        readers.fetch_add (1, std::memory_order_seq_cst);
        if (snapshotIndex == midiOutPublishedSnapshot.load (std::memory_order_acquire))
            break;
        readers.fetch_sub (1, std::memory_order_seq_cst);
    }

    auto& snapshot = midiOutSnapshots[snapshotIndex];
    ActivityReleaseGuard snapshotGuard (snapshot.readers);
    const auto* sequence = snapshot.owner.get();
    const std::uint64_t generation = snapshot.generation;

    // Pre-allocate the MIDI buffer once (P3-13): without this,
    // addEvent() grows the buffer's heap block on demand while
    // events are written. The bound covers every event this block
    // can emit (see midiOutCapacityBytesForSequence); when the
    // host's buffer already has the capacity this is a no-op. One
    // call per block is the best achievable through JUCE's API: the
    // host hands processBlock a MidiBuffer whose capacity cannot be
    // carried across blocks, and there is no way to write events
    // without going through addEvent().
    midiMessages.ensureSize (midiOutCapacityBytesForSequence (
        sequence != nullptr ? sequence->notes.size() : 0));

    if (! sendingNow && ! endedNaturally)
    {
        // Was sending: stop every sounding note cleanly in this
        // block, so toggling off mid-note never sticks. The
        // note-offs land at offset 0: nothing in this block played
        // (the transport was already stopped when the block began).
        if (midiOutLastPosition >= 0.0)
            allNotesOffMidiOut (midiMessages, 0);
        midiOutLastPosition = -1.0;
        return;
    }

    const int channel = getMidiChannel() - 1; // 0-based here

    // Melody replaced or channel changed while sending: stop every
    // sounding note on its original channel, then resync below.
    if (generation != midiOutLatchedGeneration || channel != midiOutLastChannel)
    {
        allNotesOffMidiOut (midiMessages, 0);
        midiOutLatchedGeneration = generation;
        midiOutLastChannel = channel;
        midiOutLastPosition = -1.0;
    }

    if (sequence == nullptr || sequence->empty() || hostRate <= 0.0 || numSamples <= 0)
    {
        // Both switches on and playing, but there is no melody to
        // stream: Waiting, not Streaming — nothing leaves the plugin.
        // Any note the flags say is still sounding is turned off
        // first: a melody cleared mid-stream must not stick.
        if (sendingNow)
            midiOutStatus.store (static_cast<int> (MidiOutStatus::Waiting),
                                 std::memory_order_release);
        allNotesOffMidiOut (midiMessages, 0);
        midiOutLastPosition = -1.0;
        return;
    }

    // The end of the MIDI range this block covers: the clip end when
    // the preview ended naturally inside the block, otherwise the
    // block end.
    const double effectiveEndSeconds = endedNaturally ? activeClipEndSeconds : blockEndSeconds;

    // Sample offset for an event time, clamped inside this block. An
    // event landing exactly on the block end is sent one sample early
    // (it would otherwise be missed: the next block's range starts
    // after it).
    auto sampleOffsetFor = [blockStartSeconds, hostRate, numSamples] (double seconds)
    {
        return juce::jlimit (0, numSamples - 1, static_cast<int> (
            std::floor ((seconds - blockStartSeconds) * hostRate)));
    };

    // Resync on start, seek, melody/channel change, or when sending has
    // just been enabled. A seek must first stop notes from the old position;
    // otherwise a note that is not active at the destination can stick.
    // The threshold is 0.1 ms: float jitter between blocks is ~1e-9 s,
    // but a real seek (even a sub-5 ms playhead nudge) must resync —
    // below it, a backward seek left the note-on flag set and the note
    // stuck until the next generation change.
    const bool positionDiscontinuity = midiOutLastPosition >= 0.0
        && std::abs (blockStartSeconds - midiOutLastPosition) > 1.0e-4;
    const bool resync = midiOutLastPosition < 0.0 || positionDiscontinuity;
    if (positionDiscontinuity)
        allNotesOffMidiOut (midiMessages, 0);

    // On resync, notes already active at the block start begin at offset 0.
    // We still process every start and end later in this same block below;
    // returning here would lose short notes and note-offs.
    if (resync)
    {
        for (const auto& note : sequence->notes)
        {
            if (note.startSeconds <= blockStartSeconds && note.endSeconds > blockStartSeconds)
                noteOnMidiOut (midiMessages, 0, note, channel);
        }
    }

    // Emit every start/end inside (windowStart, effectiveEnd].
    // During normal playback windowStart is the previous block end.
    // During resync it is the current block start, because active
    // notes were restored above. Notes are sorted and monophonic,
    // so same-sample note-offs are added before the following
    // note-on.
    const double windowStart = resync ? blockStartSeconds : midiOutLastPosition;
    for (const auto& note : sequence->notes)
    {
        if (note.startSeconds > windowStart && note.startSeconds <= effectiveEndSeconds)
            noteOnMidiOut (midiMessages, sampleOffsetFor (note.startSeconds), note, channel);
        if (note.endSeconds > windowStart && note.endSeconds <= effectiveEndSeconds)
            noteOffMidiOut (midiMessages, sampleOffsetFor (note.endSeconds), note);
    }

    if (endedNaturally)
    {
        // The preview ended inside this block: any note still
        // sounding is cut at the exact clip-end sample, not at
        // offset 0 — the audio in this block played up to the end.
        allNotesOffMidiOut (midiMessages, sampleOffsetFor (effectiveEndSeconds));
        midiOutLastPosition = -1.0;
        return;
    }

    midiOutLastPosition = blockEndSeconds;
}

void MelotheonAudioProcessor::startFileImport (const juce::File& file)
{
    // Exclusive input source: files only load in IMPORTED AUDIO
    // mode (the mirror of startRecording's "USE LIVE INPUT").
    if (getInputSource() != InputSource::Imported)
    {
        lastErrorMessage = "USE IMPORTED AUDIO";
        setStatus (Status::Error);
        return;
    }

    // Starting an import is a source decision (and an import in
    // flight must not be disturbed): voice auto-detection must
    // never override it later.
    userChoseSource = true;

    if (! file.existsAsFile())
    {
        lastErrorMessage = "FILE NOT FOUND";
        setStatus (Status::Error);
        return;
    }

    // Import owns the transport. Finish a valid live take, then stop
    // preview so a late file result cannot collide with Record/Play.
    if (captureEngine.isRecording() || captureEngine.limitReached())
        finalizeCapture (Status::AudioReady);
    previewEngine.stop();
    instrumentPreviewEngine.stop();
    setPlayOriginalParameter (false);
    setPlayInstrumentParameter (false);
    clearPitchAnalysis();

    setStatus (Status::Loading);

    auto alive = aliveFlag;
    fileLoader.loadFileAsync (file,
        [this, alive] (AudioFileLoader::Result result)
        {
            if (! alive->load (std::memory_order_acquire))
                return; // processor is being destroyed

            if (result.succeeded())
            {
                // The source may have changed while the file
                // decoded: a stale result must never install a clip
                // for the wrong source. The loader's generation
                // guard already drops cancelled loads; this re-check
                // covers any path that did not cancel.
                if (getInputSource() != InputSource::Imported)
                    return;
                installClip (result.clip, Status::AudioReady);
            }
            else
            {
                lastErrorMessage = result.errorMessage;
                setStatus (Status::Error);
            }
        },
        alive);
}

bool MelotheonAudioProcessor::startPitchAnalysis()
{
    // One click after recording is enough: finalize the current take
    // before launching the worker.
    if (captureEngine.isRecording() || captureEngine.limitReached())
    {
        if (! finalizeCapture (Status::AudioReady))
            return false;
    }

    if (! hasActiveClip())
    {
        setStatus (Status::EmptySource);
        return false;
    }

    // The PRE-analyze state is pushed so an accidental RE-ANALYZE
    // is undoable (the analysis overwrites manual edits).
    pushEditHistory();

    fileLoader.cancelPendingLoad();
    previewEngine.stop();
    instrumentPreviewEngine.stop();
    setPlayOriginalParameter (false);
    setPlayInstrumentParameter (false);
    clearPitchAnalysis();

    PitchAnalysisSettings settings;
    const float gate = apvts.getRawParameterValue ("noiseGate")->load();
    const float sensitivity = apvts.getRawParameterValue ("sensitivity")->load();
    settings.silenceThresholdDb = juce::jmap (
        juce::jlimit (0.0f, 1.0f, gate), -72.0f, -35.0f);
    // Higher sensitivity accepts less-periodic frames. Keep a floor so
    // background noise cannot become a confident melody by itself.
    settings.minimumConfidence = juce::jmap (
        juce::jlimit (0.0f, 1.0f, sensitivity), 0.72f, 0.42f);

    setStatus (Status::Analyzing);
    auto alive = aliveFlag;
    pitchAnalysisEngine.start (
        activeClip, settings,
        [this, alive] (std::shared_ptr<const PitchAnalysisResult> result)
        {
            if (! alive->load (std::memory_order_acquire) || result == nullptr)
                return;

            pitchAnalysis = std::move (result);
            pitchAnalysisVersion.fetch_add (1, std::memory_order_acq_rel);

            if (pitchAnalysis->hasPitch())
            {
                lastNoteSettings = makeNoteSegmentationSettings();
                publishMelodyNotes (NoteSegmenter::segment (*pitchAnalysis, lastNoteSettings));

                if (hasMelodyNotes())
                {
                    setStatus (Status::AnalysisReady);
                    if (hasInstrumentSample())
                        startInstrumentRender();
                }
                else
                {
                    setStatus (Status::NoPitch);
                }
            }
            else
            {
                publishMelodyNotes (std::make_shared<MelodyNoteSequence>());
                setStatus (Status::NoPitch);
            }
        },
        alive);
    return true;
}

void MelotheonAudioProcessor::cancelPitchAnalysis()
{
    pitchAnalysisEngine.cancel();
    if (getStatus() == Status::Analyzing)
        setStatus (hasActiveClip() ? Status::AudioReady : Status::Ready);
}

NoteSegmentationSettings MelotheonAudioProcessor::makeNoteSegmentationSettings() const
{
    NoteSegmentationSettings noteSettings;
    const float transition = apvts.getRawParameterValue ("transition")->load();
    const float variation = apvts.getRawParameterValue ("variation")->load();
    noteSettings.minimumNoteSeconds = juce::jmap (
        static_cast<double> (juce::jlimit (0.0f, 1.0f, transition)),
        0.035, 0.120);
    noteSettings.pitchHysteresisSemitones = juce::jmap (
        juce::jlimit (0.0f, 1.0f, variation), 0.58f, 0.92f);
    return noteSettings;
}

// ---- Edit history: undo / redo of melody + curve edits (P1-7) ----

MelotheonAudioProcessor::MelodyEditSnapshot
MelotheonAudioProcessor::currentEditSnapshot() const
{
    return { pitchAnalysis, melodyNotes, pitchAnalysisEdited, lastNoteSettings };
}

void MelotheonAudioProcessor::pushUndoSnapshot (
    std::shared_ptr<const MelodyEditSnapshot> snapshot)
{
    if (static_cast<int> (undoStack.size()) >= maxEditHistorySteps)
        undoStack.erase (undoStack.begin()); // drop the oldest step
    undoStack.push_back (std::move (snapshot));
}

void MelotheonAudioProcessor::pushEditHistory()
{
    // Nothing to restore later (no analysis, no melody yet): do not
    // pollute the stack with an empty step.
    if (pitchAnalysis == nullptr && melodyNotes == nullptr)
        return;
    pushUndoSnapshot (
        std::make_shared<const MelodyEditSnapshot> (currentEditSnapshot()));
    redoStack.clear();
    editHistoryVersion.fetch_add (1, std::memory_order_acq_rel);
}

void MelotheonAudioProcessor::clearEditHistory()
{
    if (undoStack.empty() && redoStack.empty())
        return;
    undoStack.clear();
    redoStack.clear();
    editHistoryVersion.fetch_add (1, std::memory_order_acq_rel);
}

void MelotheonAudioProcessor::restoreEditSnapshot (
    const MelodyEditSnapshot& snapshot)
{
    pitchAnalysis = snapshot.analysis;
    pitchAnalysisVersion.fetch_add (1, std::memory_order_acq_rel);
    pitchAnalysisEdited = snapshot.edited;
    lastNoteSettings = snapshot.noteSettings;
    publishMelodyNotes (snapshot.melody);

    if (snapshot.melody != nullptr && ! snapshot.melody->empty())
    {
        setStatus (Status::AnalysisReady);
        if (hasInstrumentSample())
            startInstrumentRender();
    }
    else if (snapshot.analysis != nullptr && snapshot.analysis->hasPitch())
    {
        setStatus (Status::AnalysisReady);
    }
    else if (hasActiveClip())
    {
        setStatus (Status::AudioReady);
    }
    else
    {
        setStatus (Status::Ready);
    }
}

bool MelotheonAudioProcessor::undoEdit()
{
    if (undoStack.empty())
        return false;
    // A stale analysis completion must not overwrite the restored
    // state: cancel() bumps the engine generation, and the worker
    // checks it before invoking the callback.
    pitchAnalysisEngine.cancel();
    redoStack.push_back (
        std::make_shared<const MelodyEditSnapshot> (currentEditSnapshot()));
    auto snapshot = undoStack.back();
    undoStack.pop_back();
    restoreEditSnapshot (*snapshot);
    editHistoryVersion.fetch_add (1, std::memory_order_acq_rel);
    setStatusHint ("UNDO");
    return true;
}

bool MelotheonAudioProcessor::redoEdit()
{
    if (redoStack.empty())
        return false;
    pitchAnalysisEngine.cancel();
    pushUndoSnapshot (
        std::make_shared<const MelodyEditSnapshot> (currentEditSnapshot()));
    auto snapshot = redoStack.back();
    redoStack.pop_back();
    restoreEditSnapshot (*snapshot);
    editHistoryVersion.fetch_add (1, std::memory_order_acq_rel);
    setStatusHint ("REDO");
    return true;
}

bool MelotheonAudioProcessor::commitPitchCurveEdit (
    const PitchAnalysisResult& editedAnalysis)
{
    if (! editedAnalysis.hasPitch())
        return false;

    pushEditHistory(); // the PRE-edit state, so this edit is undoable

    pitchAnalysis = std::make_shared<PitchAnalysisResult> (editedAnalysis);
    pitchAnalysisVersion.fetch_add (1, std::memory_order_acq_rel);

    publishMelodyNotes (NoteSegmenter::segment (*pitchAnalysis, lastNoteSettings));

    pitchAnalysisEdited = true;
    setStatusHint ("CURVE EDITED - RE-ANALYZE TO RESET");

    if (hasMelodyNotes())
    {
        setStatus (Status::AnalysisReady);
        if (hasInstrumentSample())
            startInstrumentRender();
    }
    else
    {
        setStatus (Status::NoPitch);
    }
    return true;
}

bool MelotheonAudioProcessor::commitMelodyEdit (
    const MelodyNoteSequence& editedMelody,
    const MelodyEditor::PitchShift& pitchShift)
{
    if (editedMelody.empty())
        return false;

    pushEditHistory(); // the PRE-edit state, so this edit is undoable

    // A pitch move also shifts the underlying pitch frames in that
    // note's time range by the same delta, so the orange curve and
    // the note blocks stay in sync. Time-only moves leave the frames
    // (and the recorded audio) alone.
    if (pitchShift.active && pitchAnalysis != nullptr && pitchAnalysis->hasPitch())
    {
        auto editedAnalysis = std::make_shared<PitchAnalysisResult> (*pitchAnalysis);
        NoteSegmenter::applyCurveEdit (*editedAnalysis, pitchShift.centreSeconds,
                                       pitchShift.targetMidi, pitchShift.windowSeconds);
        pitchAnalysis = std::move (editedAnalysis);
        pitchAnalysisVersion.fetch_add (1, std::memory_order_acq_rel);
    }

    publishMelodyNotes (std::make_shared<MelodyNoteSequence> (editedMelody));

    pitchAnalysisEdited = true;
    setStatusHint ("MELODY EDITED - RE-ANALYZE TO RESET");

    if (hasInstrumentSample())
        startInstrumentRender();

    return true;
}

void MelotheonAudioProcessor::setPlayOriginalParameter (bool shouldPlay)
{
    if (auto* param = apvts.getParameter ("playOriginal"); param != nullptr)
        param->setValueNotifyingHost (shouldPlay ? 1.0f : 0.0f);
}

void MelotheonAudioProcessor::setPlayInstrumentParameter (bool shouldPlay)
{
    if (auto* param = apvts.getParameter ("playInstrument"); param != nullptr)
        param->setValueNotifyingHost (shouldPlay ? 1.0f : 0.0f);
}

void MelotheonAudioProcessor::setMidiOutParameter (bool shouldEnable)
{
    if (auto* param = apvts.getParameter ("midiOut"); param != nullptr)
        param->setValueNotifyingHost (shouldEnable ? 1.0f : 0.0f);
}

void MelotheonAudioProcessor::parameterChanged (const juce::String& parameterID,
                                              float newValue)
{
    // Exclusive input source: changing the source clears the old
    // clip so the two sources can never mix. Same-source selection
    // is a no-op (no spurious clear). Host-driven changes can arrive
    // on the audio thread, where clearing the clip is unsafe: act
    // synchronously on the message thread, otherwise defer with
    // callAsync (guarded by aliveFlag, the same pattern as the
    // file-load callbacks and the SEND TO CHANNEL auto-enable below).
    if (parameterID == "inputSource")
    {
        const auto newSource = newValue > 0.5f ? InputSource::Imported
                                               : InputSource::Live;
        if (auto* mm = juce::MessageManager::getInstanceWithoutCreating();
            mm != nullptr && mm->isThisTheMessageThread())
        {
            handleInputSourceChangeOnMessageThread (newSource);
        }
        else if (mm != nullptr)
        {
            auto alive = aliveFlag;
            mm->callAsync ([this, alive, newSource]
            {
                // Re-check at run time: the source may have changed
                // again (or back) after this deferral was queued, and
                // the stale switch must not clear the current clip.
                // The lock check inside the handler covers a lock
                // that engaged meanwhile (it reverts to LIVE INPUT).
                if (alive->load (std::memory_order_acquire)
                    && getInputSource() == newSource)
                    handleInputSourceChangeOnMessageThread (newSource);
            });
        }
        else
        {
            // No MessageManager (a console test without a GUI):
            // apply inline. The switch only touches message-thread
            // state and the calling thread is the only thread
            // running, the same as the direct processor calls in
            // the tests.
            handleInputSourceChangeOnMessageThread (newSource);
        }
        return;
    }

    // P2-8: SEND TO CHANNEL on auto-enables the MIDI OUT master.
    // Only the sendToChannel change is observed, so setting midiOut
    // here cannot loop back. Turning SEND off leaves MIDI OUT
    // unchanged, and turning the master off still stops everything.
    // Host-driven changes can arrive on the audio thread, where a
    // nested parameter change is unsafe: act synchronously on the
    // message thread, otherwise defer with callAsync (guarded by
    // aliveFlag, the same pattern as the file-load callbacks).
    if (parameterID != "sendToChannel" || newValue <= 0.5f)
        return;
    if (midiOutEnabled())
        return;
    if (auto* mm = juce::MessageManager::getInstanceWithoutCreating();
        mm != nullptr && mm->isThisTheMessageThread())
    {
        setMidiOutParameter (true);
    }
    else if (mm != nullptr)
    {
        auto alive = aliveFlag;
        mm->callAsync ([this, alive]
        {
            // Re-check at run time: the user may have turned MIDI OUT
            // off (or SEND off) after this deferral was queued, and
            // the stale lambda must not force MIDI OUT back on.
            if (alive->load (std::memory_order_acquire)
                && sendToChannelEnabled() && ! midiOutEnabled())
                setMidiOutParameter (true);
        });
    }
    // No MessageManager (a console test without a GUI): nothing to
    // defer to. The P2-8 test creates one explicitly.
}

void MelotheonAudioProcessor::handleInputSourceChangeOnMessageThread (
    InputSource newSource)
{
    // A change the auto-detector itself forced: record it only.
    // Never clear anything and never mark a user choice — the
    // detector only fires while the plugin is undecided (no clip),
    // so there is nothing to clear.
    if (applyingAutoDetectSource)
    {
        lastAppliedInputSource = newSource;
        return;
    }

    // Locked by voice detection: the user cannot leave LIVE INPUT.
    // Revert the parameter to Live (setValueNotifyingHost keeps the
    // ValueTree + host in sync). The nested notification terminates:
    // the inner call is for Live, which equals
    // lastAppliedInputSource, so it returns immediately below.
    if (voiceDetectionLocked && newSource != InputSource::Live)
    {
        if (auto* param = apvts.getParameter ("inputSource"); param != nullptr)
            param->setValueNotifyingHost (0.0f);
        return;
    }

    if (newSource == lastAppliedInputSource)
        return; // same source: nothing to clear

    // A manual source change is a decision: voice auto-detection
    // must never override it later.
    userChoseSource = true;
    applyInputSourceChange (newSource);
}

void MelotheonAudioProcessor::pollVoiceDetection()
{
    // Message thread only (called by the editor's timer, ~30 Hz).
    // The audio thread only sets the flag; this applies it. Cheap
    // early return while nothing was detected or the lock is set.
    if (! voiceDetectedFlag.load (std::memory_order_acquire)
        || voiceDetectionLocked)
        return;

    applyVoiceDetection();
}

void MelotheonAudioProcessor::applyVoiceDetection()
{
    // Message thread only. Forces LIVE INPUT only while the plugin
    // is still undecided: no clip loaded AND the user has not
    // chosen a source (no manual change, no import, no recording).
    // Once the user has a clip or has chosen a source, detection
    // never switches and never clears anything — a user who
    // imported a file must not lose it because the track's voice
    // is detected later when they press PLAY.
    if (voiceDetectionLocked || userChoseSource || hasActiveClip())
        return;

    // Force the parameter through the APVTS so the ValueTree, the
    // host and the saved state all see LIVE INPUT. The listener
    // records the change only (applyingAutoDetectSource): no clear,
    // no user-choice mark. The source is always IMPORTED here (the
    // undecided default), so the change always notifies; the direct
    // lastAppliedInputSource update below covers the no-notify
    // case anyway.
    applyingAutoDetectSource = true;
    if (auto* param = apvts.getParameter ("inputSource"); param != nullptr)
        param->setValueNotifyingHost (0.0f);
    applyingAutoDetectSource = false;
    lastAppliedInputSource = InputSource::Live;

    // Lock for the session (until the plugin is closed/reloaded):
    // the dropdown greys out and later changes are reverted (see
    // handleInputSourceChangeOnMessageThread).
    voiceDetectionLocked = true;
    setStatusHint ("VOICE DETECTED - LIVE INPUT");
}

void MelotheonAudioProcessor::clearActiveClip()
{
    // The single clip-clearing path (see the header): cancel any
    // in-flight analysis/render, drop the analysis + melody + undo
    // history, detach both preview engines from the clip (they defer
    // the ownership change until any in-flight render finishes, so
    // this is safe with respect to the audio thread) and release the
    // clip. The clip version bumps only when a clip was actually
    // removed: a switch with nothing loaded must not look like a
    // new clip to the editor (or to the project-state audio cache).
    const bool hadClip = activeClip != nullptr;
    clearPitchAnalysis();
    clearEditHistory();
    previewEngine.setClip (nullptr);
    instrumentPreviewEngine.setClip (nullptr);
    activeClip.reset();
    if (hadClip)
        clipVersion.fetch_add (1, std::memory_order_acq_rel);
}

void MelotheonAudioProcessor::applyInputSourceChange (InputSource newSource)
{
    // Message thread only. The listener guards the same-source case
    // too, but the deferred path re-checks here so a queued switch
    // for an outdated source cannot clear a clip the user has since
    // re-selected.
    if (newSource == lastAppliedInputSource)
        return;
    lastAppliedInputSource = newSource;

    // A take in progress belongs to the old source: discard it.
    // stopRecording() waits for the audio writer and drops the
    // unfinalized samples (no clip is installed); the limit flag
    // goes too, so a limit take cannot finalize later either.
    const bool hadTake = captureEngine.isRecording() || captureEngine.limitReached();
    captureEngine.stopRecording();
    captureEngine.clearLimitFlag();

    // A file load in flight belongs to the old source: cancel it.
    // The loader's generation guard drops its message-thread
    // callback, and startFileImport's completion re-checks the
    // source before installing anything.
    fileLoader.cancelPendingLoad();

    // Stop playback: the old clip is going away.
    previewEngine.stop();
    instrumentPreviewEngine.stop();
    setPlayOriginalParameter (false);
    setPlayInstrumentParameter (false);

    // The message names what was lost; a switch with nothing loaded
    // needs no explanation beyond the empty state.
    const bool hadSomething = hadTake || hasActiveClip()
        || pitchAnalysis != nullptr || melodyNotes != nullptr;
    clearActiveClip();

    lastErrorMessage.clear();
    setStatus (Status::EmptySource);
    if (hadSomething)
        setStatusHint ("SOURCE CHANGED - CLIP CLEARED");
}

juce::String MelotheonAudioProcessor::getMidiRoutingHint() const
{
    // P2-9: the routing instruction for the status LCD. Shown only
    // when both MIDI switches are on, nothing is playing, and the
    // plugin is in an idle state — it must never hide Recording,
    // Analyzing, Loading, Error or playback progress.
    if (! midiOutEnabled() || ! sendToChannelEnabled())
        return {};
    if (isPreviewPlaying() || isInstrumentPreviewPlaying())
        return {};
    // Idle states only: the hint must never hide Recording,
    // Analyzing, Loading, Error or playback progress.
    const auto currentStatus = getStatus();
    if (currentStatus == Status::Ready || currentStatus == Status::AudioReady
        || currentStatus == Status::AnalysisReady || currentStatus == Status::NoPitch
        || currentStatus == Status::SoundReady || currentStatus == Status::InstrumentReady
        || currentStatus == Status::Stopped || currentStatus == Status::EmptySource)
        return "ROUTE MIDI PORT, THEN PRESS PLAY";
    return {};
}

void MelotheonAudioProcessor::startInstrumentFileImport (const juce::File& file)
{
    if (! file.existsAsFile())
    {
        lastErrorMessage = "SOUND FILE NOT FOUND";
        setStatus (Status::Error);
        return;
    }

    instrumentPreviewEngine.stop();
    setPlayInstrumentParameter (false);
    sampleRenderEngine.cancel();
    clearRenderedInstrument();
    instrumentSample.reset();
    instrumentBaseName.clear();
    instrumentDisplayText = "LOADING SOUND";
    instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
    setStatus (Status::LoadingSound);
    // The old sound is gone: stop any held audition key and drop the
    // audition source (the new one installs when the load finishes).
    installAuditionSample (nullptr, 60);

    auto alive = aliveFlag;
    instrumentFileLoader.loadFileAsync (
        file,
        [this, alive, selectedName = file.getFileNameWithoutExtension()]
        (AudioFileLoader::Result loadResult)
        {
            if (! alive->load (std::memory_order_acquire))
                return;

            if (! loadResult.succeeded())
            {
                lastErrorMessage = loadResult.errorMessage;
                instrumentDisplayText = "NO SOUND LOADED";
                instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
                setStatus (Status::Error);
                return;
            }
            if (loadResult.clip->durationSeconds > 15.0)
            {
                lastErrorMessage = "USE A SHORT ONE-SHOT";
                instrumentDisplayText = "NO SOUND LOADED";
                instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
                setStatus (Status::Error);
                return;
            }

            instrumentSample = std::move (loadResult.clip);
            // Detect the root once, on load (synchronous, message
            // thread; the one-shot is at most 15 s), so piano-key
            // audition can pitch it immediately, even before ANALYZE.
            installAuditionSample (
                instrumentSample,
                SampleInstrumentRenderer::detectRootMidi (*instrumentSample));
            instrumentBaseName = selectedName.substring (0, 22).toUpperCase();
            instrumentDisplayText = instrumentBaseName
                                  + (hasMelodyNotes() ? " / RENDERING"
                                                     : " / ANALYZE FIRST");
            instrumentVersion.fetch_add (1, std::memory_order_acq_rel);

            if (hasMelodyNotes())
                startInstrumentRender();
            else
                setStatus (Status::SoundReady);
        },
        alive);
}

void MelotheonAudioProcessor::startInstrumentRender()
{
    if (! hasInstrumentSample())
    {
        lastErrorMessage = "LOAD SOUND FIRST";
        setStatus (Status::Error);
        return;
    }
    if (! hasMelodyNotes())
    {
        lastErrorMessage = "ANALYZE FIRST";
        setStatus (Status::Error);
        return;
    }

    instrumentPreviewEngine.stop();
    setPlayInstrumentParameter (false);
    clearRenderedInstrument();
    instrumentDisplayText = instrumentBaseName + " / RENDERING";
    instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
    setStatus (Status::RenderingInstrument);

    auto alive = aliveFlag;
    sampleRenderEngine.start (
        instrumentSample, melodyNotes,
        hostSampleRate.load (std::memory_order_acquire),
        [this, alive] (SampleRenderResult result)
        {
            if (! alive->load (std::memory_order_acquire))
                return;

            if (! result.succeeded())
            {
                lastErrorMessage = result.errorMessage.isNotEmpty()
                    ? result.errorMessage : "INSTRUMENT RENDER ERROR";
                instrumentDisplayText = instrumentBaseName + " / ERROR";
                instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
                setStatus (Status::Error);
                return;
            }

            renderedInstrumentClip = std::move (result.renderedClip);
            instrumentPreviewEngine.setClip (renderedInstrumentClip);
            instrumentDisplayText = instrumentBaseName + "  ROOT "
                                  + midiNoteName (result.rootMidi);
            instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
            setStatus (Status::InstrumentReady);
        },
        alive);
}

void MelotheonAudioProcessor::clearRenderedInstrument()
{
    instrumentPreviewEngine.stop();
    instrumentPreviewEngine.setClip (nullptr);
    setPlayInstrumentParameter (false);
    if (renderedInstrumentClip != nullptr)
    {
        renderedInstrumentClip.reset();
        instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
    }
}

bool MelotheonAudioProcessor::startInstrumentPreview()
{
    if (sampleRenderEngine.isRunning())
        return false;
    if (! hasInstrumentSample())
    {
        lastErrorMessage = "LOAD SOUND FIRST";
        setStatus (Status::Error);
        return false;
    }
    if (! hasMelodyNotes())
    {
        lastErrorMessage = "ANALYZE FIRST";
        setStatus (Status::Error);
        return false;
    }
    if (! hasRenderedInstrument())
    {
        startInstrumentRender();
        return false;
    }

    previewEngine.stop();
    setPlayOriginalParameter (false);
    if (! instrumentPreviewEngine.start())
        return false;

    setStatus (Status::PlayingInstrument);
    return true;
}

void MelotheonAudioProcessor::stopInstrumentPreview()
{
    instrumentPreviewEngine.stop();
    setPlayInstrumentParameter (false);
    setStatus (Status::Stopped);
}

void MelotheonAudioProcessor::pauseInstrumentPreview()
{
    instrumentPreviewEngine.pause();
    setPlayInstrumentParameter (false);
    setStatus (Status::Stopped);
}

TempoGrid MelotheonAudioProcessor::getTempoGrid() const
{
    TempoGrid grid;
    grid.bpm = cachedHostBpm.load (std::memory_order_acquire);
    grid.timeSignatureNumerator = cachedTimeSigNumerator.load (std::memory_order_acquire);
    grid.timeSignatureDenominator = cachedTimeSigDenominator.load (std::memory_order_acquire);
    grid.ppqPosition = cachedHostPpq.load (std::memory_order_acquire);
    return grid;
}

void MelotheonAudioProcessor::refreshHostTempo()
{
    auto* hostPlayHead = getPlayHead();
    if (hostPlayHead == nullptr)
        return;
    const auto position = hostPlayHead->getPosition();
    if (! position.hasValue())
        return;

    bool changed = false;

    // BPM: only plausible host values update the cache; a missing or
    // absurd value keeps the previous tempo (120 BPM fallback when
    // the host never provided one).
    if (const auto bpm = position->getBpm(); bpm.hasValue())
    {
        const double value = *bpm;
        if (std::isfinite (value) && value >= TempoGrid::kMinimumBpm
            && value <= TempoGrid::kMaximumBpm
            && std::abs (value - cachedHostBpm.load (std::memory_order_acquire)) > 0.01)
        {
            cachedHostBpm.store (value, std::memory_order_release);
            changed = true;
        }
    }

    if (const auto timeSig = position->getTimeSignature(); timeSig.hasValue())
    {
        const int numerator = juce::jlimit (1, 16, timeSig->numerator);
        const int denominator = juce::jlimit (1, 16, timeSig->denominator);
        if (numerator != cachedTimeSigNumerator.load (std::memory_order_acquire)
            || denominator != cachedTimeSigDenominator.load (std::memory_order_acquire))
        {
            cachedTimeSigNumerator.store (numerator, std::memory_order_release);
            cachedTimeSigDenominator.store (denominator, std::memory_order_release);
            changed = true;
        }
    }

    // PPQ is cached for diagnostics; it never phases the grid (see
    // TempoGrid) and never bumps the version — it moves constantly
    // during playback and the editor does not need to track it.
    if (const auto ppq = position->getPpqPosition(); ppq.hasValue())
    {
        const double value = *ppq;
        if (std::isfinite (value) && value >= 0.0)
            cachedHostPpq.store (value, std::memory_order_release);
    }

    if (changed)
        tempoVersion.fetch_add (1, std::memory_order_acq_rel);
}

bool MelotheonAudioProcessor::exportMidiToFile (const juce::File& target)
{
    if (! hasMelodyNotes())
    {
        lastErrorMessage = "ANALYZE FIRST";
        setStatus (Status::Error);
        return false;
    }

    // Single source of truth for tempo: the cached host grid
    // (refreshed from the playhead first, 120 BPM fallback).
    refreshHostTempo();
    MidiExportSettings settings;
    settings.channel = getMidiChannel();
    settings.bpm = getTempoGrid().clampedBpm();

    juce::String error;
    if (! MidiExporter::writeToFile (*melodyNotes, target, settings, error))
    {
        lastErrorMessage = error.isNotEmpty() ? error : "MIDI WRITE ERROR";
        setStatus (Status::Error);
        return false;
    }

    setStatus (Status::MidiExported);
    return true;
}

void MelotheonAudioProcessor::installClip (std::shared_ptr<const AudioClip> clip,
                                           Status statusAfter)
{
    if (clip == nullptr || clip->isEmpty())
    {
        setStatus (Status::EmptySource);
        return;
    }

    clearPitchAnalysis();
    clearEditHistory(); // a new clip: old snapshots are incoherent
    previewEngine.setClip (clip);
    activeClip = std::move (clip);
    clipVersion.fetch_add (1, std::memory_order_acq_rel);
    setStatus (statusAfter);
}

void MelotheonAudioProcessor::clearPitchAnalysis()
{
    pitchAnalysisEngine.cancel();
    sampleRenderEngine.cancel();
    clearRenderedInstrument();
    pitchAnalysisEdited = false;
    if (pitchAnalysis != nullptr)
    {
        pitchAnalysis.reset();
        pitchAnalysisVersion.fetch_add (1, std::memory_order_acq_rel);
    }
    if (melodyNotes != nullptr)
        publishMelodyNotes (nullptr);
    if (hasInstrumentSample())
    {
        instrumentDisplayText = instrumentBaseName + " / ANALYZE FIRST";
        instrumentVersion.fetch_add (1, std::memory_order_acq_rel);
    }
}

bool MelotheonAudioProcessor::finalizeCapture (Status statusAfter)
{
    const int recorded = captureEngine.stopRecording();
    captureEngine.clearLimitFlag();

    if (recorded < 0)
        return false;

    if (recorded == 0)
    {
        setStatus (Status::NoInput);
        return false;
    }

    auto clip = captureEngine.createClip ("LIVE RECORDING");
    if (clip == nullptr || clip->isEmpty())
    {
        setStatus (Status::NoInput);
        return false;
    }

    installClip (std::move (clip), statusAfter);
    return true;
}

void MelotheonAudioProcessor::updateInputMeter (const juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    if (numSamples <= 0)
        return;

    const double rate = hostSampleRate.load (std::memory_order_acquire);
    const double dt = rate > 0.0 ? static_cast<double> (numSamples) / rate : 0.01;

    float blockPeak = 0.0f;
    double sumSquares = 0.0;
    const int numChannels = buffer.getNumChannels();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float* data = buffer.getReadPointer (ch);
        for (int i = 0; i < numSamples; ++i)
        {
            const float sample = data[i];
            const float magnitude = std::abs (sample);
            if (magnitude > blockPeak)
                blockPeak = magnitude;
            sumSquares += static_cast<double> (sample) * static_cast<double> (sample);
        }
    }

    const float blockRms = static_cast<float> (
        std::sqrt (sumSquares / static_cast<double> (numSamples * numChannels)));

    // Peak: instant attack, exponential release (tau ~0.25 s) so the
    // meter falls smoothly instead of flickering.
    const float release = static_cast<float> (std::exp (-dt / 0.25));
    meterPeakLinear = juce::jmax (blockPeak, meterPeakLinear * release);

    // RMS: one-pole smoothing (tau ~0.1 s).
    const float alpha = static_cast<float> (1.0 - std::exp (-dt / 0.1));
    meterRmsLinear += alpha * (blockRms - meterRmsLinear);

    const float peakDb = juce::jlimit (-60.0f, 0.0f,
                                       juce::Decibels::gainToDecibels (meterPeakLinear));
    const float rmsDb = juce::jlimit (-60.0f, 0.0f,
                                      juce::Decibels::gainToDecibels (meterRmsLinear));
    inputPeakDb.store (peakDb, std::memory_order_release);
    inputRmsDb.store (rmsDb, std::memory_order_release);

    // Voice auto-detection reads the same block (see
    // updateVoiceDetection): the incoming host audio, before any
    // preview overwrites the buffer.
    updateVoiceDetection (blockRms, dt);
}

void MelotheonAudioProcessor::updateVoiceDetection (float blockRms, double dt)
{
    // Audio thread only. Once a voice was detected there is nothing
    // more to accumulate: the flag is sticky for the session.
    if (voiceDetectedFlag.load (std::memory_order_acquire))
        return;

    // Threshold: the same silence definition the pitch analysis
    // uses (noiseGate mapped to -72..-35 dB), floored at -50 dB so
    // background noise / hum cannot lock the source.
    const float gate = noiseGateParam != nullptr
        ? noiseGateParam->load (std::memory_order_acquire) : 0.3f;
    const float thresholdDb = juce::jmax (
        juce::jmap (juce::jlimit (0.0f, 1.0f, gate), -72.0f, -35.0f),
        voiceDetectThresholdFloorDb);
    const float thresholdLinear = juce::Decibels::decibelsToGain (thresholdDb);

    if (blockRms >= thresholdLinear)
    {
        voiceDetectVoicedSeconds += dt;
        voiceDetectSilentSeconds = 0.0;
    }
    else
    {
        voiceDetectSilentSeconds += dt;
        // Isolated transients never add up: 2 s of continuous
        // silence resets the voiced accumulator.
        if (voiceDetectSilentSeconds >= voiceDetectSilenceResetSeconds)
            voiceDetectVoicedSeconds = 0.0;
    }

    if (voiceDetectVoicedSeconds >= voiceDetectSustainSeconds)
        voiceDetectedFlag.store (true, std::memory_order_release);
}

void MelotheonAudioProcessor::updateSilenceStatus (int numSamples)
{
    const Status current = getStatus();
    if (current != Status::Ready && current != Status::NoInput
        && current != Status::EmptySource)
        return;

    if (meterPeakLinear < 0.0001f) // below ~-80 dBFS
        silentSamples += numSamples;
    else
        silentSamples = 0;

    const double rate = hostSampleRate.load (std::memory_order_acquire);
    if (silentSamples > static_cast<int> (rate)) // ~1 s of silence
    {
        if (current == Status::Ready)
            setStatus (Status::NoInput);
    }
    else if (meterPeakLinear >= 0.0001f)
    {
        if (current == Status::NoInput)
            setStatus (Status::Ready);
        else if (current == Status::EmptySource && getInputSource() == InputSource::Live)
            setStatus (Status::Ready);
    }
}

juce::AudioProcessorEditor* MelotheonAudioProcessor::createEditor()
{
    return new MelotheonAudioProcessorEditor (*this);
}

void MelotheonAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // Full project state (see ProjectState): parameters plus the
    // clip (embedded as FLAC), the pitch analysis and the (edited)
    // melody, under a versioned wrapper. Old APVTS-only states still
    // load (see setStateInformation). The audio encoding is cached
    // per clip version: hosts may snapshot state repeatedly, and
    // re-encoding a 60 s clip on every call would waste time.
    // Expected on the message thread (as hosts call it); the clip
    // and analysis owners are message-thread-owned.
    ProjectState::EncodedAudioCache cache;
    cache.flac = cachedProjectAudioFlac;
    cache.peakScale = cachedProjectAudioPeakScale;
    cache.clipVersion = cachedProjectAudioClipVersion;
    ProjectState::Data data;
    data.clip = activeClip;
    data.analysis = pitchAnalysis;
    data.melody = melodyNotes;
    data.pitchAnalysisEdited = pitchAnalysisEdited;
    // copyState() (not apvts.state directly): it flushes pending
    // parameter changes into the tree first. apvts.state can lag
    // behind the live values until the APVTS 10 Hz timer runs, so
    // reading it directly would save stale parameters.
    auto wrapper = ProjectState::serializeState (apvts.copyState(), data, &cache,
                                                 getClipVersion());
    cachedProjectAudioFlac = cache.flac;
    cachedProjectAudioPeakScale = cache.peakScale;
    cachedProjectAudioClipVersion = cache.clipVersion;
    juce::MemoryOutputStream stream (destData, false);
    wrapper.writeToStream (stream);
}

void MelotheonAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (data == nullptr || sizeInBytes <= 0)
        return;
    auto parsed = ProjectState::deserializeState (data, sizeInBytes);
    if (! parsed.hasParams && ! parsed.hasProject)
        return; // unrecognizable: keep the current state

    // Restore touches message-thread-only state (clip install,
    // analysis publish, APVTS replace), so it runs on the message
    // thread; the alive flag guards the deferred call. When no
    // MessageManager exists at all (console tests), apply inline.
    auto alive = aliveFlag;
    auto apply = [this, parsed = std::move (parsed), alive]() mutable
    {
        if (alive == nullptr || ! alive->load (std::memory_order_acquire))
            return; // processor is being destroyed
        applyParsedState (std::move (parsed));
    };
    auto* messageManager = juce::MessageManager::getInstanceWithoutCreating();
    if (messageManager == nullptr || messageManager->isThisTheMessageThread())
        apply();
    else
        messageManager->callAsync (std::move (apply));
}

void MelotheonAudioProcessor::applyParsedState (ProjectState::Deserialized parsed)
{
    if (parsed.hasParams)
        apvts.replaceState (parsed.params);
    if (parsed.hasProject)
        applyRestoredProject (std::move (parsed.project));
}

void MelotheonAudioProcessor::applyRestoredProject (ProjectState::Data project)
{
    // A project load is a new state: the old edit history belongs
    // to whatever clip was open before (history is not saved).
    clearEditHistory();

    // The clip first: installClip clears any previous analysis and
    // melody, then the stored ones are restored below. A state with
    // no clip clears the current one, so reloading an empty project
    // does not leave stale audio behind.
    if (project.clip != nullptr && ! project.clip->isEmpty())
    {
        installClip (project.clip, Status::AudioReady);
    }
    else
    {
        clearActiveClip();
        setStatus (Status::EmptySource);
        if (project.audioSkippedTooLarge)
            setStatusHint ("AUDIO TOO LARGE - NOT SAVED");
    }

    if (project.analysis != nullptr && project.analysis->hasPitch())
    {
        pitchAnalysis = project.analysis;
        pitchAnalysisVersion.fetch_add (1, std::memory_order_acq_rel);
        pitchAnalysisEdited = project.pitchAnalysisEdited;
    }

    if (project.melody != nullptr && ! project.melody->empty())
    {
        publishMelodyNotes (project.melody);
        setStatus (Status::AnalysisReady);
        if (hasInstrumentSample())
            startInstrumentRender();
    }
    else if (project.analysis != nullptr && project.analysis->hasPitch())
    {
        setStatus (Status::AnalysisReady);
    }
}

// Required by JUCE: factory function that creates an instance of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MelotheonAudioProcessor();
}
