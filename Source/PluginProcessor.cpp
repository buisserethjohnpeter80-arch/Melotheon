// MelotheonAudioProcessor implementation (Phase 3B/3C).
// The Phase 2 audio foundation remains real-time safe. Pitch analysis
// and sample-instrument rendering run on workers; MIDI export runs only
// from an explicit message-thread button action.

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "Analysis/NoteSegmenter.h"
#include "Midi/MidiExporter.h"

#include <chrono>
#include <cmath>
#include <thread>

namespace
{
// In-flight announcement for the audition voice and the live MIDI-out
// voice: the message thread waits for the counter to drain before
// replacing the sample buffer / melody sequence. Same pattern as
// AudioPreviewEngine's RenderActivityGuard.
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
}

MelotheonAudioProcessor::~MelotheonAudioProcessor()
{
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
    layout.add (std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { "inputSource", 1 }, "Input Source",
        juce::StringArray { "LIVE INPUT", "IMPORTED AUDIO" },
        0));
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

    // The preview position this block starts at, captured before any
    // render advances it: the live MIDI-out scheduler below needs the
    // block's exact time range.
    const double midiBlockStartSeconds = getDisplayPreviewPositionSeconds();

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
    // sounding note when sending stops).
    renderMidiOut (midiMessages, midiBlockStartSeconds, buffer.getNumSamples(),
                   hostSampleRate.load (std::memory_order_acquire));

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
    melodyNotesVersion.fetch_add (1, std::memory_order_acq_rel);
    publishMidiOutSequence();
}

void MelotheonAudioProcessor::waitForMidiOutToFinish()
{
    // Bounded: the audio thread may be stopped (host idle), in which
    // case nothing can read the sequence and waiting is pointless.
    for (int i = 0; i < 2000; ++i)
    {
        if (midiOutRenderersInFlight.load (std::memory_order_seq_cst) == 0)
            return;
        std::this_thread::sleep_for (std::chrono::microseconds (50));
    }
}

void MelotheonAudioProcessor::publishMidiOutSequence()
{
    waitForMidiOutToFinish();
    midiOutSequenceForAudio.store (melodyNotes != nullptr ? melodyNotes.get() : nullptr,
                                   std::memory_order_seq_cst);
    midiOutGeneration.fetch_add (1, std::memory_order_acq_rel);
}

void MelotheonAudioProcessor::resetMidiOutState()
{
    midiOutLatchedSequence = nullptr;
    midiOutLatchedGeneration = 0;
    midiOutLastChannel = -1;
    midiOutLastPosition = -1.0;
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
                                            int numSamples, double hostRate)
{
    // The MIDI OUT toggle is the master switch and SEND TO CHANNEL is
    // the per-melody switch: both must be on, and a preview (or
    // instrument preview) must be playing, for any MIDI to leave.
    const bool sending = midiOutEnabled() && sendToChannelEnabled()
        && (previewEngine.isPlaying() || instrumentPreviewEngine.isPlaying());
    if (! sending)
    {
        // Was sending: stop every sounding note cleanly in this
        // block, so toggling off mid-note never sticks.
        if (midiOutLastPosition >= 0.0)
            allNotesOffMidiOut (midiMessages, 0);
        midiOutLastPosition = -1.0;
        return;
    }

    ActivityGuard inFlightGuard (midiOutRenderersInFlight);

    const int channel = getMidiChannel() - 1; // 0-based here
    const auto* sequence = midiOutSequenceForAudio.load (std::memory_order_seq_cst);
    const std::uint64_t generation = midiOutGeneration.load (std::memory_order_acquire);

    // Melody replaced or channel changed while sending: stop every
    // sounding note on its original channel, then resync below.
    if (generation != midiOutLatchedGeneration || sequence != midiOutLatchedSequence
        || channel != midiOutLastChannel)
    {
        allNotesOffMidiOut (midiMessages, 0);
        midiOutLatchedSequence = sequence;
        midiOutLatchedGeneration = generation;
        midiOutLastChannel = channel;
        midiOutLastPosition = -1.0;
    }

    if (sequence == nullptr || sequence->empty() || hostRate <= 0.0 || numSamples <= 0)
    {
        midiOutLastPosition = -1.0;
        return;
    }

    const double blockEndSeconds = blockStartSeconds
        + static_cast<double> (numSamples) / hostRate;

    // Sample offset for an event time, clamped inside this block. An
    // event landing exactly on the block end is sent one sample early
    // (it would otherwise be missed: the next block's range starts
    // after it).
    auto sampleOffsetFor = [blockStartSeconds, hostRate, numSamples] (double seconds)
    {
        return juce::jlimit (0, numSamples - 1, static_cast<int> (
            std::floor ((seconds - blockStartSeconds) * hostRate)));
    };

    // Resync (start, seek, melody/channel change, or sending just
    // enabled): notes containing the block start sound from offset 0.
    if (midiOutLastPosition < 0.0
        || std::abs (blockStartSeconds - midiOutLastPosition) > 0.005)
    {
        for (const auto& note : sequence->notes)
        {
            if (note.startSeconds <= blockStartSeconds && note.endSeconds > blockStartSeconds)
                noteOnMidiOut (midiMessages, 0, note, channel);
        }
        midiOutLastPosition = blockEndSeconds;
        return;
    }

    // Continuous playback: emit the starts/ends inside
    // (lastPosition, blockEnd]. Notes are sorted by start and never
    // overlap, so iterating in order emits events in time order.
    for (const auto& note : sequence->notes)
    {
        if (note.startSeconds > midiOutLastPosition && note.startSeconds <= blockEndSeconds)
            noteOnMidiOut (midiMessages, sampleOffsetFor (note.startSeconds), note, channel);
        if (note.endSeconds > midiOutLastPosition && note.endSeconds <= blockEndSeconds)
            noteOffMidiOut (midiMessages, sampleOffsetFor (note.endSeconds), note);
    }
    midiOutLastPosition = blockEndSeconds;
}

void MelotheonAudioProcessor::startFileImport (const juce::File& file)
{
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
                installClip (result.clip, Status::AudioReady);
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

bool MelotheonAudioProcessor::commitPitchCurveEdit (
    const PitchAnalysisResult& editedAnalysis)
{
    if (! editedAnalysis.hasPitch())
        return false;

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

bool MelotheonAudioProcessor::exportMidiToFile (const juce::File& target)
{
    if (! hasMelodyNotes())
    {
        lastErrorMessage = "ANALYZE FIRST";
        setStatus (Status::Error);
        return false;
    }

    MidiExportSettings settings;
    settings.channel = getMidiChannel();
    if (auto* hostPlayHead = getPlayHead(); hostPlayHead != nullptr)
    {
        if (const auto position = hostPlayHead->getPosition(); position.hasValue())
            if (const auto bpm = position->getBpm(); bpm.hasValue())
                settings.bpm = *bpm;
    }

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
    // Persist the editor parameters (APVTS state) into the host project.
    // The recorded/imported audio clip is NOT serialized: a project
    // reload without its temporary clip returns to READY / EMPTY SOURCE.
    juce::MemoryOutputStream stream (destData, false);
    apvts.state.writeToStream (stream);
}

void MelotheonAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    if (data == nullptr || sizeInBytes <= 0)
        return;
    if (auto state = juce::ValueTree::readFromData (data, static_cast<size_t> (sizeInBytes));
        state.isValid())
    {
        apvts.replaceState (state);
    }
}

// Required by JUCE: factory function that creates an instance of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MelotheonAudioProcessor();
}
