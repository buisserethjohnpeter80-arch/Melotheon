// MelotheonAudioProcessor implementation (Phase 3B/3C).
// The Phase 2 audio foundation remains real-time safe. Pitch analysis
// and sample-instrument rendering run on workers; MIDI export runs only
// from an explicit message-thread button action.

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "Analysis/NoteSegmenter.h"
#include "Midi/MidiExporter.h"

#include <cmath>

namespace
{
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
    previewEngine.stop();
    instrumentPreviewEngine.stop();
    captureEngine.stopRecording(); // discard any unfinalized take
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

    // 4. READY <-> NO INPUT transitions (only from those states).
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
                NoteSegmentationSettings noteSettings;
                const float transition = apvts.getRawParameterValue ("transition")->load();
                const float variation = apvts.getRawParameterValue ("variation")->load();
                noteSettings.minimumNoteSeconds = juce::jmap (
                    static_cast<double> (juce::jlimit (0.0f, 1.0f, transition)),
                    0.035, 0.120);
                noteSettings.pitchHysteresisSemitones = juce::jmap (
                    juce::jlimit (0.0f, 1.0f, variation), 0.58f, 0.92f);
                melodyNotes = NoteSegmenter::segment (*pitchAnalysis, noteSettings);
                melodyNotesVersion.fetch_add (1, std::memory_order_acq_rel);

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
                melodyNotes = std::make_shared<MelodyNoteSequence>();
                melodyNotesVersion.fetch_add (1, std::memory_order_acq_rel);
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
    settings.channel = juce::jlimit (1, 16, 1 + static_cast<int> (
        apvts.getRawParameterValue ("midiChannel")->load()));
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
    if (pitchAnalysis != nullptr)
    {
        pitchAnalysis.reset();
        pitchAnalysisVersion.fetch_add (1, std::memory_order_acq_rel);
    }
    if (melodyNotes != nullptr)
    {
        melodyNotes.reset();
        melodyNotesVersion.fetch_add (1, std::memory_order_acq_rel);
    }
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
