// MelotheonAudioProcessor implementation (Phase 2).
// See PluginProcessor.h for the contract: audio foundation only -
// capture, import, preview, meter, status. No pitch detection, no
// MIDI generation, no instrument playback.

#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <cmath>

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

    // Left capture panel: noise gate knob + mono toggle.
    // The noise gate is NOT applied to audio in Phase 2; the
    // parameter is kept for future pitch-analysis preprocessing.
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
        juce::ParameterID { "instrumentTarget", 1 }, "Instrument",
        juce::StringArray { "FLEX GRAND PIANO", "GRAND PIANO",
                            "ELECTRIC PIANO", "STRINGS", "CHOIR PAD" },
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

    // 3. Original-audio preview replaces the pass-through while it
    // plays; otherwise the Mixer Track input flows through unchanged.
    if (previewEngine.isPlaying())
        previewEngine.render (buffer, hostSampleRate.load (std::memory_order_acquire));

    // 4. READY <-> NO INPUT transitions (only from those states).
    updateSilenceStatus (buffer.getNumSamples());
}

// ---- Phase 2 public API ----

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
    setPlayOriginalParameter (false);

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
    setPlayOriginalParameter (false);

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

void MelotheonAudioProcessor::setPlayOriginalParameter (bool shouldPlay)
{
    if (auto* param = apvts.getParameter ("playOriginal"); param != nullptr)
        param->setValueNotifyingHost (shouldPlay ? 1.0f : 0.0f);
}

void MelotheonAudioProcessor::installClip (std::shared_ptr<const AudioClip> clip,
                                           Status statusAfter)
{
    if (clip == nullptr || clip->isEmpty())
    {
        setStatus (Status::EmptySource);
        return;
    }

    previewEngine.setClip (clip);
    activeClip = std::move (clip);
    clipVersion.fetch_add (1, std::memory_order_acq_rel);
    setStatus (statusAfter);
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
