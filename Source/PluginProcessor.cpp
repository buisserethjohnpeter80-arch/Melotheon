// MelotheonAudioProcessor implementation (Phase 1).
// See PluginProcessor.h for the contract: audio passes through
// unchanged; parameters exist for editor state only.

#include "PluginProcessor.h"
#include "PluginEditor.h"

MelotheonAudioProcessor::MelotheonAudioProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "MelotheonState", createParameterLayout())
{
}

MelotheonAudioProcessor::~MelotheonAudioProcessor() = default;

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

    // Header input source, right-panel instrument and MIDI channel.
    layout.add (std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { "inputSource", 1 }, "Input Source",
        juce::StringArray { "INPUT: VOICE", "INPUT: GUITAR",
                            "INPUT: PIANO", "INPUT: LINE" },
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

void MelotheonAudioProcessor::prepareToPlay (double /*sampleRate*/, int /*samplesPerBlock*/)
{
    // Phase 1: nothing to prepare - no DSP state exists.
}

void MelotheonAudioProcessor::releaseResources()
{
    // Phase 1: nothing to release.
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
    juce::ignoreUnused (buffer);
    midiMessages.clear();

    // Pass-through: the buffer is processed in place by the host, so
    // untouched audio flows through the Mixer Track unchanged. Phase 1
    // has no DSP: nothing here reads or writes audio samples.
    // TODO: PHASE_2_AUDIO_LOGIC - pitch detection, analysis and MIDI
    // generation will read the buffer here in a later phase.
}

juce::AudioProcessorEditor* MelotheonAudioProcessor::createEditor()
{
    return new MelotheonAudioProcessorEditor (*this);
}

void MelotheonAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // Persist the editor parameters (APVTS state) into the host project.
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
