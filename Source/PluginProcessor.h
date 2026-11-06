#pragma once

// MelotheonAudioProcessor - the VST3 audio processor (Phase 1).
//
// Phase 1 contract: the processor passes incoming audio through
// UNCHANGED. There is no pitch detection, no audio analysis, no MIDI
// generation, no instrument hosting and no DSP of any kind here.
// processBlock does no allocation, no locking, no file I/O and no
// networking.
//
// The AudioProcessorValueTreeState holds the interface parameters so
// the editor state persists in the host project and becomes
// automatable later. The parameters are NOT used for DSP in this phase.

#include <JuceHeader.h>

class MelotheonAudioProcessor : public juce::AudioProcessor
{
public:
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

    // Interface parameters (editor state only in Phase 1).
    juce::AudioProcessorValueTreeState apvts;

private:
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MelotheonAudioProcessor)
};
