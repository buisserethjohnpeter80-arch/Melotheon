#pragma once

// MelotheonAudioProcessorEditor - the plugin UI (Phase 1).
//
// Reference: the attached "MELODY CAPTURE" mockup (dark FL Studio-style
// panels, orange active states, green status displays, central piano
// roll). The FL Studio toolbar, Browser, Channel Rack, Playlist and
// outer plugin-wrapper title bar visible in the mockup are HOST context
// only: they are NOT drawn here. FL Studio creates its own wrapper.
// The FL Studio fruit logo is NOT used: the header shows a custom
// Melotheon "M" monogram drawn with juce::Graphics.
//
// Phase 1 is the VISUAL interface only. Every button, knob, display,
// dropdown, meter and panel is a real JUCE component, but:
//   * Record / Play / Stop change their visual selected state only.
//   * ANALYZE, IMPORT AUDIO, DRAG MIDI TO FL and SEND TO CHANNEL have
//     empty callbacks (TODO: PHASE_2_AUDIO_LOGIC).
//   * No pitch detection, audio analysis, MIDI generation, instrument
//     hosting, recording, choir generation or AI exists in this phase.
//   * The interface never claims real analysis or MIDI export happened.
//
// Layout: all bounds are computed in resized() from the 1400 x 560
// base coordinates in melotheon::Tokens, scaled uniformly so the
// aspect ratio is preserved. The region proportions were measured
// from the reference image (image wins over the written spec numbers
// where they differ). paint() draws only backgrounds, borders,
// separators and the static header monogram / brand text.
//
// Text encoding: all UI strings are plain ASCII, except the brand
// "MELOTHEON" (see melotheon::brandTextUtf8, explicit UTF-8 bytes).

#include <JuceHeader.h>
#include <memory>
#include "PluginProcessor.h"
#include "UI/MelotheonLookAndFeel.h"
#include "UI/PitchEditorComponent.h"
#include "UI/LevelMeterComponent.h"
#include "UI/IconButton.h"

class MelotheonAudioProcessorEditor : public juce::AudioProcessorEditor,
                                      private juce::Timer
{
public:
    explicit MelotheonAudioProcessorEditor (MelotheonAudioProcessor& processor);
    ~MelotheonAudioProcessorEditor() override;

    void paint (juce::Graphics& g) override;
    void resized() override;

private:
    void timerCallback() override;

    // Scaled rectangle from base 1400 x 560 design coordinates.
    juce::Rectangle<int> scaledRect (int x, int y, int width, int height) const;
    void styleTextButton (juce::TextButton& button, const juce::Colour& fill,
                          const juce::Colour& textColour);
    void styleKnob (juce::Slider& knob);
    void styleComboBox (juce::ComboBox& box);
    void styleSectionLabel (juce::Label& label);
    void styleLcdLabel (juce::Label& label);
    void updateKeyLabel();

    MelotheonAudioProcessor& audioProcessor;
    MelotheonLookAndFeel lookAndFeel;
    float scaleFactor = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    juce::String lastKeyText;

    // Header.
    IconButton recordButton { IconButton::Kind::Record };
    IconButton playButton { IconButton::Kind::Play };
    IconButton stopButton { IconButton::Kind::Stop };
    juce::ComboBox inputSourceBox;
    juce::Label statusLabel;
    juce::Label noteLabel;

    // Left capture panel.
    juce::Label liveInputLabel;
    juce::TextButton importAudioButton { "IMPORT AUDIO" };
    IconButton importFolderButton { IconButton::Kind::Folder };
    juce::ToggleButton monoButton { "MONO" };
    juce::Label noiseGateLabel;
    LevelMeterComponent levelMeter;
    juce::Slider noiseGateKnob;

    // Center pitch editor.
    PitchEditorComponent pitchEditor;

    // Right instrument panel.
    juce::Label instrumentLabel;
    juce::ComboBox instrumentBox;
    IconButton instrumentFolderButton { IconButton::Kind::Folder };
    juce::ToggleButton midiOutButton { "MIDI OUT" };
    juce::ComboBox midiChannelBox;
    IconButton dragMidiButton { IconButton::Kind::DragFile };
    IconButton playOriginalButton { IconButton::Kind::PlayText };
    IconButton playInstrumentButton { IconButton::Kind::PlayText };

    // Bottom correction strip.
    juce::Slider centerKnob, variationKnob, transitionKnob, sensitivityKnob, quantizeKnob;
    juce::Label centerLabel, variationLabel, transitionLabel, sensitivityLabel, quantizeLabel;
    juce::Label keyLabel;
    juce::TextButton analyzeButton { "ANALYZE" };
    juce::TextButton sendToChannelButton { "SEND TO CHANNEL" };

    // Parameter attachments (editor state persists + becomes
    // automatable later; no DSP reads these in Phase 1).
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> centerAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> variationAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> transitionAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> sensitivityAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> quantizeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> noiseGateAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> monoAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> midiOutAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> playOriginalAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> playInstrumentAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> inputSourceAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> instrumentAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> midiChannelAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MelotheonAudioProcessorEditor)
};
