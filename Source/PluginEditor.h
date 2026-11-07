#pragma once

// MelotheonAudioProcessorEditor - the plugin UI (Phase 3C).
//
// Reference: the attached "MELODY CAPTURE" mockup (dark FL Studio-style
// panels, orange active states, green status displays, central piano
// roll). The FL Studio toolbar, Browser, Channel Rack, Playlist and
// outer plugin-wrapper title bar visible in the mockup are HOST context
// only: they are NOT drawn here. FL Studio creates its own wrapper.
// The FL Studio fruit logo is NOT used: the header shows the Melotheon
// soundwave-M (logo #5: vertical waveform bars forming an M, orange to
// gold) drawn with juce::Graphics.
//
// Phase 2 connects the audio foundation to the Phase 1 interface:
// Record captures host audio, Import Audio loads WAV / AIFF / FLAC
// (+ MP3 when JUCE enables it), the meter shows real input level,
// the waveform shows the real clip, and Play / Stop / Play Original
// preview it. The layout, colours, sizes and brand text are unchanged
// from Phase 1. ANALYZE now runs offline monophonic pitch tracking and
// paints its curve/confidence and stable note blocks. The editor also
// loads a local one-shot sound, previews the detected melody with it,
// exports a standard MIDI file, and offers vertical piano zoom/scroll.
// SEND TO CHANNEL remains a later host-integration phase.
//
// The editor polls the processor on a ~30 Hz timer (meter, playhead,
// status, clip version) and never touches the audio thread directly.
// All audio work is triggered through the processor's message-thread
// API; the processor owns the engines and the active clip.
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

    // Open the import FileChooser (WAV / AIFF / FLAC, + MP3 when
    // JUCE enables it). The chooser is kept alive in fileChooser
    // until its async callback runs.
    void startImport();
    void startInstrumentImport();
    void startMidiExport();

    MelotheonAudioProcessor& audioProcessor;
    MelotheonLookAndFeel lookAndFeel;
    float scaleFactor = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    juce::String lastKeyText;

    // Polled state (timerCallback only): -1 / empty forces a refresh
    // on the first tick, so an editor opened after a clip was loaded
    // still shows it.
    int lastClipVersionSeen = -1;
    int lastAnalysisVersionSeen = -1;
    int lastMelodyNotesVersionSeen = -1;
    int lastInstrumentVersionSeen = -1;
    int lastStatusSeen = -1;
    juce::String lastStatusText;
    juce::String lastNoteText;
    juce::String lastAnalyzeButtonText;
    bool lastPlayOriginalParam = false;
    bool lastPlayInstrumentParam = false;
    std::shared_ptr<const PitchAnalysisResult> displayedAnalysis;
    std::shared_ptr<const MelodyNoteSequence> displayedNotes;

    // The async callback keeps a shared owner until the native chooser
    // closes. A Component::SafePointer protects access to this editor.
    std::shared_ptr<juce::FileChooser> fileChooser;

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
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> midiChannelAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MelotheonAudioProcessorEditor)
};
