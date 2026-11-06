// MelotheonAudioProcessorEditor implementation (Phase 1).
// See PluginEditor.h for the layout contract and the Phase 1 limits:
// visual interaction only, no audio or MIDI behaviour yet.

#include "PluginEditor.h"

namespace
{
// Base-coordinate layout (1400 x 560 design, measured from the
// reference image: the image wins over the written spec numbers
// where they differ).
// Header.
constexpr int kRecordX = 465, kRecordY = 22, kRecordSize = 54;
constexpr int kPlayX = 533, kStopX = 601;
constexpr int kInputBoxX = 685, kInputBoxY = 26, kInputBoxW = 170, kInputBoxH = 46;
constexpr int kStatusX = 880, kStatusY = 26, kStatusW = 185, kStatusH = 46;
constexpr int kNoteX = 1110, kNoteW = 280;
// Left panel.
constexpr int kLiveLabelX = 24, kLiveLabelY = 110, kLiveLabelW = 140, kLiveLabelH = 20;
constexpr int kImportX = 24, kImportY = 136, kImportW = 180, kImportH = 36;
constexpr int kImportFolderX = 210, kImportFolderSize = 36;
constexpr int kMonoX = 24, kMonoY = 184, kMonoW = 120, kMonoH = 24;
constexpr int kGateLabelX = 112, kGateLabelY = 262, kGateLabelW = 130, kGateLabelH = 20;
constexpr int kMeterX = 28, kMeterY = 246, kMeterW = 22, kMeterH = 168;
constexpr int kGateKnobX = 115, kGateKnobY = 286, kGateKnobSize = 64;
// Center panel (pitch editor has an 8 px inset inside its panel).
constexpr int kPitchX = 266, kPitchY = 106, kPitchW = 794, kPitchH = 322;
// Right panel.
constexpr int kRightLabelX = 1092, kRightLabelY = 110, kRightLabelW = 140, kRightLabelH = 20;
constexpr int kInstrBoxX = 1092, kInstrBoxY = 134, kInstrBoxW = 238, kInstrBoxH = 34;
constexpr int kInstrFolderX = 1340, kInstrFolderSize = 34;
constexpr int kMidiOutX = 1092, kMidiOutY = 180, kMidiOutW = 140, kMidiOutH = 24;
constexpr int kChannelBoxX = 1092, kChannelBoxY = 210, kChannelBoxW = 238, kChannelBoxH = 34;
constexpr int kDragX = 1092, kDragY = 258, kDragW = 258, kDragH = 40;
constexpr int kPlayOrigY = 306, kPlayInstrY = 354;
// Bottom strip.
constexpr int kKnobSize = 64, kKnobY = 450, kKnobLabelY = 516, kKnobLabelH = 20;
constexpr int kKnobCentres[] = { 108, 252, 397, 541, 686 };
constexpr int kKeyX = 880, kKeyY = 468, kKeyW = 190, kKeyH = 44;
constexpr int kAnalyzeX = 1095, kAnalyzeY = 468, kAnalyzeW = 140, kAnalyzeH = 44;
constexpr int kSendX = 1245, kSendY = 468, kSendW = 145, kSendH = 44;
} // namespace

MelotheonAudioProcessorEditor::MelotheonAudioProcessorEditor (MelotheonAudioProcessor& processorToUse)
    : juce::AudioProcessorEditor (processorToUse), audioProcessor (processorToUse)
{
    using namespace melotheon;
    setLookAndFeel (&lookAndFeel);

    // Popup menu colours (combo box dropdowns).
    lookAndFeel.setColour (juce::PopupMenu::backgroundColourId, Tokens::raisedPanel);
    lookAndFeel.setColour (juce::PopupMenu::textColourId, Tokens::primaryText);
    lookAndFeel.setColour (juce::PopupMenu::highlightedBackgroundColourId, Tokens::activeOrange);
    lookAndFeel.setColour (juce::PopupMenu::highlightedTextColourId, juce::Colours::black);

    // ---- Header ----
    addAndMakeVisible (recordButton);
    addAndMakeVisible (playButton);
    addAndMakeVisible (stopButton);
    recordButton.setClickingTogglesState (false);
    playButton.setClickingTogglesState (false);
    stopButton.setClickingTogglesState (false);
    // Record opens orange (selected) to match the reference image:
    // a solid orange record button from the moment the plugin opens.
    // Clicking it toggles it off; clicking Play untoggles it.
    recordButton.setToggleState (true, juce::dontSendNotification);
    // Visual selected state only: no audio operation is started.
    recordButton.onClick = [this]
    {
        const bool on = ! recordButton.getToggleState();
        recordButton.setToggleState (on, juce::dontSendNotification);
        if (on)
            playButton.setToggleState (false, juce::dontSendNotification);
    };
    playButton.onClick = [this]
    {
        const bool on = ! playButton.getToggleState();
        playButton.setToggleState (on, juce::dontSendNotification);
        if (on)
            recordButton.setToggleState (false, juce::dontSendNotification);
    };
    stopButton.onClick = [this]
    {
        recordButton.setToggleState (false, juce::dontSendNotification);
        playButton.setToggleState (false, juce::dontSendNotification);
    };

    addAndMakeVisible (inputSourceBox);
    styleComboBox (inputSourceBox);
    inputSourceBox.addItem ("INPUT: VOICE", 1);
    inputSourceBox.addItem ("INPUT: GUITAR", 2);
    inputSourceBox.addItem ("INPUT: PIANO", 3);
    inputSourceBox.addItem ("INPUT: LINE", 4);

    addAndMakeVisible (statusLabel);
    styleLcdLabel (statusLabel);
    statusLabel.setText ("READY", juce::dontSendNotification);

    addAndMakeVisible (noteLabel);
    styleLcdLabel (noteLabel);
    noteLabel.setText ("A3   220.0 Hz", juce::dontSendNotification);

    // ---- Left capture panel ----
    addAndMakeVisible (liveInputLabel);
    styleSectionLabel (liveInputLabel);
    liveInputLabel.setText ("LIVE INPUT", juce::dontSendNotification);

    addAndMakeVisible (importAudioButton);
    styleTextButton (importAudioButton, Tokens::raisedPanel, Tokens::primaryText);
    importAudioButton.onClick = []
    {
        // TODO: PHASE_2_AUDIO_LOGIC
    };

    addAndMakeVisible (importFolderButton);
    importFolderButton.onClick = []
    {
        // TODO: PHASE_2_AUDIO_LOGIC
    };

    addAndMakeVisible (monoButton);
    addAndMakeVisible (noiseGateLabel);
    styleSectionLabel (noiseGateLabel);
    noiseGateLabel.setText ("NOISE GATE", juce::dontSendNotification);

    addAndMakeVisible (levelMeter);
    addAndMakeVisible (noiseGateKnob);
    styleKnob (noiseGateKnob);

    // ---- Center pitch editor ----
    addAndMakeVisible (pitchEditor);

    // ---- Right instrument panel ----
    addAndMakeVisible (instrumentLabel);
    styleSectionLabel (instrumentLabel);
    instrumentLabel.setText ("INSTRUMENT", juce::dontSendNotification);

    addAndMakeVisible (instrumentBox);
    styleComboBox (instrumentBox);
    instrumentBox.addItem ("FLEX GRAND PIANO", 1);
    instrumentBox.addItem ("GRAND PIANO", 2);
    instrumentBox.addItem ("ELECTRIC PIANO", 3);
    instrumentBox.addItem ("STRINGS", 4);
    instrumentBox.addItem ("CHOIR PAD", 5);

    addAndMakeVisible (instrumentFolderButton);
    instrumentFolderButton.onClick = []
    {
        // TODO: PHASE_2_AUDIO_LOGIC
    };

    addAndMakeVisible (midiOutButton);

    addAndMakeVisible (midiChannelBox);
    styleComboBox (midiChannelBox);
    for (int ch = 1; ch <= 16; ++ch)
        midiChannelBox.addItem ("CHANNEL " + juce::String (ch), ch);

    addAndMakeVisible (dragMidiButton);
    dragMidiButton.setText ("DRAG MIDI TO FL");
    dragMidiButton.onClick = []
    {
        // TODO: PHASE_2_AUDIO_LOGIC
    };

    addAndMakeVisible (playOriginalButton);
    playOriginalButton.setText ("PLAY ORIGINAL");
    playOriginalButton.setClickingTogglesState (true);

    addAndMakeVisible (playInstrumentButton);
    playInstrumentButton.setText ("PLAY INSTRUMENT");
    playInstrumentButton.setClickingTogglesState (true);

    // ---- Bottom correction strip ----
    addAndMakeVisible (centerKnob);
    addAndMakeVisible (variationKnob);
    addAndMakeVisible (transitionKnob);
    addAndMakeVisible (sensitivityKnob);
    addAndMakeVisible (quantizeKnob);
    styleKnob (centerKnob);
    styleKnob (variationKnob);
    styleKnob (transitionKnob);
    styleKnob (sensitivityKnob);
    styleKnob (quantizeKnob);

    const std::pair<juce::Label*, const char*> knobLabels[] = {
        { &centerLabel, "CENTER" },
        { &variationLabel, "VARIATION" },
        { &transitionLabel, "TRANSITION" },
        { &sensitivityLabel, "SENSITIVITY" },
        { &quantizeLabel, "QUANTIZE" },
    };
    for (auto& [label, text] : knobLabels)
    {
        addAndMakeVisible (*label);
        label->setText (text, juce::dontSendNotification);
        label->setJustificationType (juce::Justification::centred);
        label->setColour (juce::Label::textColourId, Tokens::secondaryText);
        label->setFont (melotheonFont (Tokens::smallLabelSize, true));
    }

    addAndMakeVisible (keyLabel);
    styleLcdLabel (keyLabel);
    keyLabel.setText ("KEY: C MINOR", juce::dontSendNotification);

    addAndMakeVisible (analyzeButton);
    styleTextButton (analyzeButton, Tokens::activeOrange, juce::Colours::black);
    analyzeButton.onClick = []
    {
        // TODO: PHASE_2_AUDIO_LOGIC
    };

    addAndMakeVisible (sendToChannelButton);
    styleTextButton (sendToChannelButton, Tokens::raisedPanel, Tokens::primaryText);
    sendToChannelButton.onClick = []
    {
        // TODO: PHASE_2_AUDIO_LOGIC
    };

    // ---- Parameter attachments (after all items exist). ----
    auto& apvts = audioProcessor.apvts;
    centerAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "center", centerKnob);
    variationAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "variation", variationKnob);
    transitionAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "transition", transitionKnob);
    sensitivityAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "sensitivity", sensitivityKnob);
    quantizeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "quantize", quantizeKnob);
    noiseGateAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "noiseGate", noiseGateKnob);
    monoAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        apvts, "mono", monoButton);
    midiOutAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        apvts, "midiOut", midiOutButton);
    playOriginalAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        apvts, "playOriginal", playOriginalButton);
    playInstrumentAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        apvts, "playInstrument", playInstrumentButton);
    inputSourceAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "inputSource", inputSourceBox);
    instrumentAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "instrumentTarget", instrumentBox);
    midiChannelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "midiChannel", midiChannelBox);

    updateKeyLabel();
    startTimer (200);

    setResizable (true, true);
    setResizeLimits (1000, 400, 1800, 720);
    if (auto* boundsConstrainer = getConstrainer())
        boundsConstrainer->setFixedAspectRatio (static_cast<double> (melotheon::designWidth)
                                                / static_cast<double> (melotheon::designHeight));
    setSize (melotheon::designWidth, melotheon::designHeight);
}

MelotheonAudioProcessorEditor::~MelotheonAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

juce::Rectangle<int> MelotheonAudioProcessorEditor::scaledRect (int x, int y, int width,
                                                                int height) const
{
    return {
        static_cast<int> (offsetX + static_cast<float> (x) * scaleFactor),
        static_cast<int> (offsetY + static_cast<float> (y) * scaleFactor),
        static_cast<int> (static_cast<float> (width) * scaleFactor),
        static_cast<int> (static_cast<float> (height) * scaleFactor),
    };
}

void MelotheonAudioProcessorEditor::styleTextButton (juce::TextButton& button,
                                                    const juce::Colour& fill,
                                                    const juce::Colour& textColour)
{
    button.setColour (juce::TextButton::buttonColourId, fill);
    button.setColour (juce::TextButton::textColourOffId, textColour);
    button.setColour (juce::TextButton::textColourOnId, textColour);
}

void MelotheonAudioProcessorEditor::styleKnob (juce::Slider& knob)
{
    knob.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    knob.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    knob.setRange (0.0, 1.0, 0.01);
    knob.setRotaryParameters (juce::MathConstants<float>::pi * 1.25f,
                              juce::MathConstants<float>::pi * 2.75f, true);
}

void MelotheonAudioProcessorEditor::styleComboBox (juce::ComboBox& box)
{
    box.setEditableText (false);
    box.setJustificationType (juce::Justification::centredLeft);
    box.setColour (juce::ComboBox::textColourId, melotheon::Tokens::primaryText);
    box.setColour (juce::ComboBox::backgroundColourId, melotheon::Tokens::raisedPanel);
    box.setColour (juce::ComboBox::outlineColourId, melotheon::Tokens::border);
    box.setColour (juce::ComboBox::arrowColourId, melotheon::Tokens::primaryText);
}

void MelotheonAudioProcessorEditor::styleSectionLabel (juce::Label& label)
{
    label.setJustificationType (juce::Justification::left);
    label.setColour (juce::Label::textColourId, melotheon::Tokens::primaryText);
    label.setFont (melotheon::melotheonFont (melotheon::Tokens::labelSize, true));
}

void MelotheonAudioProcessorEditor::styleLcdLabel (juce::Label& label)
{
    label.setJustificationType (juce::Justification::centred);
    label.setColour (juce::Label::backgroundColourId, melotheon::Tokens::lcdBackground);
    label.setColour (juce::Label::outlineColourId, melotheon::Tokens::border);
    label.setColour (juce::Label::textColourId, melotheon::Tokens::statusGreen);
    label.setFont (melotheon::melotheonFont (melotheon::Tokens::lcdSize, true));
}

void MelotheonAudioProcessorEditor::updateKeyLabel()
{
    static const char* keyNames[] = { "C", "C#", "D", "D#", "E", "F",
                                      "F#", "G", "G#", "A", "A#", "B" };
    static const char* scaleNames[] = { "MAJOR", "MINOR", "DORIAN",
                                        "MIXOLYDIAN", "PHRYGIAN", "LYDIAN" };
    const int keyIndex = static_cast<int> (
        audioProcessor.apvts.getRawParameterValue ("key")->load());
    const int scaleIndex = static_cast<int> (
        audioProcessor.apvts.getRawParameterValue ("scale")->load());
    const juce::String text = "KEY: "
        + juce::String (keyNames[juce::jlimit (0, 11, keyIndex)]) + " "
        + juce::String (scaleNames[juce::jlimit (0, 5, scaleIndex)]);
    if (text != lastKeyText)
    {
        lastKeyText = text;
        keyLabel.setText (text, juce::dontSendNotification);
    }
}

void MelotheonAudioProcessorEditor::timerCallback()
{
    updateKeyLabel();
}

void MelotheonAudioProcessorEditor::resized()
{
    using namespace melotheon;
    scaleFactor = juce::jmin (static_cast<float> (getWidth()) / designWidth,
                              static_cast<float> (getHeight()) / designHeight);
    offsetX = (static_cast<float> (getWidth()) - designWidth * scaleFactor) * 0.5f;
    offsetY = (static_cast<float> (getHeight()) - designHeight * scaleFactor) * 0.5f;

    // Header.
    recordButton.setBounds (scaledRect (kRecordX, kRecordY, kRecordSize, kRecordSize));
    playButton.setBounds (scaledRect (kPlayX, kRecordY, kRecordSize, kRecordSize));
    stopButton.setBounds (scaledRect (kStopX, kRecordY, kRecordSize, kRecordSize));
    inputSourceBox.setBounds (scaledRect (kInputBoxX, kInputBoxY, kInputBoxW, kInputBoxH));
    statusLabel.setBounds (scaledRect (kStatusX, kStatusY, kStatusW, kStatusH));
    noteLabel.setBounds (scaledRect (kNoteX, kStatusY, kNoteW, kStatusH));

    // Left capture panel.
    liveInputLabel.setBounds (scaledRect (kLiveLabelX, kLiveLabelY, kLiveLabelW, kLiveLabelH));
    importAudioButton.setBounds (scaledRect (kImportX, kImportY, kImportW, kImportH));
    importFolderButton.setBounds (scaledRect (kImportFolderX, kImportY,
                                              kImportFolderSize, kImportFolderSize));
    monoButton.setBounds (scaledRect (kMonoX, kMonoY, kMonoW, kMonoH));
    noiseGateLabel.setBounds (scaledRect (kGateLabelX, kGateLabelY, kGateLabelW, kGateLabelH));
    levelMeter.setBounds (scaledRect (kMeterX, kMeterY, kMeterW, kMeterH));
    noiseGateKnob.setBounds (scaledRect (kGateKnobX, kGateKnobY, kGateKnobSize, kGateKnobSize));

    // Center pitch editor.
    pitchEditor.setBounds (scaledRect (kPitchX, kPitchY, kPitchW, kPitchH));

    // Right instrument panel.
    instrumentLabel.setBounds (scaledRect (kRightLabelX, kRightLabelY, kRightLabelW, kRightLabelH));
    instrumentBox.setBounds (scaledRect (kInstrBoxX, kInstrBoxY, kInstrBoxW, kInstrBoxH));
    instrumentFolderButton.setBounds (scaledRect (kInstrFolderX, kInstrBoxY,
                                                  kInstrFolderSize, kInstrFolderSize));
    midiOutButton.setBounds (scaledRect (kMidiOutX, kMidiOutY, kMidiOutW, kMidiOutH));
    midiChannelBox.setBounds (scaledRect (kChannelBoxX, kChannelBoxY, kChannelBoxW, kChannelBoxH));
    dragMidiButton.setBounds (scaledRect (kDragX, kDragY, kDragW, kDragH));
    playOriginalButton.setBounds (scaledRect (kDragX, kPlayOrigY, kDragW, kDragH));
    playInstrumentButton.setBounds (scaledRect (kDragX, kPlayInstrY, kDragW, kDragH));

    // Bottom correction strip.
    juce::Slider* knobs[] = { &centerKnob, &variationKnob, &transitionKnob,
                              &sensitivityKnob, &quantizeKnob };
    juce::Label* labels[] = { &centerLabel, &variationLabel, &transitionLabel,
                              &sensitivityLabel, &quantizeLabel };
    for (int i = 0; i < 5; ++i)
    {
        const int centreX = kKnobCentres[i];
        knobs[i]->setBounds (scaledRect (centreX - kKnobSize / 2, kKnobY, kKnobSize, kKnobSize));
        labels[i]->setBounds (scaledRect (centreX - 48, kKnobLabelY, 96, kKnobLabelH));
    }
    keyLabel.setBounds (scaledRect (kKeyX, kKeyY, kKeyW, kKeyH));
    analyzeButton.setBounds (scaledRect (kAnalyzeX, kAnalyzeY, kAnalyzeW, kAnalyzeH));
    sendToChannelButton.setBounds (scaledRect (kSendX, kSendY, kSendW, kSendH));
}

void MelotheonAudioProcessorEditor::paint (juce::Graphics& g)
{
    using namespace melotheon;
    const float scale = juce::jmin (static_cast<float> (getWidth()) / designWidth,
                                    static_cast<float> (getHeight()) / designHeight);
    const float ox = (static_cast<float> (getWidth()) - designWidth * scale) * 0.5f;
    const float oy = (static_cast<float> (getHeight()) - designHeight * scale) * 0.5f;
    auto sr = [&] (int x, int y, int w, int h) -> juce::Rectangle<float>
    {
        return { ox + static_cast<float> (x) * scale,
                 oy + static_cast<float> (y) * scale,
                 static_cast<float> (w) * scale,
                 static_cast<float> (h) * scale };
    };

    g.fillAll (Tokens::background);

    // Five layout panels: fill + 1 px border.
    const juce::Rectangle<int> panels[] = {
        { headerX, headerY, headerW, headerH },
        { leftX, leftY, leftW, leftH },
        { centerX, centerY, centerW, centerH },
        { rightX, rightY, rightW, rightH },
        { bottomX, bottomY, bottomW, bottomH },
    };
    for (const auto& panel : panels)
    {
        auto area = sr (panel.getX(), panel.getY(), panel.getWidth(), panel.getHeight());
        g.setColour (Tokens::panel);
        g.fillRoundedRectangle (area, Tokens::panelCornerRadius * scale);
        g.setColour (Tokens::border);
        g.drawRoundedRectangle (area, Tokens::panelCornerRadius * scale, 1.0f);
    }

    // Header: Melotheon soundwave "M" (vector version of the chosen
    // logo #5: vertical waveform bars forming an M, orange to gold).
    // Drawn with juce::Graphics so it stays crisp at any size and
    // needs no image asset; no FL Studio logo is used.
    {
        auto logoArea = sr (22, 22, 54, 54);
        g.setColour (Tokens::knobBody);
        g.fillRoundedRectangle (logoArea, Tokens::cornerRadius * scale);
        g.setColour (Tokens::border);
        g.drawRoundedRectangle (logoArea, Tokens::cornerRadius * scale, 1.0f);

        // Bar heights trace an M: tall at both sides, dip in the middle.
        static constexpr float barHeights[] = {
            0.55f, 0.80f, 0.95f, 1.00f, 0.85f, 0.65f, 0.50f, 0.35f,
            0.45f, 0.60f, 0.80f, 0.95f, 1.00f, 0.85f, 0.55f,
        };
        constexpr int numBars = 15;
        const float x = logoArea.getX(), y = logoArea.getY();
        const float w = logoArea.getWidth(), h = logoArea.getHeight();
        const float padX = w * 0.10f, padY = h * 0.12f;
        const float innerW = w - padX * 2.0f, innerH = h - padY * 2.0f;
        const float stepX = innerW / static_cast<float> (numBars);
        const float barW = stepX * 0.62f;
        const float centreY = y + h * 0.5f;
        for (int i = 0; i < numBars; ++i)
        {
            const float t = static_cast<float> (i) / static_cast<float> (numBars - 1);
            const float barH = barHeights[i] * innerH;
            const float barX = x + padX + static_cast<float> (i) * stepX
                             + (stepX - barW) * 0.5f;
            // Orange on the left, gold on the right (logo #5 gradient:
            // #FF8A1F -> #FFC94D).
            g.setColour (juce::Colour::fromRGB (
                static_cast<juce::uint8> (255),
                static_cast<juce::uint8> (138.0f + t * 63.0f),
                static_cast<juce::uint8> (31.0f + t * 46.0f)));
            g.fillRect (barX, centreY - barH * 0.5f, barW, barH);
        }
    }

    // Header: brand text MELOTHEON.
    g.setFont (melotheonFont (Tokens::brandSize, true));
    g.setColour (Tokens::primaryText);
    g.drawText (juce::String (brandTextUtf8), sr (86, 22, 320, 54),
                juce::Justification::centredLeft, true);

    // Thin vertical separator between the brand and the transport.
    g.setColour (Tokens::border);
    g.drawLine (ox + 445.0f * scale, oy + 22.0f * scale,
                ox + 445.0f * scale, oy + 66.0f * scale, 1.0f);
}
