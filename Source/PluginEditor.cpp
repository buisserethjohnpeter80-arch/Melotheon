// MelotheonAudioProcessorEditor implementation (Phase 3C).
// See PluginEditor.h for the layout contract: Phase 1 interface with
// the audio foundation, monophonic pitch analysis, MIDI export and
// local sample-instrument melody preview.

#include "PluginEditor.h"

#include "Audio/AudioFileLoader.h"

#include <algorithm>
#include <cmath>
#include <functional>

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

juce::String formatPitchFrame (const PitchFrame& frame)
{
    static const char* noteNames[] = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    const int midi = juce::roundToInt (frame.midiPitch);
    const int pitchClass = ((midi % 12) + 12) % 12;
    const int octave = midi / 12 - 1;
    return juce::String (noteNames[pitchClass]) + juce::String (octave)
         + "   " + juce::String (frame.frequencyHz, 1) + " Hz   "
         + juce::String (juce::roundToInt (frame.confidence * 100.0f)) + "%";
}

// ---- MIDI chip popup panel (Feature A) ----
// Small two-row panel opened by the MIDI chip: the top row starts a
// real OS file drag of a temp .mid, the bottom row runs the existing
// save-file export. Rows paint themselves with the existing dark
// tokens, so the panel needs no LookAndFeel and stays safe if the
// editor closes while it is open (all callbacks use SafePointers).
//
// Drag lifetime: the temp file is owned by the EDITOR, not the row.
// The row only holds the path for the drag; the editor deletes the
// file in finishMidiDrag(), which the drag callback always reaches
// (its std::function copy survives the row). The panel is not
// dismissable while a drag is in flight.
class MidiPanelRow : public juce::Component
{
public:
    MidiPanelRow (const juce::String& textToUse, float fontSizeToUse)
        : text (textToUse), fontSize (fontSizeToUse)
    {
    }

    // Plain click (export row). Invoked on mouseUp when the press
    // started on this row and barely moved.
    std::function<void()> onClick;

    // Drag row: a plain click (no drag) used to do nothing visible.
    // It now publishes a status hint so the user knows to press and
    // drag the row into FL Studio.
    std::function<void()> onClickHint;

    // Drag row: called once when a drag gesture passes the
    // threshold. Must return the temp .mid to drag, or an empty File
    // when there is no melody (the processor already set the
    // ANALYZE FIRST status then).
    std::function<juce::File()> prepareDragFile;

    // Drag row: called once when the native drag starts. The editor
    // marks the drag in flight so the panel cannot dismiss under it.
    std::function<void()> onDragStarted;

    // Drag row: called on the message thread when the native drag
    // ends (dropped or cancelled) or immediately when it could not
    // start. The editor deletes the temp file here. This callback is
    // copied into the drag lambda, so it still runs if the row (and
    // its panel) was destroyed mid-drag.
    std::function<void()> onDragFinished;

    // Drag row: called after the native drag ends (dropped or
    // cancelled) or immediately when the drag could not start.
    std::function<void()> onDragEnded;

    // Drag row: called instead of onDragEnded when there was no
    // file to drag (no melody). The panel should close.
    std::function<void()> onDragFailed;

    void setDraggable (bool shouldDrag) { draggable = shouldDrag; }

    void paint (juce::Graphics& g) override
    {
        // While a drag is in flight the row stays visibly highlighted
        // (brighter fill + orange text) so the user can see the drag
        // is live while moving toward FL Studio.
        g.fillAll (dragStarted ? melotheon::Tokens::raisedPanel.brighter (0.22f)
                  : isHovered ? melotheon::Tokens::raisedPanel.brighter (0.10f)
                              : melotheon::Tokens::raisedPanel);
        g.setColour (melotheon::Tokens::border);
        g.drawRect (getLocalBounds(), 1);
        g.setColour ((dragStarted || isHovered) ? melotheon::Tokens::activeOrange
                                                : melotheon::Tokens::primaryText);
        g.setFont (melotheon::melotheonFont (fontSize, true));
        g.drawText (text, getLocalBounds().reduced (12, 0),
                    juce::Justification::centredLeft, true);
    }

    void mouseEnter (const juce::MouseEvent&) override
    {
        isHovered = true;
        repaint();
    }

    void mouseExit (const juce::MouseEvent&) override
    {
        isHovered = false;
        repaint();
    }

    void mouseDown (const juce::MouseEvent& event) override
    {
        pressPosition = event.position;
        dragStarted = false;
    }

    void mouseDrag (const juce::MouseEvent& event) override
    {
        if (! draggable || dragStarted)
            return;
        if (event.position.getDistanceFrom (pressPosition) < 6.0f)
            return;

        dragStarted = true;
        repaint(); // the row highlights while the drag is live
        dragFile = prepareDragFile != nullptr ? prepareDragFile() : juce::File{};
        if (dragFile == juce::File{})
        {
            if (onDragFailed != nullptr)
                onDragFailed();
            return;
        }

        if (onDragStarted != nullptr)
            onDragStarted();

        // On Windows this runs the native drag and invokes the
        // callback when the drag ends, so the temp file must live
        // until then. Deletion happens in the editor (onDragFinished),
        // never here: this row may be destroyed before the drag ends.
        juce::Component::SafePointer<MidiPanelRow> safeThis (this);
        auto dragFinished = onDragFinished; // copy: survives this row
        const bool started = juce::DragAndDropContainer::performExternalDragDropOfFiles (
            juce::StringArray { dragFile.getFullPathName() }, false, this,
            [safeThis, dragFinished]
            {
                juce::MessageManager::callAsync ([safeThis, dragFinished]
                {
                    if (dragFinished != nullptr)
                        dragFinished();
                    if (safeThis != nullptr)
                        safeThis->finishDrag();
                });
            });
        if (! started)
        {
            if (onDragFinished != nullptr)
                onDragFinished();
            finishDrag();
        }
    }

    void mouseUp (const juce::MouseEvent& event) override
    {
        if (draggable)
        {
            // A plain click on the drag row never silently does
            // nothing: it shows how to drag. For the drag row,
            // dragStarted is cleared by finishDrag() once the native
            // drag ends; clearing it here could allow a second drag
            // while the first is still running.
            if (! dragStarted
                && event.position.getDistanceFrom (pressPosition) < 6.0f
                && onClickHint != nullptr)
                onClickHint();
            return;
        }
        if (event.position.getDistanceFrom (pressPosition) < 6.0f
            && onClick != nullptr)
            onClick();
    }

private:
    // Clear local drag state and tell the panel the drag is over.
    // Called on the message thread only, after onDragFinished ran
    // (the editor already deleted the temp file by then).
    void finishDrag()
    {
        dragFile = juce::File{};
        dragStarted = false;
        repaint();
        if (onDragEnded != nullptr)
            onDragEnded();
    }

    juce::String text;
    float fontSize;
    bool isHovered = false;
    bool draggable = false;
    bool dragStarted = false;
    juce::Point<float> pressPosition;
    juce::File dragFile;
};

class MidiActionPanel : public juce::Component
{
public:
    MidiActionPanel (int widthToUse, int rowHeightToUse, float fontSizeToUse)
        : dragRow ("DRAG MIDI TO FL", fontSizeToUse),
          exportRow ("EXPORT MIDI", fontSizeToUse),
          rowHeight (rowHeightToUse),
          captionHeight (juce::jmax (16, rowHeightToUse / 2))
    {
        // A caption under the rows, visible before any click, so the
        // user knows the drag row must be pressed and dragged - a
        // plain click on it does nothing by itself.
        captionLabel.setText ("PRESS AND DRAG THIS ROW INTO FL STUDIO",
                              juce::dontSendNotification);
        captionLabel.setJustificationType (juce::Justification::centred);
        captionLabel.setColour (juce::Label::textColourId,
                                melotheon::Tokens::secondaryText);
        captionLabel.setFont (melotheon::melotheonFont (fontSizeToUse * 0.75f, false));
        setSize (widthToUse, rowHeightToUse * 2 + captionHeight + 12);
        dragRow.setDraggable (true);
        dragRow.onClickHint = [this]
        {
            if (onClickHint != nullptr)
                onClickHint();
        };
        dragRow.prepareDragFile = [this]
        {
            return onPrepareDragFile != nullptr ? onPrepareDragFile() : juce::File{};
        };
        dragRow.onDragStarted = [this]
        {
            if (onDragStarted != nullptr)
                onDragStarted();
        };
        dragRow.onDragFinished = [this]
        {
            if (onDragFinished != nullptr)
                onDragFinished();
        };
        dragRow.onDragEnded = [this]
        {
            if (onDismissRequested != nullptr)
                onDismissRequested();
        };
        dragRow.onDragFailed = [this]
        {
            if (onDismissRequested != nullptr)
                onDismissRequested();
        };
        exportRow.onClick = [this]
        {
            if (onExportMidi != nullptr)
                onExportMidi();
        };
        addAndMakeVisible (dragRow);
        addAndMakeVisible (exportRow);
        addAndMakeVisible (captionLabel);
    }

    std::function<void()> onExportMidi;
    std::function<juce::File()> onPrepareDragFile;
    std::function<void()> onClickHint;
    std::function<void()> onDragStarted;
    std::function<void()> onDragFinished;
    std::function<void()> onDismissRequested;

    void paint (juce::Graphics& g) override
    {
        g.fillAll (melotheon::Tokens::panel);
        g.setColour (melotheon::Tokens::border);
        g.drawRect (getLocalBounds(), 1);
    }

    void resized() override
    {
        const int pad = 4;
        const int w = getWidth() - pad * 2;
        dragRow.setBounds (pad, pad, w, rowHeight);
        exportRow.setBounds (pad, pad + rowHeight, w, rowHeight);
        captionLabel.setBounds (pad, pad + rowHeight * 2 + 2, w, captionHeight);
    }

private:
    MidiPanelRow dragRow;
    MidiPanelRow exportRow;
    juce::Label captionLabel;
    int rowHeight;
    int captionHeight;
};
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

    // Record toggles a live take. It starts OFF: the button only
    // shows its active orange state while actually recording.
    recordButton.onClick = [this]
    {
        if (audioProcessor.isRecording())
        {
            audioProcessor.stopRecording(); // finalizes -> AUDIO READY
            recordButton.setToggleState (false, juce::dontSendNotification);
            playButton.setToggleState (false, juce::dontSendNotification);
            audioProcessor.setPlayOriginalParameter (false);
        }
        else if (audioProcessor.startRecording())
        {
            recordButton.setToggleState (true, juce::dontSendNotification);
            playButton.setToggleState (false, juce::dontSendNotification);
            audioProcessor.setPlayOriginalParameter (false);
        }
        else
        {
            // Not LIVE INPUT, or a take is already running: the
            // processor set the status message (e.g. USE LIVE INPUT).
            recordButton.setToggleState (false, juce::dontSendNotification);
        }
    };

    // Play toggles preview from the current internal position.
    // Pressing it again pauses (position kept); Stop resets it.
    playButton.onClick = [this]
    {
        if (audioProcessor.isPreviewPlaying())
        {
            audioProcessor.pausePreview();
            playButton.setToggleState (false, juce::dontSendNotification);
            audioProcessor.setPlayOriginalParameter (false);
        }
        else if (audioProcessor.startPreview())
        {
            playButton.setToggleState (true, juce::dontSendNotification);
            recordButton.setToggleState (false, juce::dontSendNotification);
            audioProcessor.setPlayOriginalParameter (true);
        }
        else
        {
            // No clip to play: the processor set EMPTY SOURCE.
            playButton.setToggleState (false, juce::dontSendNotification);
        }
    };

    // Stop finalizes a take in progress, otherwise stops preview
    // and returns the cursor to the beginning.
    stopButton.onClick = [this]
    {
        if (audioProcessor.isPitchAnalyzing())
            audioProcessor.cancelPitchAnalysis();
        else if (audioProcessor.isRecording())
            audioProcessor.stopRecording();
        else
            audioProcessor.stopPreview();

        recordButton.setToggleState (false, juce::dontSendNotification);
        playButton.setToggleState (false, juce::dontSendNotification);
        audioProcessor.setPlayOriginalParameter (false);
    };

    addAndMakeVisible (inputSourceBox);
    styleComboBox (inputSourceBox);
    inputSourceBox.addItem ("LIVE INPUT", 1);
    inputSourceBox.addItem ("IMPORTED AUDIO", 2);

    addAndMakeVisible (statusLabel);
    styleLcdLabel (statusLabel);
    statusLabel.setText ("READY", juce::dontSendNotification);

    addAndMakeVisible (noteLabel);
    styleLcdLabel (noteLabel);
    noteLabel.setText ("PITCH --", juce::dontSendNotification);

    // ---- Left capture panel ----
    addAndMakeVisible (liveInputLabel);
    styleSectionLabel (liveInputLabel);
    liveInputLabel.setText ("LIVE INPUT", juce::dontSendNotification);

    addAndMakeVisible (importAudioButton);
    styleTextButton (importAudioButton, Tokens::raisedPanel, Tokens::primaryText);
    importAudioButton.onClick = [this] { startImport(); };

    addAndMakeVisible (importFolderButton);
    importFolderButton.onClick = [this] { startImport(); };

    addAndMakeVisible (monoButton);
    addAndMakeVisible (noiseGateLabel);
    styleSectionLabel (noiseGateLabel);
    noiseGateLabel.setText ("NOISE GATE", juce::dontSendNotification);

    addAndMakeVisible (levelMeter);
    addAndMakeVisible (noiseGateKnob);
    styleKnob (noiseGateKnob);

    // ---- Center pitch editor ----
    addAndMakeVisible (pitchEditor);
    // Dragging the green playhead seeks the preview cursor (the
    // editor timer keeps the displayed line in sync after that).
    pitchEditor.onPlayheadSeek = [this] (double seconds)
    {
        audioProcessor.setPreviewPositionSeconds (seconds);
    };
    // Piano keys audition the loaded sound (sine fallback); the note
    // name shows in the note LCD while the key is held.
    pitchEditor.onAuditionStart = [this] (int midi)
    {
        audioProcessor.startAudition (midi);
        auditioningMidi = midi;
    };
    pitchEditor.onAuditionStop = [this]
    {
        audioProcessor.stopAudition();
        auditioningMidi = -1;
    };
    // Manual curve correction: the note name shows while dragging,
    // and the edited curve commits (re-segmented) on release.
    pitchEditor.onCurveEditNote = [this] (int midi)
    {
        curveEditMidi = midi;
    };
    pitchEditor.onCurveEditCommitted = [this] (std::shared_ptr<PitchAnalysisResult> edited)
    {
        curveEditMidi = -1;
        if (edited != nullptr)
            audioProcessor.commitPitchCurveEdit (*edited);
    };

    // ---- Right instrument panel ----
    addAndMakeVisible (instrumentLabel);
    styleSectionLabel (instrumentLabel);
    instrumentLabel.setText ("INSTRUMENT", juce::dontSendNotification);

    addAndMakeVisible (instrumentBox);
    styleComboBox (instrumentBox);
    instrumentBox.addItem ("NO SOUND LOADED", 1);
    instrumentBox.setSelectedId (1, juce::dontSendNotification);
    instrumentBox.setTooltip ("Loaded one-shot instrument sound");

    addAndMakeVisible (instrumentFolderButton);
    instrumentFolderButton.setTooltip ("Load a local instrument one-shot");
    instrumentFolderButton.onClick = [this] { startInstrumentImport(); };

    addAndMakeVisible (midiOutButton);

    addAndMakeVisible (midiChannelBox);
    styleComboBox (midiChannelBox);
    for (int ch = 1; ch <= 16; ++ch)
        midiChannelBox.addItem ("CHANNEL " + juce::String (ch), ch);

    addAndMakeVisible (dragMidiButton);
    dragMidiButton.setText ("MIDI");
    dragMidiButton.setTooltip ("MIDI actions: drag the melody to FL Studio or export a .mid file");
    dragMidiButton.onClick = [this] { openMidiPanel(); };

    addAndMakeVisible (playOriginalButton);
    playOriginalButton.setText ("PLAY ORIGINAL");
    playOriginalButton.setClickingTogglesState (true);
    // The button is attached to the "playOriginal" parameter, so a
    // click already flipped its toggle state when this runs.
    playOriginalButton.onClick = [this]
    {
        if (playOriginalButton.getToggleState())
        {
            if (audioProcessor.startPreview())
            {
                playButton.setToggleState (true, juce::dontSendNotification);
                recordButton.setToggleState (false, juce::dontSendNotification);
            }
            else
            {
                // No clip to play: keep the parameter in sync (off).
                audioProcessor.setPlayOriginalParameter (false);
                playButton.setToggleState (false, juce::dontSendNotification);
            }
        }
        else
        {
            audioProcessor.stopPreview();
            playButton.setToggleState (false, juce::dontSendNotification);
        }
    };

    addAndMakeVisible (playInstrumentButton);
    playInstrumentButton.setText ("PLAY INSTRUMENT");
    playInstrumentButton.setClickingTogglesState (true);
    playInstrumentButton.setTooltip ("Replay the melody with the loaded sound");
    playInstrumentButton.onClick = [this]
    {
        if (playInstrumentButton.getToggleState())
        {
            if (audioProcessor.startInstrumentPreview())
            {
                playButton.setToggleState (false, juce::dontSendNotification);
                playOriginalButton.setToggleState (false, juce::dontSendNotification);
            }
            else
            {
                audioProcessor.setPlayInstrumentParameter (false);
            }
        }
        else
        {
            audioProcessor.pauseInstrumentPreview();
        }
    };

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
    analyzeButton.onClick = [this]
    {
        if (audioProcessor.isPitchAnalyzing())
        {
            audioProcessor.cancelPitchAnalysis();
            return;
        }

        if (audioProcessor.startPitchAnalysis())
        {
            recordButton.setToggleState (false, juce::dontSendNotification);
            playButton.setToggleState (false, juce::dontSendNotification);
            audioProcessor.setPlayOriginalParameter (false);
        }
    };

    addAndMakeVisible (sendToChannelButton);
    styleTextButton (sendToChannelButton, Tokens::raisedPanel, Tokens::primaryText);
    sendToChannelButton.onClick = []
    {
        // TODO: PHASE_3_AUDIO_LOGIC
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
    midiChannelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "midiChannel", midiChannelBox);

    updateKeyLabel();
    // ~30 Hz: meter, playhead, status, analysis progress and clip polling.
    startTimer (33);

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
    // A held audition key must not keep sounding after the editor
    // is gone (no mouseUp will ever arrive for it).
    audioProcessor.stopAudition();
    // Force the panel closed even if a drag was in flight (the
    // editor is being destroyed; the drag callback's SafePointer to
    // this editor is null by then, so nothing else can clean up).
    midiDragInFlight = false;
    dismissMidiPanel();
    if (midiDragFile.existsAsFile())
        midiDragFile.deleteFile(); // best effort; ignore failure
    fileChooser.reset();
    setLookAndFeel (nullptr);
}

void MelotheonAudioProcessorEditor::startImport()
{
    if (fileChooser != nullptr)
        return; // a chooser is already open

    auto chooser = std::make_shared<juce::FileChooser> (
        "Import Audio", juce::File{}, AudioFileLoader::getFileFilter());
    fileChooser = chooser;

    juce::Component::SafePointer<MelotheonAudioProcessorEditor> safeEditor (this);
    chooser->launchAsync (
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [safeEditor, chooser] (const juce::FileChooser& completedChooser)
        {
            const juce::File file = completedChooser.getResult();

            if (safeEditor == nullptr)
                return; // editor was closed while choosing

            // Reset only the chooser that produced this callback. The
            // captured shared_ptr keeps it alive until this lambda ends.
            if (safeEditor->fileChooser == chooser)
                safeEditor->fileChooser.reset();

            if (file.existsAsFile())
                safeEditor->audioProcessor.startFileImport (file);
        });
}

void MelotheonAudioProcessorEditor::startInstrumentImport()
{
    if (fileChooser != nullptr)
        return;

    auto chooser = std::make_shared<juce::FileChooser> (
        "Load Instrument Sound", juce::File{}, AudioFileLoader::getFileFilter());
    fileChooser = chooser;

    juce::Component::SafePointer<MelotheonAudioProcessorEditor> safeEditor (this);
    chooser->launchAsync (
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [safeEditor, chooser] (const juce::FileChooser& completedChooser)
        {
            const juce::File file = completedChooser.getResult();
            if (safeEditor == nullptr)
                return;

            if (safeEditor->fileChooser == chooser)
                safeEditor->fileChooser.reset();

            if (file.existsAsFile())
                safeEditor->audioProcessor.startInstrumentFileImport (file);
        });
}

void MelotheonAudioProcessorEditor::startMidiExport()
{
    if (fileChooser != nullptr)
        return;

    if (! audioProcessor.hasMelodyNotes())
    {
        // Let the processor publish the same actionable LCD error used by
        // every other export entry point.
        audioProcessor.exportMidiToFile (juce::File{});
        return;
    }

    const juce::File suggested = juce::File::getSpecialLocation (
        juce::File::userDocumentsDirectory).getChildFile ("Melotheon Melody.mid");
    auto chooser = std::make_shared<juce::FileChooser> (
        "Export Melody as MIDI", suggested, "*.mid");
    fileChooser = chooser;

    juce::Component::SafePointer<MelotheonAudioProcessorEditor> safeEditor (this);
    chooser->launchAsync (
        juce::FileBrowserComponent::saveMode
            | juce::FileBrowserComponent::canSelectFiles
            | juce::FileBrowserComponent::warnAboutOverwriting,
        [safeEditor, chooser] (const juce::FileChooser& completedChooser)
        {
            juce::File file = completedChooser.getResult();
            if (safeEditor == nullptr)
                return;

            if (safeEditor->fileChooser == chooser)
                safeEditor->fileChooser.reset();

            if (file != juce::File{})
            {
                file = file.withFileExtension ("mid");
                safeEditor->audioProcessor.exportMidiToFile (file);
            }
        });
}

void MelotheonAudioProcessorEditor::dismissMidiPanel()
{
    // A drag in flight keeps the panel (and its drag source) alive
    // until the drag callback runs; finishMidiDrag() clears the flag.
    // The destructor forces this off before calling here.
    if (midiDragInFlight)
        return;

    if (midiCallOutBox != nullptr)
    {
        midiCallOutBox->dismiss();
        midiCallOutBox = nullptr;
    }
}

void MelotheonAudioProcessorEditor::finishMidiDrag()
{
    midiDragInFlight = false;
    if (midiDragFile.existsAsFile())
        midiDragFile.deleteFile(); // best effort; ignore failure
    midiDragFile = juce::File{};
}

void MelotheonAudioProcessorEditor::openMidiPanel()
{
    // Toggle: clicking the chip while the panel is open closes it.
    if (midiCallOutBox != nullptr)
    {
        dismissMidiPanel();
        return;
    }

    const auto buttonBounds = dragMidiButton.getBounds();
    const int rowHeight = buttonBounds.getHeight();
    auto panel = std::make_unique<MidiActionPanel> (
        buttonBounds.getWidth(), rowHeight, 13.0f * scaleFactor);

    juce::Component::SafePointer<MelotheonAudioProcessorEditor> safeEditor (this);
    panel->onExportMidi = [safeEditor]
    {
        if (safeEditor == nullptr)
            return;
        safeEditor->dismissMidiPanel();
        safeEditor->startMidiExport();
    };
    panel->onClickHint = [safeEditor]
    {
        if (safeEditor == nullptr)
            return;
        safeEditor->audioProcessor.setStatusHint ("PRESS AND DRAG THIS ROW INTO FL STUDIO");
    };
    panel->onPrepareDragFile = [safeEditor]() -> juce::File
    {
        if (safeEditor == nullptr)
            return {};
        juce::File file = safeEditor->audioProcessor.prepareMidiDragFile();
        // Never start a drag from a missing or empty file: the user
        // would see a drag begin and FL would receive nothing.
        if (file == juce::File{} || ! file.existsAsFile() || file.getSize() < 20)
            return {};
        // Editor-owned from here: it outlives the panel/row and is
        // deleted only in finishMidiDrag().
        safeEditor->midiDragFile = file;
        return file;
    };
    panel->onDragStarted = [safeEditor]
    {
        if (safeEditor == nullptr)
            return;
        safeEditor->midiDragInFlight = true;
    };
    panel->onDragFinished = [safeEditor]
    {
        if (safeEditor == nullptr)
            return;
        safeEditor->finishMidiDrag();
    };
    panel->onDismissRequested = [safeEditor]
    {
        if (safeEditor == nullptr)
            return;
        safeEditor->dismissMidiPanel();
    };

    auto& box = juce::CallOutBox::launchAsynchronously (
        std::move (panel), buttonBounds, this);
    midiCallOutBox = &box;
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
    using namespace melotheon;
    updateKeyLabel();

    // ---- Status LCD (green; orange for Error / Limit Reached). ----
    const auto status = audioProcessor.getStatus();
    const int statusId = static_cast<int> (status);
    const juce::String statusText = audioProcessor.getStatusText();
    // A transient hint (e.g. "DRAG THIS ROW INTO FL STUDIO") takes
    // over the LCD for a few seconds, then the normal status returns.
    const int hintVersion = audioProcessor.getStatusHintVersion();
    if (hintVersion != lastHintVersionSeen)
    {
        lastHintVersionSeen = hintVersion;
        activeHintText = audioProcessor.getStatusHint();
        hintExpiry = juce::Time::getCurrentTime() + juce::RelativeTime::seconds (3.0);
    }
    juce::String displayStatusText = statusText;
    if (activeHintText.isNotEmpty()
        && juce::Time::getCurrentTime() < hintExpiry)
        displayStatusText = activeHintText;
    if (displayStatusText != lastStatusText)
    {
        lastStatusText = displayStatusText;
        statusLabel.setText (displayStatusText, juce::dontSendNotification);
    }
    if (statusId != lastStatusSeen)
    {
        lastStatusSeen = statusId;
        const bool isWarning = (status == MelotheonAudioProcessor::Status::Error
                             || status == MelotheonAudioProcessor::Status::LimitReached
                             || status == MelotheonAudioProcessor::Status::NoPitch);
        statusLabel.setColour (juce::Label::textColourId,
                               isWarning ? Tokens::activeOrange : Tokens::statusGreen);
    }

    // ---- Input meter: smoothed peak dB (-60..0) -> 0..1. ----
    const float peakDb = audioProcessor.getInputPeakDb();
    levelMeter.setLevel (juce::jlimit (0.0f, 1.0f, (peakDb + 60.0f) / 60.0f));

    // ---- Waveform: install a new clip when the processor publishes one. ----
    const int version = audioProcessor.getClipVersion();
    if (version != lastClipVersionSeen)
    {
        lastClipVersionSeen = version;
        pitchEditor.setClip (audioProcessor.getActiveClip());
    }

    // ---- Pitch-analysis result and progress. ----
    const int analysisVersion = audioProcessor.getPitchAnalysisVersion();
    if (analysisVersion != lastAnalysisVersionSeen)
    {
        lastAnalysisVersionSeen = analysisVersion;
        displayedAnalysis = audioProcessor.getPitchAnalysis();
        pitchEditor.setPitchAnalysis (displayedAnalysis);
    }

    const int melodyNotesVersion = audioProcessor.getMelodyNotesVersion();
    if (melodyNotesVersion != lastMelodyNotesVersionSeen)
    {
        lastMelodyNotesVersionSeen = melodyNotesVersion;
        displayedNotes = audioProcessor.getMelodyNotes();
        pitchEditor.setNoteSequence (displayedNotes);
    }

    const int instrumentVersion = audioProcessor.getInstrumentVersion();
    if (instrumentVersion != lastInstrumentVersionSeen)
    {
        lastInstrumentVersionSeen = instrumentVersion;
        instrumentBox.changeItemText (1, audioProcessor.getInstrumentDisplayText());
        instrumentBox.setSelectedId (1, juce::dontSendNotification);
    }

    const bool analyzing = audioProcessor.isPitchAnalyzing();
    const float analysisProgress = audioProcessor.getPitchAnalysisProgress();
    pitchEditor.setAnalysisProgress (analyzing ? analysisProgress : -1.0f);

    const juce::String analyzeText = analyzing
        ? "CANCEL" : (displayedAnalysis != nullptr ? "RE-ANALYZE" : "ANALYZE");
    if (analyzeText != lastAnalyzeButtonText)
    {
        lastAnalyzeButtonText = analyzeText;
        analyzeButton.setButtonText (analyzeText);
    }

    juce::String noteText = "PITCH --";
    // Live interaction wins over the analysis frame: the note under a
    // curve drag, then the auditioned piano key.
    auto noteTextForMidi = [] (int midi) -> juce::String
    {
        PitchFrame frame;
        frame.midiPitch = static_cast<float> (midi);
        frame.frequencyHz = static_cast<float> (
            440.0 * std::pow (2.0, (static_cast<double> (midi) - 69.0) / 12.0));
        frame.confidence = 1.0f;
        return formatPitchFrame (frame);
    };
    if (curveEditMidi >= 0)
    {
        noteText = noteTextForMidi (curveEditMidi);
    }
    else if (auditioningMidi >= 0)
    {
        noteText = noteTextForMidi (auditioningMidi);
    }
    else if (analyzing)
    {
        noteText = "PITCH   "
                 + juce::String (juce::roundToInt (analysisProgress * 100.0f)) + "%";
    }
    else if (displayedAnalysis != nullptr && displayedAnalysis->hasPitch())
    {
        const double position = audioProcessor.getDisplayPreviewPositionSeconds();
        const PitchFrame* frame = displayedAnalysis->findNearestVoicedFrame (position);
        if (frame == nullptr && position <= 0.001)
        {
            const auto found = std::find_if (
                displayedAnalysis->frames.begin(), displayedAnalysis->frames.end(),
                [] (const PitchFrame& candidate) { return candidate.voiced; });
            if (found != displayedAnalysis->frames.end())
                frame = &*found;
        }
        if (frame != nullptr)
            noteText = formatPitchFrame (*frame);
    }
    if (noteText != lastNoteText)
    {
        lastNoteText = noteText;
        noteLabel.setText (noteText, juce::dontSendNotification);
    }

    // ---- Playhead follows preview playback. ----
    // The playhead cannot be grabbed while recording.
    pitchEditor.setSeekEnabled (! audioProcessor.isRecording());
    pitchEditor.setPlayheadPosition (audioProcessor.getDisplayPreviewPositionSeconds());

    // ---- Playback ended naturally: STOPPED + untoggle preview. ----
    if (audioProcessor.consumePlaybackEnded())
    {
        audioProcessor.setStatus (MelotheonAudioProcessor::Status::Stopped);
        audioProcessor.setPlayOriginalParameter (false);
        playButton.setToggleState (false, juce::dontSendNotification);
    }
    if (audioProcessor.consumeInstrumentPlaybackEnded())
    {
        audioProcessor.setStatus (MelotheonAudioProcessor::Status::InstrumentReady);
        audioProcessor.setPlayInstrumentParameter (false);
    }

    // ---- A take that hit the 60 s limit is finalized here (message
    // thread): the audio thread only flags the limit. ----
    if (status == MelotheonAudioProcessor::Status::LimitReached)
        audioProcessor.finalizeRecordingFromLimit();

    // ---- Keep the transport buttons in sync with the processor. ----
    const bool recording = audioProcessor.isRecording();
    if (recordButton.getToggleState() != recording)
        recordButton.setToggleState (recording, juce::dontSendNotification);

    const bool playing = audioProcessor.isPreviewPlaying();
    const bool instrumentPlaying = audioProcessor.isInstrumentPreviewPlaying();
    if (playButton.getToggleState() != playing)
        playButton.setToggleState (playing, juce::dontSendNotification);

    if (lastPlayOriginalParam != playing)
    {
        lastPlayOriginalParam = playing;
        audioProcessor.setPlayOriginalParameter (playing);
    }
    if (playInstrumentButton.getToggleState() != instrumentPlaying)
        playInstrumentButton.setToggleState (instrumentPlaying, juce::dontSendNotification);
    if (lastPlayInstrumentParam != instrumentPlaying)
    {
        lastPlayInstrumentParam = instrumentPlaying;
        audioProcessor.setPlayInstrumentParameter (instrumentPlaying);
    }
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
