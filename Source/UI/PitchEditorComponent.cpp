// PitchEditorComponent implementation. All drawing is proportional to
// the component bounds, so the editor stays aligned when resized.
// Phase 3 adds a real confidence-weighted pitch curve, segmented note
// blocks and a scrollable/zoomable vertical piano view.

#include "PitchEditorComponent.h"
#include "MelotheonLookAndFeel.h"
#include "../Analysis/NoteSegmenter.h"
#include <cmath>

const PitchEditorComponent::Note PitchEditorComponent::placeholderNotes[] = {
    { 0.0, 1.0, 48 }, // C3
    { 1.0, 1.0, 50 }, // D3
    { 2.0, 1.0, 52 }, // E3
    { 3.0, 1.0, 55 }, // G3
    { 4.0, 2.0, 57 }, // A3
    { 6.0, 1.0, 59 }, // B3
    { 7.0, 1.0, 60 }, // C4
    { 8.0, 2.0, 62 }, // D4
    { 10.0, 1.0, 64 }, // E4
    { 11.0, 1.0, 67 }, // G4
    { 12.0, 2.0, 69 }, // A4 (melody peak)
    { 14.0, 2.0, 57 }, // A3
};

PitchEditorComponent::PitchEditorComponent()
{
    using namespace melotheon;
    for (auto* button : { &zoomOutButton, &fitButton, &zoomInButton })
    {
        addAndMakeVisible (*button);
        button->setColour (juce::TextButton::buttonColourId, Tokens::raisedPanel);
        button->setColour (juce::TextButton::buttonOnColourId, Tokens::activeOrange);
        button->setColour (juce::TextButton::textColourOffId, Tokens::primaryText);
        button->setColour (juce::TextButton::textColourOnId, juce::Colours::black);
        button->setMouseClickGrabsKeyboardFocus (false);
    }
    zoomOutButton.setTooltip ("Reduce piano key height");
    fitButton.setTooltip ("Fit detected melody");
    zoomInButton.setTooltip ("Enlarge piano key height");
    zoomOutButton.onClick = [this] { changeZoom (1.0f / 1.25f); };
    zoomInButton.onClick = [this] { changeZoom (1.25f); };
    fitButton.onClick = [this] { fitToContent(); };

    // Note-edit tool strip (MOVE / CUT / STRETCH): same dark tokens
    // as the zoom buttons, selected tool highlighted orange. The
    // small font keeps three labels inside the bar-number strip.
    for (auto* button : { &moveToolButton, &cutToolButton, &stretchToolButton,
                          &undoButton, &redoButton })
    {
        addAndMakeVisible (*button);
        button->setLookAndFeel (&smallToolButtonLookAndFeel);
        button->setColour (juce::TextButton::buttonColourId, Tokens::raisedPanel);
        button->setColour (juce::TextButton::buttonOnColourId, Tokens::activeOrange);
        button->setColour (juce::TextButton::textColourOffId, Tokens::primaryText);
        button->setColour (juce::TextButton::textColourOnId, juce::Colours::black);
        button->setMouseClickGrabsKeyboardFocus (false);
    }
    moveToolButton.setTooltip ("Move notes: drag a block to any pitch or time");
    cutToolButton.setTooltip ("Cut notes: click a block to split it in two");
    stretchToolButton.setTooltip ("Stretch notes: drag a block edge to lengthen or shorten it");
    undoButton.setTooltip ("Undo the last note or curve edit");
    redoButton.setTooltip ("Redo the last undone edit");
    moveToolButton.onClick = [this] { setEditTool (EditTool::Move); };
    cutToolButton.onClick = [this] { setEditTool (EditTool::Cut); };
    stretchToolButton.onClick = [this] { setEditTool (EditTool::Stretch); };
    undoButton.onClick = [this] { if (onUndoRequested != nullptr) onUndoRequested(); };
    redoButton.onClick = [this] { if (onRedoRequested != nullptr) onRedoRequested(); };
    moveToolButton.setToggleState (true, juce::dontSendNotification);
    undoButton.setEnabled (false);
    redoButton.setEnabled (false);

    addAndMakeVisible (verticalScrollBar);
    verticalScrollBar.setAutoHide (false);
    verticalScrollBar.setColour (juce::ScrollBar::backgroundColourId,
                                 Tokens::pitchBackground.brighter (0.08f));
    verticalScrollBar.setColour (juce::ScrollBar::thumbColourId,
                                 Tokens::activeOrange.withAlpha (0.78f));
    verticalScrollBar.setColour (juce::ScrollBar::trackColourId,
                                 Tokens::border);
    verticalScrollBar.addListener (this);

    fitToContent();
}

PitchEditorComponent::~PitchEditorComponent()
{
    verticalScrollBar.removeListener (this);
}

bool PitchEditorComponent::isBlackKey (int midiNote) const
{
    const int pc = midiNote % 12;
    return pc == 1 || pc == 3 || pc == 6 || pc == 8 || pc == 10;
}

juce::String PitchEditorComponent::noteName (int midiNote) const
{
    static const char* names[] = { "C", "C#", "D", "D#", "E", "F",
                                   "F#", "G", "G#", "A", "A#", "B" };
    return juce::String (names[midiNote % 12])
         + juce::String ((midiNote / 12) - 1);
}

void PitchEditorComponent::setClip (std::shared_ptr<const AudioClip> clip)
{
    activeClip = std::move (clip);
    pitchAnalysis.reset();
    noteSequence.reset();
    analysisProgress = -1.0f;
    rebuildWaveformCache();
    playheadSeconds = (activeClip != nullptr) ? 0.0 : -1.0;
    fitToContent();
}

void PitchEditorComponent::setTempoGrid (const TempoGrid& grid)
{
    if (tempoGrid == grid)
        return;
    tempoGrid = grid;
    // The grid geometry (and the tool-strip placement) changed.
    resized();
    repaint();
}

int PitchEditorComponent::displayTotalBeats() const
{
    if (activeClip == nullptr || activeClip->durationSeconds <= 0.0)
        return placeholderTotalBeats;
    return tempoGrid.totalBeats (activeClip->durationSeconds);
}

double PitchEditorComponent::displayBeatsPerBar() const
{
    if (activeClip == nullptr || activeClip->durationSeconds <= 0.0)
        return placeholderBeatsPerBar;
    return tempoGrid.beatsPerBar();
}

void PitchEditorComponent::setPitchAnalysis (
    std::shared_ptr<const PitchAnalysisResult> analysis)
{
    pitchAnalysis = std::move (analysis);
    analysisProgress = -1.0f;
    fitToContent();
}

void PitchEditorComponent::setNoteSequence (
    std::shared_ptr<const MelodyNoteSequence> notes)
{
    noteSequence = std::move (notes);
    fitToContent();
}

void PitchEditorComponent::setAnalysisProgress (float progress)
{
    const float next = progress < 0.0f ? -1.0f : juce::jlimit (0.0f, 1.0f, progress);
    const int oldPercent = analysisProgress < 0.0f
        ? -1 : juce::roundToInt (analysisProgress * 100.0f);
    const int newPercent = next < 0.0f ? -1 : juce::roundToInt (next * 100.0f);
    if (oldPercent == newPercent)
        return;

    analysisProgress = next;
    repaint();
}

void PitchEditorComponent::setPlayheadPosition (double seconds)
{
    if (activeClip == nullptr)
        return;
    if (std::abs (seconds - playheadSeconds) < 0.005)
        return;

    playheadSeconds = seconds;
    repaint();
}

void PitchEditorComponent::resized()
{
    // The peak cache is resolution-independent (fixed 1024 columns),
    // so a resize needs no rebuild; the playhead position is
    // re-derived from seconds in paint().
    const int scrollWidth = static_cast<int> (scrollBarWidth);
    verticalScrollBar.setBounds (getWidth() - scrollWidth,
                                 static_cast<int> (barNumberHeight),
                                 scrollWidth,
                                 juce::jmax (0, getHeight() - static_cast<int> (barNumberHeight)));

    const int buttonHeight = 20;
    const int y = 3;
    const int right = getWidth() - scrollWidth - 4;
    zoomInButton.setBounds (right - 22, y, 22, buttonHeight);
    fitButton.setBounds (right - 62, y, 36, buttonHeight);
    zoomOutButton.setBounds (right - 88, y, 22, buttonHeight);

    // Tool strip (UNDO | REDO | MOVE | CUT | STRETCH) in the
    // bar-number strip, in a gap between bar numbers, so it never
    // covers one. The bar starts come from the display grid (host
    // tempo when a clip is loaded), so the strip moves when the tempo
    // changes; when no gap fits it right-aligns before the KEY ZOOM
    // label and nudges left past any number (see
    // toolStripLeftForBarStarts).
    const float stripWidth = 40.0f + 3.0f + 40.0f + 3.0f + 40.0f
                           + 3.0f + 32.0f + 3.0f + 52.0f;
    std::vector<float> barStarts;
    const int numBars = tempoGrid.totalBars (
        activeClip != nullptr ? activeClip->durationSeconds : 0.0);
    for (int bar = 0; bar < numBars; ++bar)
        barStarts.push_back (beatToXValue (tempoGrid.barStartBeat (bar)));
    if (activeClip == nullptr)
    {
        for (int bar = 0; bar < 4; ++bar)
            barStarts.push_back (beatToXValue (
                static_cast<double> (bar) * placeholderBeatsPerBar));
    }
    const float zoneLeft = keyboardWidth + 2.0f;
    const float zoneRight = static_cast<float> (getWidth())
                          - scrollBarWidth - 152.0f - 4.0f;
    const int toolLeft = toolStripLeftForBarStarts (
        barStarts, zoneLeft, zoneRight, stripWidth);
    stretchToolButton.setBounds (toolLeft + 164, y, 52, buttonHeight);
    cutToolButton.setBounds (toolLeft + 129, y, 32, buttonHeight);
    moveToolButton.setBounds (toolLeft + 86, y, 40, buttonHeight);
    redoButton.setBounds (toolLeft + 43, y, 40, buttonHeight);
    undoButton.setBounds (toolLeft, y, 40, buttonHeight);
    updateScrollBar();
}

void PitchEditorComponent::mouseWheelMove (
    const juce::MouseEvent& event,
    const juce::MouseWheelDetails& wheel)
{
    if (event.mods.isCtrlDown() || event.mods.isCommandDown())
    {
        changeZoom (wheel.deltaY >= 0.0f ? 1.25f : 1.0f / 1.25f);
        return;
    }

    const double currentOffset = static_cast<double> (
        absoluteHighestMidi - visibleHighestMidi);
    setScrollOffset (currentOffset - static_cast<double> (wheel.deltaY) * 5.0);
}

float PitchEditorComponent::playheadXForSeconds (double seconds) const
{
    if (activeClip == nullptr || activeClip->durationSeconds <= 0.0)
        return keyboardWidth;

    const float gridW = static_cast<float> (getWidth())
                      - keyboardWidth - scrollBarWidth;
    return keyboardWidth + static_cast<float> (
        juce::jlimit (0.0, activeClip->durationSeconds, seconds)
        / activeClip->durationSeconds) * gridW;
}

double PitchEditorComponent::secondsForX (float x) const
{
    if (activeClip == nullptr || activeClip->durationSeconds <= 0.0)
        return -1.0;

    const float gridW = static_cast<float> (getWidth())
                      - keyboardWidth - scrollBarWidth;
    if (gridW <= 0.0f)
        return -1.0;

    const double fraction = juce::jlimit (
        0.0, 1.0, (static_cast<double> (x) - keyboardWidth) / gridW);
    return fraction * activeClip->durationSeconds;
}

bool PitchEditorComponent::isNearPlayhead (float x) const
{
    if (activeClip == nullptr || ! seekEnabled)
        return false;

    // Small grab tolerance around the green line: normal clicks
    // elsewhere in the roll do nothing new.
    constexpr float grabTolerance = 6.0f;
    return std::abs (x - playheadXForSeconds (playheadSeconds)) <= grabTolerance;
}

float PitchEditorComponent::rowHeightValue() const
{
    const int numRows = visibleHighestMidi - visibleLowestMidi + 1;
    return (static_cast<float> (getHeight()) - barNumberHeight)
         / static_cast<float> (juce::jmax (1, numRows));
}

float PitchEditorComponent::midiToYValue (float midiNote) const
{
    return barNumberHeight
         + (static_cast<float> (visibleHighestMidi) - midiNote) * rowHeightValue();
}

float PitchEditorComponent::midiAtYValue (float y) const
{
    // Inverse of the curve's y (midiToY + half a row).
    return static_cast<float> (visibleHighestMidi)
         - (y - barNumberHeight - rowHeightValue() * 0.5f) / rowHeightValue();
}

int PitchEditorComponent::keyMidiAtY (float y) const
{
    const int midi = static_cast<int> (std::floor (
        static_cast<float> (visibleHighestMidi)
        - (y - barNumberHeight) / rowHeightValue()));
    return juce::jlimit (visibleLowestMidi, visibleHighestMidi, midi);
}

bool PitchEditorComponent::curvePointNear (float x, float y,
                                           double& secondsOut,
                                           float& midiOut) const
{
    if (pitchAnalysis == nullptr || ! pitchAnalysis->hasPitch()
        || activeClip == nullptr || activeClip->durationSeconds <= 0.0
        || pitchAnalysis->hopSeconds <= 0.0
        || pitchAnalysis->frames.empty())
        return false;

    const double seconds = secondsForX (x);
    if (seconds < 0.0)
        return false;

    const int centre = juce::jlimit (
        0, static_cast<int> (pitchAnalysis->frames.size()) - 1,
        static_cast<int> (std::llround (seconds / pitchAnalysis->hopSeconds)));

    // The curve between frames is a straight segment, so the nearest
    // of the centre frame and its neighbours is close enough for an
    // 8 px grab tolerance.
    float bestDistance = curveGrabTolerancePx;
    bool found = false;
    for (int offset = -2; offset <= 2; ++offset)
    {
        const int i = centre + offset;
        if (i < 0 || i >= static_cast<int> (pitchAnalysis->frames.size()))
            continue;
        const auto& frame = pitchAnalysis->frames[static_cast<size_t> (i)];
        if (! frame.voiced)
            continue;
        const float px = playheadXForSeconds (frame.timeSeconds);
        const float py = midiToYValue (frame.midiPitch) + rowHeightValue() * 0.5f;
        const float distance = std::hypot (x - px, y - py);
        if (distance <= bestDistance)
        {
            bestDistance = distance;
            secondsOut = frame.timeSeconds;
            midiOut = frame.midiPitch;
            found = true;
        }
    }
    return found;
}

void PitchEditorComponent::applyCurveDragAt (float x, float y)
{
    if (dragCurveEdited == nullptr || dragCurveOriginal == nullptr)
        return;

    const double seconds = secondsForX (x);
    if (seconds < 0.0)
        return;

    // Semitone resolution: the curve snaps to whole notes while
    // dragging, and the editor shows the note name in the note LCD.
    const int targetMidi = juce::jlimit (
        24, 108, juce::roundToInt (midiAtYValue (y)));

    // Recompute from the drag-start snapshot every move: the edit is
    // one application of the current target, never an accumulation.
    *dragCurveEdited = *dragCurveOriginal;
    NoteSegmenter::applyCurveEdit (
        *dragCurveEdited, seconds,
        static_cast<float> (targetMidi), curveEditWindowSeconds);

    if (onCurveEditNote != nullptr)
        onCurveEditNote (targetMidi);
    repaint();
}

void PitchEditorComponent::setEditTool (EditTool tool)
{
    if (editTool == tool)
        return;
    editTool = tool;
    moveToolButton.setToggleState (tool == EditTool::Move, juce::dontSendNotification);
    cutToolButton.setToggleState (tool == EditTool::Cut, juce::dontSendNotification);
    stretchToolButton.setToggleState (tool == EditTool::Stretch, juce::dontSendNotification);
    repaint();
}

void PitchEditorComponent::setUndoRedoEnabled (bool canUndo, bool canRedo)
{
    undoButton.setEnabled (canUndo);
    redoButton.setEnabled (canRedo);
}

PitchEditorComponent::GridCursor PitchEditorComponent::cursorForPosition (
    EditTool tool, bool overKeyboard, bool nearPlayhead, bool overNote,
    int noteEdge, bool nearCurve)
{
    // Priority: keyboard, playhead, note block, curve. The curve-edit
    // cursor is offered ONLY in MOVE mode: in CUT / STRETCH a click
    // on the curve does nothing, so the cursor must not promise it.
    if (overKeyboard)
        return GridCursor::PointingHand;
    if (nearPlayhead)
        return GridCursor::LeftRightResize;
    if (overNote)
    {
        if (tool == EditTool::Cut)
            return GridCursor::Crosshair;
        if (tool == EditTool::Stretch && noteEdge != 0)
            return GridCursor::LeftRightResize;
        return GridCursor::DraggingHand;
    }
    if (nearCurve && tool == EditTool::Move)
        return GridCursor::UpDownResize;
    return GridCursor::Normal;
}

juce::MouseCursor PitchEditorComponent::toMouseCursor (GridCursor cursor)
{
    switch (cursor)
    {
        case GridCursor::PointingHand:    return juce::MouseCursor::PointingHandCursor;
        case GridCursor::LeftRightResize: return juce::MouseCursor::LeftRightResizeCursor;
        case GridCursor::Crosshair:       return juce::MouseCursor::CrosshairCursor;
        case GridCursor::DraggingHand:    return juce::MouseCursor::DraggingHandCursor;
        case GridCursor::UpDownResize:    return juce::MouseCursor::UpDownResizeCursor;
        case GridCursor::Normal: default: return juce::MouseCursor::NormalCursor;
    }
}

float PitchEditorComponent::beatWidthValue() const
{
    const float gridW = static_cast<float> (getWidth())
                      - keyboardWidth - scrollBarWidth;
    // Seconds-based: one beat is beatSeconds() of the clip, so a beat
    // line and a note at the same time share the same x (notes, the
    // curve and the playhead are all drawn from seconds). The old
    // width (gridW / totalBeats) stretched beats to fill the width,
    // so notes sat off their grid lines whenever the clip was not a
    // whole number of beats.
    if (activeClip != nullptr && activeClip->durationSeconds > 0.0)
        return gridW * static_cast<float> (
            tempoGrid.beatSeconds() / activeClip->durationSeconds);
    return gridW / static_cast<float> (juce::jmax (1, displayTotalBeats()));
}

float PitchEditorComponent::beatToXValue (double beat) const
{
    return keyboardWidth + static_cast<float> (beat) * beatWidthValue();
}

double PitchEditorComponent::sequenceDurationSeconds() const
{
    if (noteSequence != nullptr && noteSequence->durationSeconds > 0.0)
        return noteSequence->durationSeconds;
    if (activeClip != nullptr)
        return activeClip->durationSeconds;
    return 0.0;
}

juce::Rectangle<float> PitchEditorComponent::noteBlockRect (const MelodyNote& note) const
{
    if (activeClip == nullptr || activeClip->durationSeconds <= 0.0)
        return {};

    const float gridX = keyboardWidth;
    const float gridW = static_cast<float> (getWidth()) - keyboardWidth - scrollBarWidth;
    const float gridH = static_cast<float> (getHeight()) - barNumberHeight;
    const auto [lowestMidi, highestMidi] = visibleMidiRange();
    if (note.midiNote < lowestMidi || note.midiNote > highestMidi)
        return {};

    const double start = juce::jlimit (
        0.0, activeClip->durationSeconds, note.startSeconds);
    const double end = juce::jlimit (
        start, activeClip->durationSeconds, note.endSeconds);
    if (end <= start)
        return {};

    const float x = gridX
        + static_cast<float> (start / activeClip->durationSeconds) * gridW;
    const float right = gridX
        + static_cast<float> (end / activeClip->durationSeconds) * gridW;
    const float y = midiToYValue (static_cast<float> (note.midiNote)) + 1.0f;
    const float rowH = rowHeightValue();
    juce::Rectangle<float> block (
        x + 1.0f, y, juce::jmax (2.0f, right - x - 2.0f),
        juce::jmax (2.0f, rowH - 2.0f));
    return block.getIntersection (
        juce::Rectangle<float> (gridX, barNumberHeight, gridW, gridH));
}

int PitchEditorComponent::noteAtPosition (float x, float y, int& edgeOut) const
{
    edgeOut = 0;
    const MelodyNoteSequence* source = dragNoteEdited != nullptr
        ? dragNoteEdited.get() : noteSequence.get();
    if (source == nullptr || source->empty() || activeClip == nullptr)
        return -1;

    // Topmost first: later notes are drawn over earlier ones.
    for (int i = static_cast<int> (source->notes.size()) - 1; i >= 0; --i)
    {
        const auto block = noteBlockRect (source->notes[static_cast<size_t> (i)]);
        if (block.isEmpty() || ! block.contains (x, y))
            continue;

        if (editTool == EditTool::Stretch)
        {
            constexpr float edgeZone = 6.0f;
            if (x - block.getX() <= edgeZone)
                edgeOut = -1;
            else if (block.getRight() - x <= edgeZone)
                edgeOut = 1;
        }
        return i;
    }
    return -1;
}

void PitchEditorComponent::applyNoteDragAt (float x, float y)
{
    if (dragNoteEdited == nullptr || dragNoteOriginal == nullptr || dragNoteIndex < 0)
        return;
    const double duration = sequenceDurationSeconds();
    const double clickSeconds = secondsForX (x);
    if (clickSeconds < 0.0)
        return;

    // Recompute from the drag-start snapshot every move: the edit is
    // one application of the current target, never an accumulation.
    *dragNoteEdited = *dragNoteOriginal;
    if (dragNoteEdge == 0)
    {
        // MOVE: vertical = pitch (semitone snap), horizontal = time
        // (snap to the host-tempo beat grid). The grab offset keeps
        // the note from jumping under the mouse. Duration stays the same.
        const double targetStart = MelodyEditor::snapSecondsToBeat (
            clickSeconds - dragNoteGrabOffsetSeconds,
            beatLengthSeconds(), duration);
        const int targetMidi = juce::jlimit (
            24, 108, juce::roundToInt (midiAtYValue (y)));
        MelodyEditor::moveNote (*dragNoteEdited, dragNoteIndex,
                                targetStart, targetMidi, duration);
        if (onNoteEditNote != nullptr)
            onNoteEditNote (dragNoteEdited->notes[
                static_cast<size_t> (dragNoteIndex)].midiNote);
    }
    else
    {
        // STRETCH: the grabbed edge follows the mouse (beat snap);
        // pitch, frames and recorded audio are untouched.
        const double edgeSeconds = MelodyEditor::snapSecondsToBeat (
            clickSeconds, beatLengthSeconds(), duration);
        MelodyEditor::stretchNoteEdge (*dragNoteEdited, dragNoteIndex,
                                       dragNoteEdge < 0, edgeSeconds, duration);
    }
    repaint();
}

void PitchEditorComponent::cutNoteAt (int noteIndex, float x)
{
    if (noteSequence == nullptr || noteIndex < 0
        || noteIndex >= static_cast<int> (noteSequence->notes.size()))
        return;
    const double clickSeconds = secondsForX (x);
    if (clickSeconds < 0.0)
        return;

    auto edited = std::make_shared<MelodyNoteSequence> (*noteSequence);
    // Snap to the beat grid when the click is already close to one;
    // an off-grid click keeps its exact time. A click near an edge
    // is clamped into the splittable range, so every CUT click on a
    // note splits it (a note shorter than two minimum pieces cannot
    // be split and is left alone).
    const double cutSeconds = MelodyEditor::snapSecondsToBeatIfClose (
        clickSeconds, beatLengthSeconds(), sequenceDurationSeconds());
    const auto& target = noteSequence->notes[static_cast<size_t> (noteIndex)];
    const double clampedCut = juce::jlimit (
        target.startSeconds + MelodyEditor::kMinimumNoteSeconds,
        target.endSeconds - MelodyEditor::kMinimumNoteSeconds, cutSeconds);
    if (! MelodyEditor::splitNoteAt (*edited, noteIndex, clampedCut))
        return; // note too short to split into two valid pieces

    if (onNoteEditCommitted != nullptr)
        onNoteEditCommitted (edited, MelodyEditor::PitchShift {});
}

void PitchEditorComponent::mouseDown (const juce::MouseEvent& event)
{
    draggingPlayhead = false;
    draggingCurve = false;
    dragCurveOriginal.reset();
    dragCurveEdited.reset();
    draggingNote = false;
    dragNoteIndex = -1;
    dragNoteEdge = 0;
    dragNoteOriginal.reset();
    dragNoteEdited.reset();

    const float x = event.position.x;
    const float y = event.position.y;

    // Piano keyboard (left of the grid): audition the key. Keys are
    // left of the grid and the playhead/curve are inside it, so the
    // two gestures can never overlap.
    if (x < keyboardWidth && y >= barNumberHeight)
    {
        auditionMidi = keyMidiAtY (y);
        if (onAuditionStart != nullptr)
            onAuditionStart (auditionMidi);
        repaint();
        return;
    }
    if (x < keyboardWidth || activeClip == nullptr)
        return; // bar-number strip / keyboard edge: nothing to grab

    // Grid click priority: playhead (within 6 px of the green line)
    // first, then note block, then curve. A click inside a note block
    // grabs the block, never the curve behind it.
    if (isNearPlayhead (x))
    {
        draggingPlayhead = true;
        return;
    }

    if (noteSequence != nullptr && ! noteSequence->empty())
    {
        int edge = 0;
        const int noteIndex = noteAtPosition (x, y, edge);
        if (noteIndex >= 0)
        {
            const auto& note = noteSequence->notes[static_cast<size_t> (noteIndex)];

            if (editTool == EditTool::Cut)
            {
                cutNoteAt (noteIndex, x);
                return;
            }

            // MOVE / STRETCH: start a drag from a snapshot so the
            // working copy is recomputed every move and the stored
            // melody only changes on release (commit). In STRETCH
            // mode an edge grab stretches; a body grab moves.
            const double clickSeconds = secondsForX (x);
            dragNoteOriginal = noteSequence;
            dragNoteEdited = std::make_shared<MelodyNoteSequence> (*noteSequence);
            dragNoteIndex = noteIndex;
            dragNoteEdge = (editTool == EditTool::Stretch) ? edge : 0;
            dragNoteGrabOffsetSeconds = clickSeconds >= 0.0
                ? juce::jlimit (0.0, note.endSeconds - note.startSeconds,
                                clickSeconds - note.startSeconds)
                : 0.0;
            draggingNote = true;
            if (onNoteEditNote != nullptr)
                onNoteEditNote (note.midiNote);
            repaint();
            return;
        }
    }

    // Curve (MOVE tool only): the note blocks above already claimed
    // their clicks, so the curve can never be grabbed through a block.
    if (editTool == EditTool::Move)
    {
        double curveSeconds = 0.0;
        float curveMidi = 0.0f;
        if (curvePointNear (x, y, curveSeconds, curveMidi))
        {
            dragCurveOriginal = pitchAnalysis;
            dragCurveEdited = std::make_shared<PitchAnalysisResult> (*pitchAnalysis);
            draggingCurve = true;
            applyCurveDragAt (x, y);
        }
    }
}

void PitchEditorComponent::mouseDrag (const juce::MouseEvent& event)
{
    if (draggingPlayhead)
    {
        const double seconds = secondsForX (event.position.x);
        if (seconds < 0.0)
            return;

        if (onPlayheadSeek != nullptr)
            onPlayheadSeek (seconds);
        setPlayheadPosition (seconds);
        return;
    }

    if (draggingNote)
    {
        applyNoteDragAt (event.position.x, event.position.y);
        return;
    }

    if (draggingCurve)
    {
        applyCurveDragAt (event.position.x, event.position.y);
        return;
    }

    // Audition: sliding across keys while held re-triggers the key
    // under the mouse (glissando). Outside the keyboard the last key
    // keeps sounding until release.
    if (auditionMidi >= 0 && event.position.x < keyboardWidth
        && event.position.y >= barNumberHeight)
    {
        const int midi = keyMidiAtY (event.position.y);
        if (midi != auditionMidi)
        {
            auditionMidi = midi;
            if (onAuditionStart != nullptr)
                onAuditionStart (midi);
            repaint();
        }
    }
}

void PitchEditorComponent::mouseUp (const juce::MouseEvent&)
{
    draggingPlayhead = false;

    if (draggingCurve)
    {
        draggingCurve = false;
        if (dragCurveEdited != nullptr && onCurveEditCommitted != nullptr)
            onCurveEditCommitted (dragCurveEdited);
        dragCurveOriginal.reset();
        dragCurveEdited.reset();
    }

    if (draggingNote)
    {
        draggingNote = false;
        if (dragNoteEdited != nullptr && dragNoteOriginal != nullptr
            && dragNoteIndex >= 0
            && dragNoteIndex < static_cast<int> (dragNoteOriginal->notes.size())
            && dragNoteIndex < static_cast<int> (dragNoteEdited->notes.size()))
        {
            const auto& original = dragNoteOriginal->notes[static_cast<size_t> (dragNoteIndex)];
            const auto& edited = dragNoteEdited->notes[static_cast<size_t> (dragNoteIndex)];
            // A plain click (no move) commits nothing, so the melody
            // is never marked edited by accident.
            const bool changed = edited.startSeconds != original.startSeconds
                              || edited.endSeconds != original.endSeconds
                              || edited.midiNote != original.midiNote;
            if (changed && onNoteEditCommitted != nullptr)
            {
                // A pitch move also shifts the underlying pitch frames
                // in that note's time range by the same delta, so the
                // curve and the notes stay in sync. Time-only moves
                // and stretches leave the frames alone.
                MelodyEditor::PitchShift shift;
                if (edited.midiNote != original.midiNote)
                {
                    shift.active = true;
                    shift.centreSeconds = 0.5 * (original.startSeconds + original.endSeconds);
                    shift.targetMidi = static_cast<float> (edited.midiNote);
                    shift.windowSeconds = original.endSeconds - original.startSeconds;
                }
                onNoteEditCommitted (dragNoteEdited, shift);
            }
        }
        dragNoteOriginal.reset();
        dragNoteEdited.reset();
        dragNoteIndex = -1;
        dragNoteEdge = 0;
        if (onNoteEditNote != nullptr)
            onNoteEditNote (-1);
    }

    if (auditionMidi >= 0)
    {
        auditionMidi = -1;
        if (onAuditionStop != nullptr)
            onAuditionStop();
        repaint();
    }
}

void PitchEditorComponent::mouseExit (const juce::MouseEvent&)
{
    // Safety net: a key held while the mouse leaves the component
    // must not keep sounding (mouseUp normally covers release).
    if (auditionMidi >= 0)
    {
        auditionMidi = -1;
        if (onAuditionStop != nullptr)
            onAuditionStop();
        repaint();
    }
}

void PitchEditorComponent::mouseMove (const juce::MouseEvent& event)
{
    const bool overKeyboard = event.position.x < keyboardWidth
        && event.position.y >= barNumberHeight;
    // The playhead can only be grabbed inside the grid (mouseDown
    // returns early left of it), so the resize cursor must not show
    // for the strip above the keyboard when the playhead is at 0.
    const bool nearPlayhead = event.position.x >= keyboardWidth
        && isNearPlayhead (event.position.x);

    bool overNote = false;
    int edge = 0;
    if (noteSequence != nullptr && ! noteSequence->empty())
        overNote = noteAtPosition (event.position.x, event.position.y, edge) >= 0;

    bool nearCurve = false;
    if (! overNote)
    {
        double curveSeconds = 0.0;
        float curveMidi = 0.0f;
        nearCurve = curvePointNear (event.position.x, event.position.y,
                                    curveSeconds, curveMidi);
    }

    setMouseCursor (toMouseCursor (cursorForPosition (
        editTool, overKeyboard, nearPlayhead, overNote, edge, nearCurve)));
}

void PitchEditorComponent::rebuildWaveformCache()
{
    waveformPeaks.clear();

    if (activeClip == nullptr)
        return;

    const auto& mono = activeClip->analysisMonoBuffer;
    const int numSamples = mono.getNumSamples();
    if (numSamples <= 0)
        return;

    const float* data = mono.getReadPointer (0);
    waveformPeaks.reserve (waveformColumns);

    // Bounded strided scan: at most 128 samples per column, so the
    // cost stays flat even for a 5-minute import.
    for (int col = 0; col < waveformColumns; ++col)
    {
        const int start = static_cast<int> (
            (static_cast<juce::int64> (col) * numSamples) / waveformColumns);
        const int end = static_cast<int> (
            (static_cast<juce::int64> (col + 1) * numSamples) / waveformColumns);
        const int range = juce::jmax (1, end - start);
        const int stride = juce::jmax (1, range / 128);

        float minValue = 0.0f, maxValue = 0.0f;
        for (int i = start; i < end; i += stride)
        {
            const float s = data[i];
            minValue = juce::jmin (minValue, s);
            maxValue = juce::jmax (maxValue, s);
        }
        waveformPeaks.emplace_back (minValue, maxValue);
    }
}

int PitchEditorComponent::fittedNoteCount() const
{
    return fittedHighestMidi - fittedLowestMidi + 1;
}

void PitchEditorComponent::fitToContent()
{
    int low = defaultLowestMidi;
    int high = defaultHighestMidi;

    if (noteSequence != nullptr && ! noteSequence->empty())
    {
        low = noteSequence->lowestMidi - 3;
        high = noteSequence->highestMidi + 3;
    }
    else if (pitchAnalysis != nullptr && pitchAnalysis->hasPitch())
    {
        low = static_cast<int> (std::floor (pitchAnalysis->minMidi)) - 3;
        high = static_cast<int> (std::ceil (pitchAnalysis->maxMidi)) + 3;
    }

    low = juce::jlimit (absoluteLowestMidi, absoluteHighestMidi, low);
    high = juce::jlimit (absoluteLowestMidi, absoluteHighestMidi, high);
    if (high < low)
        std::swap (low, high);

    // FIT never compresses the keyboard to more than two octaves unless the
    // melody genuinely needs the extra range. A narrow melody gets at least
    // 18 visible semitones, which makes the piano much easier to read.
    constexpr int minimumFitNotes = 18;
    const int currentCount = high - low + 1;
    if (currentCount < minimumFitNotes)
    {
        const int missing = minimumFitNotes - currentCount;
        low -= missing / 2;
        high += missing - missing / 2;
        if (low < absoluteLowestMidi)
        {
            high += absoluteLowestMidi - low;
            low = absoluteLowestMidi;
        }
        if (high > absoluteHighestMidi)
        {
            low -= high - absoluteHighestMidi;
            high = absoluteHighestMidi;
        }
    }

    fittedLowestMidi = juce::jlimit (absoluteLowestMidi, absoluteHighestMidi, low);
    fittedHighestMidi = juce::jlimit (fittedLowestMidi, absoluteHighestMidi, high);
    visibleLowestMidi = fittedLowestMidi;
    visibleHighestMidi = fittedHighestMidi;
    verticalZoom = 1.0f;
    updateScrollBar();
    repaint();
}

void PitchEditorComponent::changeZoom (float multiplier)
{
    const float nextZoom = juce::jlimit (1.0f, 4.0f, verticalZoom * multiplier);
    if (std::abs (nextZoom - verticalZoom) < 0.001f)
        return;

    const float centre = 0.5f * static_cast<float> (
        visibleLowestMidi + visibleHighestMidi);
    verticalZoom = nextZoom;
    const int visibleCount = juce::jlimit (
        7, absoluteHighestMidi - absoluteLowestMidi + 1,
        static_cast<int> (std::lround (
            static_cast<float> (fittedNoteCount()) / verticalZoom)));
    int low = static_cast<int> (std::lround (centre - 0.5f * (visibleCount - 1)));
    low = juce::jlimit (absoluteLowestMidi,
                        absoluteHighestMidi - visibleCount + 1, low);
    visibleLowestMidi = low;
    visibleHighestMidi = low + visibleCount - 1;
    updateScrollBar();
    repaint();
}

void PitchEditorComponent::setScrollOffset (double requestedOffset)
{
    const int visibleCount = visibleHighestMidi - visibleLowestMidi + 1;
    const int totalCount = absoluteHighestMidi - absoluteLowestMidi + 1;
    const double offset = juce::jlimit (
        0.0, static_cast<double> (juce::jmax (0, totalCount - visibleCount)),
        requestedOffset);
    visibleHighestMidi = absoluteHighestMidi
                       - static_cast<int> (std::lround (offset));
    visibleLowestMidi = visibleHighestMidi - visibleCount + 1;
    updateScrollBar();
    repaint();
}

void PitchEditorComponent::scrollBarMoved (
    juce::ScrollBar* scrollBarThatHasMoved,
    double newRangeStart)
{
    if (scrollBarThatHasMoved == &verticalScrollBar && ! updatingScrollBar)
        setScrollOffset (newRangeStart);
}

void PitchEditorComponent::updateScrollBar()
{
    const int visibleCount = visibleHighestMidi - visibleLowestMidi + 1;
    const int totalCount = absoluteHighestMidi - absoluteLowestMidi + 1;
    const double offset = static_cast<double> (
        absoluteHighestMidi - visibleHighestMidi);

    updatingScrollBar = true;
    verticalScrollBar.setRangeLimits (0.0, static_cast<double> (totalCount));
    verticalScrollBar.setCurrentRange (
        offset, static_cast<double> (visibleCount), juce::dontSendNotification);
    updatingScrollBar = false;
}

std::pair<int, int> PitchEditorComponent::visibleMidiRange() const
{
    return { visibleLowestMidi, visibleHighestMidi };
}

void PitchEditorComponent::paint (juce::Graphics& g)
{
    using namespace melotheon;
    const float width = static_cast<float> (getWidth());
    const float height = static_cast<float> (getHeight());

    g.setColour (Tokens::pitchBackground);
    g.fillRect (0.0f, 0.0f, width, height);

    const float gridX = keyboardWidth;
    const float gridY = barNumberHeight;
    const float gridW = width - keyboardWidth - scrollBarWidth;
    const float gridH = height - barNumberHeight;
    const auto [lowestMidi, highestMidi] = visibleMidiRange();
    const int numRows = highestMidi - lowestMidi + 1;
    const float rowH = gridH / static_cast<float> (numRows);
    // Grid from the host tempo: beats are 60 / BPM seconds, bars
    // follow the time signature (placeholder: fixed 16 beats). The
    // beat width is seconds-based (see beatWidthValue), so beat lines
    // and note blocks share the same x for the same time.
    const int numBeats = displayTotalBeats();
    const double beatsPerBarValue = displayBeatsPerBar();
    const float beatW = beatWidthValue();

    g.setFont (melotheonFont (10.0f, true));
    g.setColour (Tokens::secondaryText);
    g.drawText ("KEY ZOOM",
                juce::Rectangle<float> (width - scrollBarWidth - 152.0f, 0.0f,
                                        58.0f, barNumberHeight),
                juce::Justification::centredRight, true);

    auto midiToY = [&] (float midiNote) -> float
    {
        // Higher notes are higher on screen.
        return gridY + (static_cast<float> (highestMidi) - midiNote) * rowH;
    };
    auto beatToX = [&] (double beat) -> float
    {
        return gridX + static_cast<float> (beat) * beatW;
    };

    // ---- Bar numbers across the top (one per bar in the clip). ----
    // When bars are dense, draw only every Nth number so each stays
    // legible in a fixed-width rect: the old beat-wide rect clipped
    // numbers to nothing at fast tempos / long clips. The last bar's
    // number is skipped when it falls past the right edge (a partial
    // final beat has no line there either).
    const int numBars = activeClip != nullptr
        ? tempoGrid.totalBars (activeClip->durationSeconds) : 4;
    const float barSpacingPx = beatW * static_cast<float> (beatsPerBarValue);
    const int barNumberStride = juce::jmax (1, static_cast<int> (std::ceil (
        28.0f / juce::jmax (1.0f, barSpacingPx))));
    g.setFont (melotheonFont (12.0f, true));
    g.setColour (Tokens::secondaryText);
    for (int bar = 0; bar < numBars; bar += barNumberStride)
    {
        const float barX = beatToX (static_cast<double> (bar) * beatsPerBarValue);
        if (barX > gridX + gridW)
            break;
        g.drawText (juce::String (bar + 1),
                    juce::Rectangle<float> (
                        barX + 4.0f, 0.0f, 26.0f, barNumberHeight),
                    juce::Justification::centredLeft, true);
    }

    // ---- Horizontal semitone lines; stronger lines on C (octaves). ----
    for (int midi = lowestMidi; midi <= highestMidi; ++midi)
    {
        const float y = midiToY (static_cast<float> (midi));
        const bool isOctave = (midi % 12 == 0);
        g.setColour (isOctave ? Tokens::border.withAlpha (0.8f) : Tokens::gridLine);
        g.drawLine (gridX, y, gridX + gridW, y, isOctave ? 1.0f : 0.5f);
    }
    g.setColour (Tokens::border.withAlpha (0.8f));
    g.drawLine (gridX, height, width, height, 1.0f);

    // ---- Vertical beat lines; stronger lines on bar starts. ----
    // Beat lines past the right edge (a partial final beat) are
    // skipped: the beat width is seconds-based, so the last beat
    // line can fall beyond the clip end.
    for (int beat = 0; beat <= numBeats; ++beat)
    {
        const float x = beatToX (beat);
        if (x > gridX + gridW)
            break;
        const double remainder = std::fmod (
            static_cast<double> (beat), beatsPerBarValue);
        const bool isBar = remainder < 1e-9
                        || beatsPerBarValue - remainder < 1e-9;
        g.setColour (isBar ? Tokens::border.withAlpha (0.8f) : Tokens::gridLine);
            g.drawLine (x, gridY, x, gridY + gridH, isBar ? 1.0f : 0.5f);
    }
    // Bar lines that fall between beats (odd meters such as 5/8,
    // where a bar is 2.5 quarter-note beats) are drawn separately;
    // bar starts on whole beats are already drawn above.
    for (int bar = 1; bar < numBars; ++bar)
    {
        const double barBeat = static_cast<double> (bar) * beatsPerBarValue;
        if (std::abs (barBeat - std::round (barBeat)) < 1e-9)
            continue;
        const float x = beatToX (barBeat);
        if (x > gridX + gridW)
            break;
        g.setColour (Tokens::border.withAlpha (0.8f));
        g.drawLine (x, gridY, x, gridY + gridH, 1.0f);
    }

    if (activeClip != nullptr)
    {
        // ---- Real waveform from the cached peaks. ----
        // One vertical min/max line per pixel column, same grey as
        // the reference design, spanning the complete clip.
        if (! waveformPeaks.empty())
        {
            const float centreY = gridY + gridH * 0.5f;
            const float maxAmp = gridH * 0.30f;
            const int pixelColumns = juce::jmax (1, static_cast<int> (gridW));
            g.setColour (Tokens::waveformGrey.withAlpha (0.5f));
            for (int px = 0; px < pixelColumns; ++px)
            {
                const int col = juce::jmin (waveformColumns - 1,
                    static_cast<int> ((static_cast<float> (px) / gridW)
                                      * static_cast<float> (waveformColumns)));
                const auto [minValue, maxValue] = waveformPeaks[static_cast<size_t> (col)];
                const float x = gridX + static_cast<float> (px);
                g.drawLine (x, centreY - maxValue * maxAmp,
                            x, centreY - minValue * maxAmp, 1.0f);
            }
        }

        // ---- Segmented melody notes. ----
        // These blocks are the stable notes used by both MIDI export and
        // the sample instrument. The original confidence controls opacity,
        // so uncertain detector edges remain visually honest. While the
        // user drags a note, paint the working copy so the block
        // follows the mouse; the stored melody only changes on release
        // (commit).
        const MelodyNoteSequence* notesSource = dragNoteEdited != nullptr
            ? dragNoteEdited.get() : noteSequence.get();
        if (notesSource != nullptr && ! notesSource->empty()
            && activeClip->durationSeconds > 0.0)
        {
            for (const auto& note : notesSource->notes)
            {
                if (note.midiNote < lowestMidi || note.midiNote > highestMidi)
                    continue;

                const double start = juce::jlimit (
                    0.0, activeClip->durationSeconds, note.startSeconds);
                const double end = juce::jlimit (
                    start, activeClip->durationSeconds, note.endSeconds);
                if (end <= start)
                    continue;

                const float x = gridX
                    + static_cast<float> (start / activeClip->durationSeconds) * gridW;
                const float right = gridX
                    + static_cast<float> (end / activeClip->durationSeconds) * gridW;
                const float y = midiToY (static_cast<float> (note.midiNote)) + 1.0f;
                juce::Rectangle<float> block (
                    x + 1.0f, y, juce::jmax (2.0f, right - x - 2.0f),
                    juce::jmax (2.0f, rowH - 2.0f));
                block = block.getIntersection (
                    juce::Rectangle<float> (gridX, gridY, gridW, gridH));
                if (block.isEmpty())
                    continue;

                const float confidence = juce::jlimit (0.0f, 1.0f, note.confidence);
                g.setColour (Tokens::activeOrange.withAlpha (0.48f + confidence * 0.38f));
                g.fillRoundedRectangle (block, juce::jmin (3.0f, block.getHeight() * 0.2f));
                g.setColour (Tokens::brightOrange.withAlpha (0.72f + confidence * 0.28f));
                g.drawRoundedRectangle (block,
                                        juce::jmin (3.0f, block.getHeight() * 0.2f), 1.0f);

                if (block.getWidth() >= 34.0f && block.getHeight() >= 12.0f)
                {
                    g.setFont (melotheonFont (juce::jlimit (8.0f, 11.0f,
                                                           block.getHeight() * 0.62f), true));
                    g.setColour (juce::Colours::black.withAlpha (0.76f));
                    g.drawText (noteName (note.midiNote), block.reduced (4.0f, 0.0f),
                                juce::Justification::centredLeft, true);
                }
            }
        }

        // ---- Real confidence-weighted pitch curve (SOURCE PITCH). ----
        // Each adjacent voiced frame becomes one source-pitch segment.
        // The alpha follows detector confidence, so uncertain edges stay
        // visible without looking as authoritative as stable pitch.
        // While the user drags the curve, paint the working copy so
        // the line follows the mouse; the stored analysis only changes
        // on release (commit).
        const PitchAnalysisResult* curveSource = dragCurveEdited != nullptr
            ? dragCurveEdited.get() : pitchAnalysis.get();
        if (curveSource != nullptr && curveSource->hasPitch()
            && activeClip->durationSeconds > 0.0)
        {
            const auto& frames = curveSource->frames;
            for (size_t i = 1; i < frames.size(); ++i)
            {
                const auto& previous = frames[i - 1];
                const auto& current = frames[i];
                if (! previous.voiced || ! current.voiced)
                    continue;
                if (current.timeSeconds - previous.timeSeconds
                    > curveSource->hopSeconds * 1.75)
                    continue;

                const float x1 = gridX + static_cast<float> (
                    juce::jlimit (0.0, activeClip->durationSeconds, previous.timeSeconds)
                    / activeClip->durationSeconds) * gridW;
                const float x2 = gridX + static_cast<float> (
                    juce::jlimit (0.0, activeClip->durationSeconds, current.timeSeconds)
                    / activeClip->durationSeconds) * gridW;
                const float y1 = midiToY (previous.midiPitch) + rowH * 0.5f;
                const float y2 = midiToY (current.midiPitch) + rowH * 0.5f;
                const float confidence = juce::jlimit (
                    0.0f, 1.0f, 0.5f * (previous.confidence + current.confidence));

                g.setColour (sourcePitchColour.withAlpha (0.35f + confidence * 0.65f));
                g.drawLine (x1, y1, x2, y2, 1.3f + confidence * 0.9f);
            }
        }
        else
        {
            g.setFont (melotheonFont (13.0f, true));
            g.setColour (pitchAnalysis != nullptr
                             ? Tokens::activeOrange : Tokens::secondaryText);
            juce::String message;
            if (analysisProgress >= 0.0f)
                message = "ANALYZING "
                        + juce::String (juce::roundToInt (analysisProgress * 100.0f)) + "%";
            else if (pitchAnalysis != nullptr)
                message = "NO STABLE PITCH FOUND";
            else
                message = juce::String::fromUTF8 ("AUDIO READY \xC2\xB7 CLICK ANALYZE");

            g.drawText (message,
                        juce::Rectangle<float> (gridX, gridY, gridW, gridH),
                        juce::Justification::centred, true);
        }
    }
    else
    {
    // ---- Dense grey waveform behind the notes. ----
    // Deterministic filled shape (layered sines + a slow envelope
    // varying 0.4..1.0), mirrored around the grid centre so it fills
    // much of the editor height like the reference image.
    // Placeholder only.
    {
        const float centreY = gridY + gridH * 0.5f;
        const float maxAmp = gridH * 0.30f;
        const int steps = 420;
        auto waveOffset = [&] (float t) -> float
        {
            const float env = 0.4f + 0.6f * (0.5f + 0.5f * std::sin (t * 6.28318f * 1.5f + 0.9f));
            return maxAmp * env
                * (0.55f * std::sin (t * 6.28318f * 9.0f)
                   + 0.30f * std::sin (t * 6.28318f * 23.0f + 1.7f)
                   + 0.15f * std::sin (t * 6.28318f * 41.0f + 0.4f));
        };
        juce::Path top;
        for (int i = 0; i <= steps; ++i)
        {
            const float t = static_cast<float> (i) / static_cast<float> (steps);
            const float x = gridX + t * gridW;
            const float y = centreY - waveOffset (t);
            if (i == 0)
                top.startNewSubPath (x, y);
            else
                top.lineTo (x, y);
        }
        // Filled mirrored shape: top edge left-to-right, bottom edge
        // right-to-left.
        juce::Path fill = top;
        for (int i = steps; i >= 0; --i)
        {
            const float t = static_cast<float> (i) / static_cast<float> (steps);
            fill.lineTo (gridX + t * gridW, centreY + waveOffset (t));
        }
        fill.closeSubPath();
        g.setColour (Tokens::waveformGrey.withAlpha (0.35f));
        g.fillPath (fill);
        // 1 px signal line over the fill.
        g.setColour (Tokens::waveformGrey.withAlpha (0.5f));
        g.strokePath (top, juce::PathStrokeType (1.0f));
    }

    // ---- Orange monophonic note blocks. ----
    for (int i = 0; i < numPlaceholderNotes; ++i)
    {
        const Note& note = placeholderNotes[i];
        const float x = beatToX (note.startBeat) + 1.0f;
        const float w = static_cast<float> (note.lengthBeats) * beatW - 2.0f;
        const float y = midiToY (static_cast<float> (note.midiNote)) + 1.0f;
        const float h = rowH - 2.0f;
        juce::Rectangle<float> block (x, y, w, h);

        g.setColour (Tokens::activeOrange);
        g.fillRoundedRectangle (block, 2.0f);
        g.setColour (Tokens::brightOrange);
        g.drawRoundedRectangle (block, 2.0f, 1.0f);
    }

    // ---- ONE continuous stepped pitch curve over all notes. ----
    // The line climbs from note to note (portamento-style ramps at
    // the boundaries) and falls back down, like the reference image:
    // within each note it eases in from the previous note's pitch and
    // eases out toward the next note's pitch, with only a small
    // vibrato in the middle. Every sample is clamped to its own
    // note block, so the curve never leaves the blocks.
    {
        juce::Path curve;
        bool firstPoint = true;
        for (int i = 0; i < numPlaceholderNotes; ++i)
        {
            const Note& note = placeholderNotes[i];
            const float x = beatToX (note.startBeat) + 1.0f;
            const float w = static_cast<float> (note.lengthBeats) * beatW - 2.0f;
            const float y = midiToY (static_cast<float> (note.midiNote)) + 1.0f;
            const float h = rowH - 2.0f;
            const float centreY = y + h * 0.5f;
            const float prevCentreY = (i == 0)
                ? centreY
                : midiToY (static_cast<float> (placeholderNotes[i - 1].midiNote)) + rowH * 0.5f;
            const float nextCentreY = (i == numPlaceholderNotes - 1)
                ? centreY
                : midiToY (static_cast<float> (placeholderNotes[i + 1].midiNote)) + rowH * 0.5f;
            const float startY = juce::jlimit (y + 1.0f, y + h - 1.0f, prevCentreY);
            const float endY = juce::jlimit (y + 1.0f, y + h - 1.0f, nextCentreY);
            const int steps = 48;
            for (int s = 0; s <= steps; ++s)
            {
                const float t = static_cast<float> (s) / static_cast<float> (steps);
                float py = centreY;
                if (t < 0.15f)
                    py = startY + (centreY - startY) * (t / 0.15f);
                else if (t > 0.85f)
                    py = centreY + (endY - centreY) * ((t - 0.85f) / 0.15f);
                py += std::sin (t * 6.28318f * 2.0f) * h * 0.15f;
                py = juce::jlimit (y + 1.0f, y + h - 1.0f, py);
                const float px = x + t * w;
                if (firstPoint)
                {
                    curve.startNewSubPath (px, py);
                    firstPoint = false;
                }
                else
                {
                    curve.lineTo (px, py);
                }
            }
        }
        g.setColour (sourcePitchColour);
        g.strokePath (curve, juce::PathStrokeType (1.5f));
    }
    } // else: placeholder notes / curve (hidden once a clip is loaded)

    // ---- Piano keyboard on the left. ----
    {
        // White-key background first.
        g.setColour (juce::Colour (0xffe8e8e8));
        g.fillRect (0.0f, gridY, keyboardWidth, gridH);
        // White keys: one rectangle per white note. The held audition
        // key lights up orange.
        for (int midi = lowestMidi; midi <= highestMidi; ++midi)
        {
            if (isBlackKey (midi))
                continue;
            const float y = midiToY (static_cast<float> (midi));
            g.setColour (midi == auditionMidi ? Tokens::activeOrange
                                             : juce::Colour (0xfff2f2f2));
            g.fillRect (0.0f, y, keyboardWidth, rowH);
            g.setColour (juce::Colour (0xff9a9a9a));
            g.drawLine (0.0f, y + rowH, keyboardWidth, y + rowH, 0.5f);
        }
        // Black keys drawn over the white keys.
        for (int midi = lowestMidi; midi <= highestMidi; ++midi)
        {
            if (! isBlackKey (midi))
                continue;
            const float y = midiToY (static_cast<float> (midi));
            g.setColour (midi == auditionMidi ? Tokens::brightOrange
                                             : juce::Colour (0xff1a1a1a));
            g.fillRect (0.0f, y, keyboardWidth * 0.62f, rowH);
        }
        // Label every visible C key. The range remains C3-C5 until a
        // real result needs a lower or higher register.
        g.setFont (melotheonFont (11.0f, true));
        g.setColour (juce::Colour (0xff333333));
        const int firstC = ((lowestMidi + 11) / 12) * 12;
        for (int midi = firstC; midi <= highestMidi; midi += 12)
        {
            const float y = midiToY (static_cast<float> (midi));
            g.drawText (noteName (midi),
                        juce::Rectangle<float> (keyboardWidth - 24.0f, y, 22.0f, rowH),
                        juce::Justification::centredLeft, true);
        }
        // Right edge of the keyboard.
        g.setColour (Tokens::border);
        g.drawLine (keyboardWidth, gridY, keyboardWidth, height, 1.0f);
    }

    // ---- Green vertical playhead with a marker. ----
    // With a clip it follows preview playback; otherwise it sits at
    // the static placeholder position (bar 2, beat 5.5). The x comes
    // from playheadXForSeconds() so the drawn line and the drag
    // geometry always agree.
    {
        const float playheadX = (activeClip != nullptr)
            ? playheadXForSeconds (playheadSeconds)
            : beatToX (5.5);
        g.setColour (Tokens::statusGreen);
        g.drawLine (playheadX, gridY, playheadX, height, 1.5f);
        juce::Path marker;
        marker.startNewSubPath (playheadX - 6.0f, 0.0f);
        marker.lineTo (playheadX + 6.0f, 0.0f);
        marker.lineTo (playheadX, 9.0f);
        marker.closeSubPath();
        g.fillPath (marker);
    }

    // ---- Legend: SOURCE PITCH (detected curve) vs EDITED MIDI ----
    // (note blocks). Top-right of the grid, below the bar strip, so
    // a time-moved note sitting away from the curve reads as two
    // different things, not a drawing error.
    {
        g.setFont (melotheonFont (10.0f, true));
        const auto& font = g.getCurrentFont();
        const float sourceW = juce::GlyphArrangement::getStringWidth (font, sourcePitchLegendText);
        const float editedW = juce::GlyphArrangement::getStringWidth (font, editedMidiLegendText);
        const float swatchW = 10.0f;
        const float swatchH = 6.0f;
        const float gap = 4.0f;
        const float between = 10.0f;
        const float totalW = swatchW + gap + sourceW + between
                           + swatchW + gap + editedW;
        float x = width - scrollBarWidth - 6.0f - totalW;
        const float y = gridY + 5.0f;
        const float swatchY = y + (13.0f - swatchH) * 0.5f;
        g.setColour (sourcePitchColour);
        g.fillRoundedRectangle (juce::Rectangle<float> (x, swatchY, swatchW, swatchH), 1.0f);
        x += swatchW + gap;
        g.drawText (sourcePitchLegendText,
                    juce::Rectangle<float> (x, y, sourceW, 13.0f),
                    juce::Justification::centredLeft, true);
        x += sourceW + between;
        g.setColour (editedMidiColour);
        g.fillRoundedRectangle (juce::Rectangle<float> (x, swatchY, swatchW, swatchH), 1.0f);
        x += swatchW + gap;
        g.drawText (editedMidiLegendText,
                    juce::Rectangle<float> (x, y, editedW, 13.0f),
                    juce::Justification::centredLeft, true);
    }

    // ---- Outer border. ----
    g.setColour (Tokens::border);
    g.drawRect (0.0f, 0.0f, width, height, 1.0f);
}
