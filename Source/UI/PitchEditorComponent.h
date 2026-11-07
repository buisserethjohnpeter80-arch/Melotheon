#pragma once

// PitchEditorComponent: the central piano-roll editor.
//
// Drawn entirely with juce::Graphics / juce::Path - no raster images.
//
// Before audio is loaded (Phase 1 behaviour): static placeholder
// data - a fixed monophonic melody, a subtle grey waveform behind
// the notes and one green playhead.
//
// After a clip is loaded: the real waveform is drawn from a cached
// min/max peak table. The real monophonic pitch curve and segmented
// MIDI-note blocks are overlaid. The keyboard has professional vertical
// navigation: +/- zoom, FIT, a scrollbar and mouse-wheel scrolling.
//
// Layout: piano keyboard on the left (C3 at the bottom, C5 at the
// top), bar numbers 1-4 across the top, 4 bars of 4 beats each.

#include <JuceHeader.h>

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "../Audio/AudioClip.h"
#include "../Analysis/MelodyEditor.h"
#include "../Analysis/MelodyNotes.h"
#include "../Analysis/PitchAnalysis.h"
#include "MelotheonLookAndFeel.h"

class PitchEditorComponent : public juce::Component,
                             private juce::ScrollBar::Listener
{
public:
    PitchEditorComponent();
    ~PitchEditorComponent() override;

    // Install a clip (message thread). nullptr restores the
    // placeholder view. The waveform peak cache is rebuilt here, so
    // paint() never scans audio.
    void setClip (std::shared_ptr<const AudioClip> clip);

    // Install/clear the immutable worker result. The pitch range is
    // fitted automatically while preserving the same piano-roll style.
    void setPitchAnalysis (std::shared_ptr<const PitchAnalysisResult> analysis);
    void setNoteSequence (std::shared_ptr<const MelodyNoteSequence> notes);

    // 0..1 while the worker is active, negative when idle. This only
    // changes the centered helper text; progress also appears in the
    // header LCD.
    void setAnalysisProgress (float progress);

    // Preview playhead position in seconds. Only used while a clip
    // is loaded; negative values restore the static placeholder
    // playhead. Repaints only when the position actually moved.
    void setPlayheadPosition (double seconds);

    // Called when the user drags the green playhead (message thread).
    // Seconds are clamped to the clip range. Empty by default: the
    // component then only moves its own displayed playhead.
    std::function<void (double)> onPlayheadSeek;

    // Piano-key audition (message thread). mouseDown on a keyboard
    // key starts the note, mouseUp / mouseExit stops it; sliding
    // across keys while held re-triggers (glissando). The editor
    // forwards these to the processor's audition voice and shows the
    // note name in the note LCD.
    std::function<void (int)> onAuditionStart; // MIDI note
    std::function<void()> onAuditionStop;

    // Manual pitch-curve correction (message thread). While the user
    // drags the orange curve, onCurveEditNote reports the quantized
    // semitone under the mouse (the editor shows it in the note LCD).
    // On release, onCurveEditCommitted hands the edited analysis to
    // the editor, which commits it to the processor (re-segmentation
    // + instrument re-render). The grey waveform is never editable.
    std::function<void (int)> onCurveEditNote; // quantized MIDI note
    std::function<void (std::shared_ptr<PitchAnalysisResult>)> onCurveEditCommitted;

    // Note-edit tools (MOVE / CUT / STRETCH strip in the bar-number
    // strip). MOVE is the default. The selected tool is highlighted
    // orange; the tool only changes how clicks inside a note block
    // behave — the playhead, keyboard audition and curve drag keep
    // their existing priority.
    enum class EditTool { Move, Cut, Stretch };
    EditTool getEditTool() const { return editTool; }
    void setEditTool (EditTool tool);

    // Manual note editing (message thread). While the user drags a
    // note block, onNoteEditNote reports the semitone under the mouse
    // (the editor shows it in the note LCD). On release,
    // onNoteEditCommitted hands the edited sequence to the editor,
    // which commits it to the processor (melody replace + instrument
    // re-render). The pitchShift travels with a pitch move so the
    // underlying pitch frames move with the note; it is inactive for
    // time-only moves and stretches. CUT commits immediately on click.
    std::function<void (int)> onNoteEditNote; // quantized MIDI note
    std::function<void (std::shared_ptr<MelodyNoteSequence>,
                        const MelodyEditor::PitchShift&)> onNoteEditCommitted;

    // False while recording: the playhead cannot be grabbed then.
    // The editor updates this from its timer.
    void setSeekEnabled (bool enabled) { seekEnabled = enabled; }

    void paint (juce::Graphics& g) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent& event) override;
    void mouseDrag (const juce::MouseEvent& event) override;
    void mouseUp (const juce::MouseEvent& event) override;
    void mouseExit (const juce::MouseEvent& event) override;
    void mouseMove (const juce::MouseEvent& event) override;
    void mouseWheelMove (const juce::MouseEvent& event,
                         const juce::MouseWheelDetails& wheel) override;

    float getVerticalZoom() const { return verticalZoom; }

private:
    struct Note
    {
        double startBeat;   // in beats, 0.0 = start of bar 1
        double lengthBeats;
        int midiNote;       // 48 = C3, 60 = C4, 72 = C5
    };

    // Static placeholder melody: monophonic, no overlapping notes.
    static const Note placeholderNotes[];
    static constexpr int numPlaceholderNotes = 12;

    static constexpr int defaultLowestMidi = 48; // C3
    static constexpr int defaultHighestMidi = 72; // C5
    static constexpr int absoluteLowestMidi = 24; // C1
    static constexpr int absoluteHighestMidi = 108; // C8
    static constexpr int totalBeats = 16; // 4 bars x 4 beats
    static constexpr int beatsPerBar = 4;
    static constexpr float keyboardWidth = 94.0f;
    static constexpr float barNumberHeight = 26.0f;
    static constexpr float scrollBarWidth = 13.0f;

    // Fixed-resolution peak cache for the real waveform: one
    // (min, max) pair per column, computed once in setClip() from
    // the clip's mono buffer with a bounded strided scan.
    static constexpr int waveformColumns = 1024;

    void rebuildWaveformCache();
    void fitToContent();
    void changeZoom (float multiplier);
    void updateScrollBar();
    void setScrollOffset (double offset);
    int fittedNoteCount() const;
    void scrollBarMoved (juce::ScrollBar* scrollBarThatHasMoved,
                         double newRangeStart) override;

    // Playhead drag geometry (same constants as paint(), so the
    // grabbed line and the drawn line always agree).
    float playheadXForSeconds (double seconds) const;
    double secondsForX (float x) const; // -1 when no clip is loaded
    bool isNearPlayhead (float x) const;

    // Curve / keyboard geometry (same constants as paint()).
    float rowHeightValue() const;
    float midiToYValue (float midiNote) const;
    // Inverse of the curve's y (centre of the row for a pitch).
    float midiAtYValue (float y) const;
    // The keyboard row under a y position (whole semitone row).
    int keyMidiAtY (float y) const;
    // True when (x, y) is within the grab tolerance of a voiced
    // curve point; returns that point's time and pitch.
    bool curvePointNear (float x, float y, double& secondsOut,
                         float& midiOut) const;
    // Apply the current curve drag at a mouse position (recomputed
    // from the drag-start snapshot every move, so edits never
    // accumulate). No-op when no curve drag is active.
    void applyCurveDragAt (float x, float y);

    // Note-block geometry (same constants as paint(), so the grabbed
    // block and the drawn block always agree).
    float beatWidthValue() const;
    float beatToXValue (double beat) const;
    juce::Rectangle<float> noteBlockRect (const MelodyNote& note) const;
    // Index of the note block under (x, y), or -1. In STRETCH mode
    // edgeOut is -1 (left edge) / +1 (right edge) / 0 (body); in
    // other modes it is always 0. Later notes are drawn over earlier
    // ones, so the topmost match wins.
    int noteAtPosition (float x, float y, int& edgeOut) const;
    // Clip duration for beat snapping (the sequence duration when
    // available, else the clip's).
    double sequenceDurationSeconds() const;
    // Apply the current note drag at a mouse position (recomputed
    // from the drag-start snapshot every move). No-op when no note
    // drag is active.
    void applyNoteDragAt (float x, float y);
    // Split the note at the click position (CUT tool): commits
    // immediately. No-op when the click is too close to an edge.
    void cutNoteAt (int noteIndex, float x);

    // Half-width of the smooth edit window around the grab point.
    static constexpr double curveEditWindowSeconds = 0.15;
    // Grab tolerance for the orange curve, in pixels.
    static constexpr float curveGrabTolerancePx = 8.0f;

    bool isBlackKey (int midiNote) const;
    juce::String noteName (int midiNote) const;
    std::pair<int, int> visibleMidiRange() const;

    std::shared_ptr<const AudioClip> activeClip;
    std::shared_ptr<const PitchAnalysisResult> pitchAnalysis;
    std::shared_ptr<const MelodyNoteSequence> noteSequence;
    std::vector<std::pair<float, float>> waveformPeaks;
    double playheadSeconds = -1.0;
    float analysisProgress = -1.0f;
    bool seekEnabled = true;
    bool draggingPlayhead = false;

    // Piano-key audition: the MIDI note currently held (-1 = none).
    // The held key is highlighted in paint().
    int auditionMidi = -1;

    // Manual curve drag: the analysis snapshot at grab time and the
    // working copy that follows the mouse. paint() draws the working
    // copy while a drag is active; the processor only sees it on
    // mouseUp (commit), so note blocks update once, on release.
    bool draggingCurve = false;
    std::shared_ptr<const PitchAnalysisResult> dragCurveOriginal;
    std::shared_ptr<PitchAnalysisResult> dragCurveEdited;

    // Manual note drag (MOVE / STRETCH): the sequence snapshot at
    // grab time and the working copy that follows the mouse. paint()
    // draws the working copy while a drag is active; the processor
    // only sees it on mouseUp (commit). CUT commits immediately and
    // never starts a drag.
    EditTool editTool = EditTool::Move;
    bool draggingNote = false;
    int dragNoteIndex = -1;
    int dragNoteEdge = 0; // -1 = left edge, 0 = body, +1 = right edge
    double dragNoteGrabOffsetSeconds = 0.0;
    std::shared_ptr<const MelodyNoteSequence> dragNoteOriginal;
    std::shared_ptr<MelodyNoteSequence> dragNoteEdited;

    juce::TextButton zoomOutButton { "-" };
    juce::TextButton fitButton { "FIT" };
    juce::TextButton zoomInButton { "+" };
    // Smaller text-button font for the MOVE/CUT/STRETCH strip: the
    // default 15 pt button font is too wide for three labels in the
    // bar-number strip. Derives from the app look-and-feel so only
    // the font changes. Declared before the tool buttons so it is
    // destroyed after them.
    class SmallToolButtonLookAndFeel : public MelotheonLookAndFeel
    {
    public:
        juce::Font getTextButtonFont (juce::TextButton&, int) override
        {
            return melotheon::melotheonFont (11.0f, true);
        }
    };
    SmallToolButtonLookAndFeel smallToolButtonLookAndFeel;
    juce::TextButton moveToolButton { "MOVE" };
    juce::TextButton cutToolButton { "CUT" };
    juce::TextButton stretchToolButton { "STRETCH" };
    juce::ScrollBar verticalScrollBar { true };
    float verticalZoom = 1.0f;
    int fittedLowestMidi = defaultLowestMidi;
    int fittedHighestMidi = defaultHighestMidi;
    int visibleLowestMidi = defaultLowestMidi;
    int visibleHighestMidi = defaultHighestMidi;
    bool updatingScrollBar = false;
};
