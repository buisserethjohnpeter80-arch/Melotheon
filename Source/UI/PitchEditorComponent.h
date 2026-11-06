#pragma once

// PitchEditorComponent: the central piano-roll editor.
//
// Drawn entirely with juce::Graphics / juce::Path - no raster images.
//
// Before audio is loaded (Phase 1 behaviour): static placeholder
// data - a fixed monophonic melody, a subtle grey waveform behind
// the notes and one green playhead.
//
// After a clip is loaded (Phase 2): the placeholder notes and pitch
// curve are hidden (no pitch analysis has happened yet). The real
// waveform of the active clip is drawn from a cached min/max peak
// table, the green playhead follows preview playback, and a small
// centered message reads "AUDIO READY · CLICK ANALYZE".
//
// Layout: piano keyboard on the left (C3 at the bottom, C5 at the
// top), bar numbers 1-4 across the top, 4 bars of 4 beats each.

#include <JuceHeader.h>

#include <memory>
#include <utility>
#include <vector>

#include "../Audio/AudioClip.h"

class PitchEditorComponent : public juce::Component
{
public:
    PitchEditorComponent() = default;

    // Install a clip (message thread). nullptr restores the
    // placeholder view. The waveform peak cache is rebuilt here, so
    // paint() never scans audio.
    void setClip (std::shared_ptr<const AudioClip> clip);

    // Preview playhead position in seconds. Only used while a clip
    // is loaded; negative values restore the static placeholder
    // playhead. Repaints only when the position actually moved.
    void setPlayheadPosition (double seconds);

    void paint (juce::Graphics& g) override;
    void resized() override;

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

    static constexpr int lowestMidi = 48; // C3
    static constexpr int highestMidi = 72; // C5
    static constexpr int totalBeats = 16; // 4 bars x 4 beats
    static constexpr int beatsPerBar = 4;
    static constexpr float keyboardWidth = 70.0f;
    static constexpr float barNumberHeight = 18.0f;

    // Fixed-resolution peak cache for the real waveform: one
    // (min, max) pair per column, computed once in setClip() from
    // the clip's mono buffer with a bounded strided scan.
    static constexpr int waveformColumns = 1024;

    void rebuildWaveformCache();

    bool isBlackKey (int midiNote) const;
    juce::String noteName (int midiNote) const;

    std::shared_ptr<const AudioClip> activeClip;
    std::vector<std::pair<float, float>> waveformPeaks;
    double playheadSeconds = -1.0;
};
