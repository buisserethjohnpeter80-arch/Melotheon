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
// min/max peak table. Phase 3A overlays the real monophonic pitch
// curve with confidence-dependent orange segments. It does not draw
// note blocks because note segmentation belongs to Phase 3B.
//
// Layout: piano keyboard on the left (C3 at the bottom, C5 at the
// top), bar numbers 1-4 across the top, 4 bars of 4 beats each.

#include <JuceHeader.h>

#include <memory>
#include <utility>
#include <vector>

#include "../Audio/AudioClip.h"
#include "../Analysis/PitchAnalysis.h"

class PitchEditorComponent : public juce::Component
{
public:
    PitchEditorComponent() = default;

    // Install a clip (message thread). nullptr restores the
    // placeholder view. The waveform peak cache is rebuilt here, so
    // paint() never scans audio.
    void setClip (std::shared_ptr<const AudioClip> clip);

    // Install/clear the immutable worker result. The pitch range is
    // fitted automatically while preserving the same piano-roll style.
    void setPitchAnalysis (std::shared_ptr<const PitchAnalysisResult> analysis);

    // 0..1 while the worker is active, negative when idle. This only
    // changes the centered helper text; progress also appears in the
    // header LCD.
    void setAnalysisProgress (float progress);

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

    static constexpr int defaultLowestMidi = 48; // C3
    static constexpr int defaultHighestMidi = 72; // C5
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
    std::pair<int, int> visibleMidiRange() const;

    std::shared_ptr<const AudioClip> activeClip;
    std::shared_ptr<const PitchAnalysisResult> pitchAnalysis;
    std::vector<std::pair<float, float>> waveformPeaks;
    double playheadSeconds = -1.0;
    float analysisProgress = -1.0f;
};
