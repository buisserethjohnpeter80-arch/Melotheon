#pragma once

// PitchEditorComponent: the central piano-roll editor.
//
// Drawn entirely with juce::Graphics / juce::Path - no raster images.
// Phase 1 shows STATIC placeholder data only: a fixed monophonic
// melody, a subtle grey waveform behind the notes and one green
// playhead. There is no pitch detection, no analysis and no MIDI
// generation behind it yet (see PHASE_2_AUDIO_LOGIC in the editor).
//
// Layout: piano keyboard on the left (C3 at the bottom, C5 at the
// top), bar numbers 1-4 across the top, 4 bars of 4 beats each.

#include <JuceHeader.h>

class PitchEditorComponent : public juce::Component
{
public:
    PitchEditorComponent() = default;

    void paint (juce::Graphics& g) override;

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

    bool isBlackKey (int midiNote) const;
    juce::String noteName (int midiNote) const;
};
