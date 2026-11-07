#pragma once

// Discrete monophonic notes derived from the continuous 10 ms pitch map.
// Times are kept in seconds so the sequence remains independent from the
// host tempo. MIDI export converts seconds to beats using the current host
// BPM, while the internal sample renderer uses the times directly.

#include <algorithm>
#include <vector>

struct MelodyNote
{
    double startSeconds = 0.0;
    double endSeconds = 0.0;
    int midiNote = 60;
    float sourceMidiPitch = 60.0f;
    float velocity = 0.75f;   // normalised 0..1
    float confidence = 0.0f;  // normalised 0..1

    double durationSeconds() const
    {
        return std::max (0.0, endSeconds - startSeconds);
    }
};

struct MelodyNoteSequence
{
    std::vector<MelodyNote> notes;
    double durationSeconds = 0.0;
    int lowestMidi = 60;
    int highestMidi = 60;

    bool empty() const { return notes.empty(); }
};

struct NoteSegmentationSettings
{
    // Notes shorter than this are treated as pitch-detector jitter.
    double minimumNoteSeconds = 0.06;
    // Short unvoiced gaps between matching pitches are bridged.
    double maximumGapSeconds = 0.04;
    // Hysteresis keeps ordinary vocal vibrato inside one MIDI note.
    float pitchHysteresisSemitones = 0.72f;
};
