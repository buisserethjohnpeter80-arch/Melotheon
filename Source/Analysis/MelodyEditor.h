// Melody editing operations shared by the piano-roll editor and the
// unit tests: beat snapping, note move / split / stretch. Pure
// functions over MelodyNoteSequence — no GUI, no audio thread, no
// JUCE — so the transforms are tested without a plugin window.
//
// Beat geometry: the piano roll draws exactly 16 display beats across
// the clip (see PitchEditorComponent::totalBeats), so one beat is
// durationSeconds / 16. Moves snap to that grid, consistently with
// what is drawn. The snap itself only needs the clip duration; the
// BPM only decides which beat the grid represents, and the grid is
// drawn from the duration, so no host BPM is needed here. (MIDI
// export, which maps time to musical beats, uses the host BPM with a
// 120 BPM fallback — see MidiExporter.)

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>

#include "MelodyNotes.h"

namespace MelodyEditor
{
    // Display beats shown across the clip (must match the piano roll).
    inline constexpr double kDisplayBeats = 16.0;

    // Smallest note a manual edit may produce (seconds). Matches the
    // stretch minimum used by the editor.
    inline constexpr double kMinimumNoteSeconds = 0.03;

    // Pitch-frame shift applied together with a note edit: when a note
    // moves in pitch, the underlying pitch frames in that note's time
    // range move by the same delta (centre = original note centre,
    // target = new pitch, window = original note duration), so the
    // orange curve and the note blocks stay in sync. Time-only moves
    // leave it inactive — the recorded audio stays where it was.
    struct PitchShift
    {
        bool active = false;
        double centreSeconds = 0.0;
        float targetMidi = 60.0f;
        double windowSeconds = 0.15;
    };

    inline double secondsToDisplayBeat (double seconds, double durationSeconds)
    {
        if (durationSeconds <= 0.0)
            return 0.0;
        return kDisplayBeats * seconds / durationSeconds;
    }

    inline double displayBeatToSeconds (double beat, double durationSeconds)
    {
        if (durationSeconds <= 0.0)
            return 0.0;
        return beat * durationSeconds / kDisplayBeats;
    }

    // Snap a time to the nearest display beat; unchanged when no
    // duration is known.
    inline double snapSecondsToDisplayBeat (double seconds, double durationSeconds)
    {
        if (durationSeconds <= 0.0)
            return seconds;
        const double beat = std::round (secondsToDisplayBeat (seconds, durationSeconds));
        return std::clamp (displayBeatToSeconds (beat, durationSeconds),
                           0.0, durationSeconds);
    }

    // Snap to the nearest display beat only when the time is already
    // within toleranceBeats of it (used by CUT, so an off-grid click
    // keeps its exact time instead of jumping to a beat).
    inline double snapSecondsToDisplayBeatIfClose (double seconds,
                                                   double durationSeconds,
                                                   double toleranceBeats = 0.25)
    {
        if (durationSeconds <= 0.0)
            return seconds;
        const double beat = secondsToDisplayBeat (seconds, durationSeconds);
        const double nearest = std::round (beat);
        if (std::abs (beat - nearest) <= toleranceBeats)
            return std::clamp (displayBeatToSeconds (nearest, durationSeconds),
                               0.0, durationSeconds);
        return seconds;
    }

    // Recompute the derived lowest/highest MIDI from the notes (the
    // sequence duration is the clip duration and is left untouched).
    inline void refreshSequenceMetadata (MelodyNoteSequence& sequence)
    {
        if (sequence.empty())
        {
            sequence.lowestMidi = 60;
            sequence.highestMidi = 60;
            return;
        }
        int lowest = sequence.notes.front().midiNote;
        int highest = lowest;
        for (const auto& note : sequence.notes)
        {
            lowest = std::min (lowest, note.midiNote);
            highest = std::max (highest, note.midiNote);
        }
        sequence.lowestMidi = lowest;
        sequence.highestMidi = highest;
    }

    // Move note `index` to [newStartSeconds, newStartSeconds + duration)
    // at newMidiNote. The move is clamped so the note never overlaps a
    // neighbour and never leaves [0, durationSeconds]: when the gap
    // between neighbours is too small for the note, the time stays
    // where it was (the pitch still applies). Returns false for a bad
    // index or an empty sequence.
    inline bool moveNote (MelodyNoteSequence& sequence, int index,
                          double newStartSeconds, int newMidiNote,
                          double durationSeconds)
    {
        if (index < 0 || index >= static_cast<int> (sequence.notes.size()))
            return false;

        auto& note = sequence.notes[static_cast<size_t> (index)];
        const double noteDuration = note.endSeconds - note.startSeconds;
        const double duration = durationSeconds > 0.0 ? durationSeconds : note.endSeconds;

        double low = 0.0;
        double high = duration - noteDuration;
        if (index > 0)
            low = std::max (low, sequence.notes[static_cast<size_t> (index - 1)].endSeconds);
        if (index + 1 < static_cast<int> (sequence.notes.size()))
            high = std::min (high, sequence.notes[static_cast<size_t> (index + 1)].startSeconds
                                      - noteDuration);

        double start = newStartSeconds;
        if (high >= low)
            start = std::clamp (start, low, high);
        else
            start = note.startSeconds; // no room to move in time

        note.startSeconds = start;
        note.endSeconds = start + noteDuration;
        note.midiNote = std::clamp (newMidiNote, 0, 127);
        note.sourceMidiPitch = static_cast<float> (note.midiNote);
        refreshSequenceMetadata (sequence);
        return true;
    }

    // Split note `index` into two notes at cutSeconds (same pitch,
    // velocity and confidence). Both pieces must be at least
    // minimumSeconds long; otherwise the split is refused. Returns
    // false for a bad index or a cut too close to an edge.
    inline bool splitNoteAt (MelodyNoteSequence& sequence, int index,
                             double cutSeconds,
                             double minimumSeconds = kMinimumNoteSeconds)
    {
        if (index < 0 || index >= static_cast<int> (sequence.notes.size()))
            return false;

        const auto original = sequence.notes[static_cast<size_t> (index)];
        if (cutSeconds - original.startSeconds < minimumSeconds
            || original.endSeconds - cutSeconds < minimumSeconds)
            return false;

        auto first = original;
        auto second = original;
        first.endSeconds = cutSeconds;
        second.startSeconds = cutSeconds;
        sequence.notes[static_cast<size_t> (index)] = first;
        sequence.notes.insert (sequence.notes.begin() + index + 1, second);
        refreshSequenceMetadata (sequence);
        return true;
    }

    // Lengthen/shorten note `index` by moving its left or right edge
    // to edgeSeconds (duration only — pitch and frames untouched).
    // The edge never passes a neighbour, never leaves the clip, and
    // the note never shrinks below minimumSeconds. Returns false for
    // a bad index or a refused stretch.
    inline bool stretchNoteEdge (MelodyNoteSequence& sequence, int index,
                                 bool leftEdge, double edgeSeconds,
                                 double durationSeconds,
                                 double minimumSeconds = kMinimumNoteSeconds)
    {
        if (index < 0 || index >= static_cast<int> (sequence.notes.size()))
            return false;

        auto& note = sequence.notes[static_cast<size_t> (index)];
        const double duration = durationSeconds > 0.0 ? durationSeconds : note.endSeconds;

        if (leftEdge)
        {
            double low = 0.0;
            if (index > 0)
                low = std::max (low, sequence.notes[static_cast<size_t> (index - 1)].endSeconds);
            const double high = note.endSeconds - minimumSeconds;
            if (high < low)
                return false;
            note.startSeconds = std::clamp (edgeSeconds, low, high);
        }
        else
        {
            const double low = note.startSeconds + minimumSeconds;
            double high = duration;
            if (index + 1 < static_cast<int> (sequence.notes.size()))
                high = std::min (high, sequence.notes[static_cast<size_t> (index + 1)].startSeconds);
            if (high < low)
                return false;
            note.endSeconds = std::clamp (edgeSeconds, low, high);
        }
        refreshSequenceMetadata (sequence);
        return true;
    }
}
