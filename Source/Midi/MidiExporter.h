#pragma once

#include <JuceHeader.h>

#include "../Analysis/MelodyNotes.h"

struct MidiExportSettings
{
    double bpm = 120.0;
    int channel = 1;
    int ticksPerQuarterNote = 960;
};

class MidiExporter
{
public:
    // Build a standard type-1 MIDI file. Note times in seconds are mapped to
    // ticks using BPM so the exported melody retains the sung timing.
    static juce::MidiFile createMidiFile (
        const MelodyNoteSequence& sequence,
        const MidiExportSettings& settings = {});

    // Atomic-at-file-level export used by the UI. Returns false and a short
    // LCD-safe error when there are no notes or the target cannot be written.
    static bool writeToFile (const MelodyNoteSequence& sequence,
                             const juce::File& target,
                             const MidiExportSettings& settings,
                             juce::String& errorMessage);
};
