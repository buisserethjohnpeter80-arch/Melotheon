#pragma once

#include <memory>

#include "MelodyNotes.h"
#include "PitchAnalysis.h"

class NoteSegmenter
{
public:
    // Converts the continuous monophonic pitch curve into stable,
    // non-overlapping MIDI-note events. This is deterministic and does no
    // allocation on the audio thread; callers run it after offline analysis.
    static std::shared_ptr<MelodyNoteSequence> segment (
        const PitchAnalysisResult& analysis,
        const NoteSegmentationSettings& settings = {});

    // Manual pitch-curve correction (message thread). Moves the voiced
    // frames near centreSeconds toward targetMidi with a raised-cosine
    // falloff over +/-windowSeconds, so the edit is smooth instead of a
    // single-frame spike. Unvoiced frames are left alone (there is no
    // curve to grab there) and frequencyHz is recomputed from the new
    // MIDI pitch. Callers re-run segment() afterwards so the note
    // blocks follow the corrected curve. Pure function: no audio-thread
    // state, no allocation beyond the frame vector itself.
    static void applyCurveEdit (PitchAnalysisResult& analysis,
                                double centreSeconds,
                                float targetMidi,
                                double windowSeconds);
};
