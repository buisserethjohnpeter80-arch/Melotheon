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
};
