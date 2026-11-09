// MELOTHÉON — Real Audio Validation (no microphone required).
//
// Offline validation of the production analysis pipeline on real recorded
// audio (sung melodies, guitar, violin, clarinet, saxophone, bassoon).
// The user has no microphone, so nothing here depends on live input.
//
// What this file does:
//   * Loads real recordings through the EXISTING CsvAnnotationDataset
//     interface (manifest + F0/notes CSVs) and runs the real production
//     code — MonophonicPitchAnalyzer::analyze(), NoteSegmenter::segment(),
//     MidiExporter::createMidiFile() — at Levels A/B/C, under both the
//     struct-default and the production-mapped settings.
//   * Prints one stable `BENCH REAL ...` line per case per settings set,
//     so the pre-Phase-2A binary and the Phase 2A binary can be compared
//     on IDENTICAL recordings by diffing their printed numbers.
//   * Runs the imported-audio workflow test at engine level
//     (load WAV -> analyze -> pitch curve -> MIDI notes -> export MIDI ->
//     edit + Undo/Redo).
//
// What it does NOT do:
//   * It does not modify any production code, threshold, or setting.
//   * It does not invent annotations: every reference frame/note comes
//     from a dataset file converted by the documented conversion script
//     (~/workspace/melotheon-datasets/convert_annotations.py). Entries
//     with no note-level ground truth (Bach10 stems) are reported as
//     UNSCORED at Level B, never scored against invented notes.
//   * It does not prove the FL Studio plugin interface works. The
//     workflow test is engine-level only (no host, no processBlock
//     audio, no message pump, no GUI, no SEND TO CHANNEL live MIDI-out).
//     That limitation is printed by the test and stated in the report.
//
// Dataset location (environment variables; absolute paths):
//   MELOTHEON_REAL_AUDIO_MANIFEST — path to one manifest.csv
//   MELOTHEON_REAL_AUDIO_DIR      — datasets root; manifests are read
//                                   from <DIR>/manifest.csv, else from
//                                   each immediate subdirectory's
//                                   manifest.csv (sorted by name).
// When neither is set (or no manifest is found) the test prints
// `BENCH REAL: SKIPPED` and returns {0, 0}: zero CHECKs, zero failures,
// so CI stays green without the datasets. Audio is never committed to
// the repo; datasets live under ~/workspace/melotheon-datasets/.
//
// CHECK discipline (measurement, not assertion): per-case CHECKs assert
// only properties true under BOTH segmenters (analysis non-null, one
// observation per reference frame, Level C MIDI integrity of whatever
// notes were produced). The vibrato-consolidation delta is the
// DIFFERENCE in the printed note counts / precision / recall / F1
// between the pre-2A and Phase 2A binaries — never a pass/fail on a
// single binary.

#pragma once

#include "PitchBenchmark.h"

namespace pitchbenchmark
{

// Runs the real-audio validation. Returns {checks, failures} in the
// same hand-rolled style as runPitchBenchmarkTests(); returns {0, 0}
// (and prints BENCH REAL: SKIPPED) when no dataset manifest is found.
BenchmarkTestOutcome runRealAudioValidationTests();

} // namespace pitchbenchmark
