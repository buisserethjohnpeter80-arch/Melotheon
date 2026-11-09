// MELOTHEON Phase 1 — Professional Pitch Detection Benchmark tests.
//
// Measurement tests. They run the real production analysis code
// (analyze / segment / createMidiFile) on deterministic synthetic signals
// and print the measured numbers as BENCH lines, so the CI log carries
// the baseline. CHECKs assert only behaviour the production code
// guarantees (or the audit's own recommended bounds); the printed numbers
// are the measurement. A failing CHECK here is information to report —
// production code is NEVER changed to make a benchmark test pass.
//
// Correctness tests and performance benchmarks are separate: no timing
// value gates a CHECK (timing must not fail on CPU noise).
//
// Uses the same hand-rolled CHECK / CHECK_NEAR style as
// Tests/MelotheonTests.cpp, with its own counters. main() folds the
// returned outcome into the suite totals.

#include <JuceHeader.h>

#include "Analysis/MonophonicPitchAnalyzer.h"
#include "Analysis/NoteSegmenter.h"
#include "PitchBenchmark.h"
#include "RealAudioValidation.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace pitchbenchmark
{
namespace
{

int checks = 0;
int failures = 0;

void check (bool condition, const char* expression, int line)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cout << "FAIL (line " << line << "): " << expression << "\n";
    }
}

bool nearDouble (double a, double b, double epsilon)
{
    return std::abs (a - b) <= epsilon;
}

#define CHECK(cond) check ((cond), #cond, __LINE__)
#define CHECK_NEAR(a, b, eps) check (nearDouble ((a), (b), (eps)), #a " ~= " #b, __LINE__)

std::string fmt (double value, int precision = 2)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision (precision) << value;
    return stream.str();
}

void benchPrint (const std::string& line)
{
    std::cout << line << "\n";
}

juce::File getBenchmarkTestDir()
{
    juce::File dir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                         .getChildFile ("MelotheonBenchmarkTests");
    dir.createDirectory();
    return dir;
}

// A pure sine case with a fully-voiced reference at one frequency.
AudioTestCase makeSineCase (const std::string& name,
                            double sampleRate,
                            double durationSeconds,
                            double frequencyHz,
                            float amplitude = 0.5f)
{
    AudioTestCase testCase;
    testCase.name = name;
    testCase.sampleRate = sampleRate;
    testCase.audio = makeSineTone (sampleRate, durationSeconds, frequencyHz, amplitude);
    testCase.sourceDescription = "synthetic pure sine (deterministic)";
    testCase.referenceFrames = buildReferenceFrames (
        durationSeconds, kDefaultHopSeconds,
        [=] (double) { return frequencyHz; },
        [] (double) { return true; });
    ReferenceNote note;
    note.startSeconds = 0.0;
    note.endSeconds = durationSeconds;
    note.frequencyHz = frequencyHz;
    note.midiNote = static_cast<int> (std::lround (frequencyToMidiDouble (frequencyHz)));
    testCase.referenceNotes.push_back (note);
    return testCase;
}

// One-line Level A summary: "BENCH A <name>: RPA=..% RCA=..% oct=..% ...".
void printLevelA (const std::string& name, const CentsErrorStats& stats,
                  const VoicingConfusion& voicing)
{
    benchPrint ("BENCH A " + name
        + ": RPA=" + fmt (stats.rawPitchAccuracy * 100.0, 1) + "%"
        + " RCA=" + fmt (stats.rawChromaAccuracy * 100.0, 1) + "%"
        + " octErr=" + fmt (stats.octaveErrorRate * 100.0, 1) + "%"
        + " (up1=" + std::to_string (stats.octaveBreakdown.up1)
        + " down1=" + std::to_string (stats.octaveBreakdown.down1)
        + " up2=" + std::to_string (stats.octaveBreakdown.up2)
        + " down2=" + std::to_string (stats.octaveBreakdown.down2)
        + " up3=" + std::to_string (stats.octaveBreakdown.up3)
        + " down3=" + std::to_string (stats.octaveBreakdown.down3) + ")"
        + " meanAbs=" + (stats.hasConditionalStats ? fmt (stats.meanAbsCents, 1) + "c" : "n/a")
        + " medianAbs=" + (stats.hasConditionalStats ? fmt (stats.medianAbsCents, 1) + "c" : "n/a")
        + " voicedRef=" + std::to_string (stats.voicedReferenceFrames)
        + " detected=" + std::to_string (stats.detectedVoicedFrames)
        + " missed=" + std::to_string (stats.missedVoicedFrames)
        + " voicing P=" + (voicing.precisionDefined ? fmt (voicing.precision, 3) : "undef")
        + " R=" + (voicing.recallDefined ? fmt (voicing.recall, 3) : "undef")
        + " F1=" + (voicing.f1Defined ? fmt (voicing.f1, 3) : "undef"));
}

void printLevelB (const std::string& name, const NoteMatchResult& match)
{
    benchPrint ("BENCH B " + name
        + ": matched=" + std::to_string (match.matched)
        + " missed=" + std::to_string (match.missed)
        + " false=" + std::to_string (match.falsePositives)
        + " P=" + (match.precisionDefined ? fmt (match.precision, 3) : "undef")
        + " R=" + (match.recallDefined ? fmt (match.recall, 3) : "undef")
        + " F1=" + (match.f1Defined ? fmt (match.f1, 3) : "undef")
        + " onsetMeanAbs=" + (std::isnan (match.meanAbsOnsetErrorMs) ? "n/a" : fmt (match.meanAbsOnsetErrorMs, 1) + "ms")
        + " onsetMedAbs=" + (std::isnan (match.medianAbsOnsetErrorMs) ? "n/a" : fmt (match.medianAbsOnsetErrorMs, 1) + "ms")
        + " offsetMeanAbs=" + (std::isnan (match.meanAbsOffsetErrorMs) ? "n/a" : fmt (match.meanAbsOffsetErrorMs, 1) + "ms")
        + " offsetMedAbs=" + (std::isnan (match.medianAbsOffsetErrorMs) ? "n/a" : fmt (match.medianAbsOffsetErrorMs, 1) + "ms")
        + " durMeanAbs=" + (std::isnan (match.meanAbsDurationErrorMs) ? "n/a" : fmt (match.meanAbsDurationErrorMs, 1) + "ms"));
}

// ---- Part 6: octave-classifier unit test (benchmark-only logic) ----

void testBenchmarkOctaveClassifierUnit()
{
    const auto checkClass = [] (double cents, bool isError, int octaves, bool up)
    {
        const OctaveClassification result = classifyOctaveError (cents);
        CHECK (result.isOctaveError == isError);
        CHECK (result.octaves == octaves);
        CHECK (result.up == up);
    };

    checkClass (0.0, false, 0, false);
    checkClass (50.0, false, 0, false);
    checkClass (-50.0, false, 0, false);
    checkClass (700.0, false, 0, false);
    checkClass (1100.0, false, 0, false);
    checkClass (1300.0, false, 0, false);
    checkClass (1150.0, true, 1, true);   // inclusive lower boundary
    checkClass (-1150.0, true, 1, false);
    checkClass (1199.0, true, 1, true);
    checkClass (1200.0, true, 1, true);
    checkClass (-1200.0, true, 1, false);
    checkClass (1250.0, true, 1, true);   // inclusive upper boundary
    checkClass (2350.0, true, 2, true);
    checkClass (2400.0, true, 2, true);
    checkClass (-2400.0, true, 2, false);
    checkClass (3600.0, true, 3, true);
    checkClass (-3600.0, true, 3, false);
    checkClass (-3650.0, true, 3, false);
    benchPrint ("BENCH unit octave-classifier: 18 cases checked");
}

// ---- Level A: 440 Hz sine baseline ----

void testBenchmarkSine440()
{
    const auto testCase = makeSineCase ("sine-440", 48000.0, 1.0, 440.0, 0.5f);
    const LevelAResult result = runLevelA (testCase);
    CHECK (result.analysis != nullptr);
    if (result.analysis == nullptr)
        return;
    printLevelA ("sine440-48k", result.pitchStats, result.voicing);

    // The existing suite already proves median 440 +/-3 Hz at 44.1/48/96 kHz;
    // +/-3 Hz at 440 Hz is ~12 cents, so RPA at +/-50 cents must be ~100%.
    CHECK (result.pitchStats.rawPitchAccuracy >= 0.95);
    CHECK (result.pitchStats.rawChromaAccuracy >= 0.95);
    CHECK (result.pitchStats.octaveBreakdown.total == 0);
    CHECK (result.pitchStats.hasConditionalStats);
    if (result.pitchStats.hasConditionalStats)
        CHECK (result.pitchStats.medianAbsCents <= 25.0);
    CHECK (result.analysis->medianConfidence > 0.75f);
    CHECK_NEAR (result.analysis->medianFrequencyHz, 440.0f, 3.0f);
}

// ---- F1: harmonic-rich octave stress ----

void testBenchmarkHarmonicOctaveStress()
{
    // Saw-like stack (strong 2nd/3rd harmonics) and a voice-like formant
    // stack, at 110 / 220 / 440 Hz. Audit F1 bound: median within
    // +/-25 cents of the fundamental, zero frames an octave up over >= 1 s.
    const std::vector<float> sawAmps { 0.5f, 0.35f, 0.25f, 0.15f };
    const std::vector<float> voiceAmps { 0.5f, 0.4f, 0.35f, 0.2f, 0.15f, 0.1f };

    for (double frequency : { 110.0, 220.0, 440.0 })
    {
        AudioTestCase testCase;
        testCase.name = "harmonic-saw-" + std::to_string (static_cast<int> (frequency));
        testCase.sampleRate = 48000.0;
        testCase.audio = makeHarmonicTone (48000.0, 1.0, frequency, sawAmps);
        testCase.sourceDescription = "synthetic harmonic stack 1/h-ish (deterministic)";
        testCase.referenceFrames = buildReferenceFrames (
            1.0, kDefaultHopSeconds,
            [=] (double) { return frequency; },
            [] (double) { return true; });
        const LevelAResult result = runLevelA (testCase);
        CHECK (result.analysis != nullptr);
        if (result.analysis == nullptr)
            continue;
        printLevelA ("harmonic-saw-" + std::to_string (static_cast<int> (frequency))
                     + "Hz", result.pitchStats, result.voicing);
        CHECK (result.pitchStats.hasConditionalStats);
        if (result.pitchStats.hasConditionalStats)
            CHECK (result.pitchStats.medianAbsCents <= 25.0);
        CHECK (result.pitchStats.octaveBreakdown.up1 == 0);
        CHECK (result.pitchStats.octaveBreakdown.up2 == 0);
        CHECK (result.pitchStats.octaveBreakdown.up3 == 0);
        CHECK (result.pitchStats.rawPitchAccuracy >= 0.90);
    }

    AudioTestCase voiceCase;
    voiceCase.name = "harmonic-voice-220";
    voiceCase.sampleRate = 48000.0;
    voiceCase.audio = makeHarmonicTone (48000.0, 1.0, 220.0, voiceAmps);
    voiceCase.sourceDescription = "synthetic voice-like formant stack (deterministic)";
    voiceCase.referenceFrames = buildReferenceFrames (
        1.0, kDefaultHopSeconds,
        [] (double) { return 220.0; },
        [] (double) { return true; });
    const LevelAResult voiceResult = runLevelA (voiceCase);
    CHECK (voiceResult.analysis != nullptr);
    if (voiceResult.analysis != nullptr)
    {
        printLevelA ("harmonic-voice-220Hz", voiceResult.pitchStats, voiceResult.voicing);
        CHECK (voiceResult.pitchStats.hasConditionalStats);
        if (voiceResult.pitchStats.hasConditionalStats)
            CHECK (voiceResult.pitchStats.medianAbsCents <= 25.0);
        CHECK (voiceResult.pitchStats.octaveBreakdown.up1 == 0);
        CHECK (voiceResult.pitchStats.octaveBreakdown.up2 == 0);
        CHECK (voiceResult.pitchStats.octaveBreakdown.up3 == 0);
    }
}

// ---- Voicing: digital silence (Part 9) ----

void testBenchmarkSilence()
{
    AudioTestCase testCase;
    testCase.name = "silence";
    testCase.sampleRate = 48000.0;
    testCase.audio = makeSilence (48000.0, 1.0);
    testCase.sourceDescription = "digital silence";
    testCase.referenceFrames = buildReferenceFrames (
        1.0, kDefaultHopSeconds,
        [] (double) { return 0.0; },
        [] (double) { return false; });

    const LevelAResult result = runLevelA (testCase);
    CHECK (result.analysis != nullptr);
    if (result.analysis == nullptr)
        return;
    CHECK (result.analysis->voicedFrameCount == 0);
    CHECK (! result.analysis->hasPitch());
    printLevelA ("silence", result.pitchStats, result.voicing);

    // No voiced reference frames => recall is UNDEFINED (not 1.0).
    // No detected voiced frames => precision is UNDEFINED (not 1.0).
    CHECK (result.voicing.truePositives == 0);
    CHECK (result.voicing.falsePositives == 0);
    CHECK (result.voicing.trueNegatives == 100);
    CHECK (result.voicing.falseNegatives == 0);
    CHECK (! result.voicing.precisionDefined);
    CHECK (! result.voicing.recallDefined);
    CHECK (! result.voicing.f1Defined);
}

// ---- Voicing: white noise only (Part 9 + F7 control) ----

void testBenchmarkWhiteNoise()
{
    AudioTestCase testCase;
    testCase.name = "white-noise-only";
    testCase.sampleRate = 48000.0;
    testCase.audio = makeWhiteNoise (48000.0, 1.0, 0.5f, 12345);
    testCase.sourceDescription = "deterministic white noise, seed 12345";
    testCase.referenceFrames = buildReferenceFrames (
        1.0, kDefaultHopSeconds,
        [] (double) { return 0.0; },
        [] (double) { return false; });

    const LevelAResult result = runLevelA (testCase);
    CHECK (result.analysis != nullptr);
    if (result.analysis == nullptr)
        return;
    printLevelA ("white-noise-only", result.pitchStats, result.voicing);

    // Structural consistency only: the false-positive rate on noise is a
    // MEASUREMENT, not a pass/fail threshold — quasi-periodic noise can
    // pass the YIN fallback (audit W3), and this test records how often.
    CHECK (result.voicing.truePositives + result.voicing.falsePositives
           + result.voicing.trueNegatives + result.voicing.falseNegatives == 100);
    CHECK (result.voicing.truePositives == 0);
    CHECK (result.voicing.falseNegatives == 0);
    benchPrint ("BENCH A white-noise-only: false-positive frames = "
                + std::to_string (result.voicing.falsePositives) + "/100");
}

// ---- Part 8: short notes (40/60/78/100/150/250/500 ms) ----

void testBenchmarkShortNotes()
{
    // Four back-to-back notes, 2-3 semitone steps (above the 0.72/0.75 st
    // hysteresis): C4 D4 E4 G4. Both the struct defaults (min note 60 ms)
    // and the production-mapped settings (min note 77.5 ms) are measured
    // and labelled.
    const std::vector<double> frequencies { 261.63, 293.66, 329.63, 392.00 };
    const std::vector<double> durations { 0.040, 0.060, 0.078, 0.100, 0.150, 0.250, 0.500 };

    for (double duration : durations)
    {
        std::vector<SequenceNoteSpec> specs;
        for (size_t i = 0; i < frequencies.size(); ++i)
            specs.push_back ({ frequencies[i], static_cast<double> (i) * duration,
                               duration, 0.5f });
        AudioTestCase testCase = makeNoteSequenceCase (
            "short-notes-" + std::to_string (static_cast<int> (duration * 1000)) + "ms",
            48000.0, duration * 4.0, specs);

        for (int settingsIndex = 0; settingsIndex < 2; ++settingsIndex)
        {
            const bool production = settingsIndex == 1;
            const PitchAnalysisSettings analysisSettings = production
                ? productionMappedAnalysisSettings() : PitchAnalysisSettings{};
            const NoteSegmentationSettings segmentationSettings = production
                ? productionMappedSegmentationSettings() : NoteSegmentationSettings{};
            const LevelBResult result = runLevelB (
                testCase, analysisSettings, segmentationSettings);
            const std::string label = std::string ("short-")
                + std::to_string (static_cast<int> (duration * 1000)) + "ms-"
                + (production ? "production" : "struct-defaults");
            printLevelB (label, result.noteMatch);

            // Attribute every miss to its stage (detection vs segmentation).
            std::string attributionText;
            if (result.levelA.analysis != nullptr)
            {
                for (const auto& referenceNote : testCase.referenceNotes)
                {
                    bool matched = false;
                    for (const auto& detected : result.notes->notes)
                        if (detected.midiNote == referenceNote.midiNote
                            && std::abs (detected.startSeconds - referenceNote.startSeconds) <= 0.05)
                            matched = true;
                    if (matched)
                        continue;
                    const ShortNoteAttribution attribution = attributeShortNoteFailure (
                        *result.levelA.analysis, referenceNote);
                    attributionText += " [miss midi=" + std::to_string (referenceNote.midiNote)
                        + " stage=" + (attribution.stage == ShortNoteFailureStage::Segmentation
                                       ? "segmentation" : "detection")
                        + " coverage=" + fmt (attribution.frameCoverage, 2)
                        + " voiced=" + std::to_string (attribution.voicedFrames)
                        + "/" + std::to_string (attribution.expectedFrames) + "]";
                }
            }
            benchPrint ("BENCH B " + label + " attribution:" + attributionText
                + (attributionText.empty() ? " (no misses)" : ""));

            // Notes at/above 100 ms are inside the documented operating
            // range (min note 60/77.5 ms): they must all be recalled.
            // Boundary caveat at production settings: the segmenter's
            // 5-frame median can shift a boundary by up to 2 frames, so
            // a 100 ms note (10 frames) can shrink to the 8-frame
            // minimum — 3 of 4 is accepted there, 4 of 4 from 150 ms.
            // Notes below 100 ms are reported per-duration, NOT failed:
            // a 40 ms note is outside the documented operating range by
            // design (minimum note length), so a miss there is a
            // measurement, not a failure.
            if (! production && duration >= 0.100)
            {
                CHECK (result.noteMatch.matched == 4);
                CHECK (result.noteMatch.missed == 0);
                CHECK (result.noteMatch.falsePositives == 0);
            }
            if (production && duration >= 0.150)
            {
                CHECK (result.noteMatch.matched == 4);
                CHECK (result.noteMatch.missed == 0);
                CHECK (result.noteMatch.falsePositives == 0);
            }
            if (production && std::abs (duration - 0.100) < 1e-9)
                CHECK (result.noteMatch.matched >= 3);
        }
    }
}

// ---- F2: vibrato through the analyzer ----

void testBenchmarkVibrato()
{
    for (double depth : { 0.3, 0.7, 1.0 })
    {
        AudioTestCase testCase;
        testCase.name = "vibrato-" + fmt (depth, 1) + "st";
        testCase.sampleRate = 48000.0;
        testCase.audio = makeVibratoTone (48000.0, 2.0, 440.0, depth, 5.5, 0.5f);
        testCase.sourceDescription = "synthetic vibrato, 5.5 Hz, instantaneous "
                                     "pitch trajectory as reference";
        testCase.referenceFrames = vibratoReferenceFrames (2.0, kDefaultHopSeconds,
                                                           440.0, depth, 5.5);
        ReferenceNote note;
        note.startSeconds = 0.0;
        note.endSeconds = 2.0;
        note.frequencyHz = 440.0;
        note.midiNote = 69;
        testCase.referenceNotes.push_back (note);

        const LevelBResult result = runLevelB (testCase);
        CHECK (result.levelA.analysis != nullptr);
        if (result.levelA.analysis == nullptr)
            continue;
        printLevelA ("vibrato-" + fmt (depth, 1) + "st", result.levelA.pitchStats,
                     result.levelA.voicing);
        printLevelB ("vibrato-" + fmt (depth, 1) + "st", result.noteMatch);
        benchPrint ("BENCH B vibrato-" + fmt (depth, 1) + "st: notes="
                    + std::to_string (result.notes->notes.size()));

        // +/-0.3 and +/-0.7 st stay inside the 0.72 st hysteresis, so the
        // curve must be followed and the note must stay whole. +/-1.0 st
        // exceeds the hysteresis and used to split into 8 notes (Phase 1
        // baseline); the Phase 2A vibrato consolidation must keep it
        // whole as well.
        if (depth <= 0.7)
        {
            CHECK (result.levelA.pitchStats.hasConditionalStats);
            if (result.levelA.pitchStats.hasConditionalStats)
                CHECK (result.levelA.pitchStats.medianAbsCents <= 50.0);
            CHECK (result.notes->notes.size() == 1);
            if (result.notes->notes.size() == 1)
                CHECK (result.notes->notes[0].midiNote == 69);
        }
        if (depth >= 1.0)
        {
            CHECK (result.notes->notes.size() == 1);
            if (result.notes->notes.size() == 1)
            {
                CHECK (result.notes->notes[0].midiNote == 69);
                CHECK_NEAR (result.notes->notes[0].startSeconds, 0.0, 0.05);
                CHECK_NEAR (result.notes->notes[0].endSeconds, 2.0, 0.05);
            }
        }
    }
}

// ---- Phase 2A: vibrato-aware segmentation ----
//
// Phase 1 baseline (measured, not assumed): +/-1.0 st sustained
// vibrato at 5.5 Hz split one 2 s note into 8; +/-0.3/0.7 st stayed
// whole; glissando became an 11-note staircase; short-note recall was
// 0/4 at 40/60/78 ms (production). This function re-measures all of
// those after the vibrato-consolidation pass, under BOTH struct-default
// and production-mapped segmentation settings, and adds the
// discrimination cases the fix must not break: genuine semitone steps,
// repeated notes with distinct attacks, fast alternation, irregular
// vibrato, and a clip starting mid-vibrato.
void testBenchmarkVibratoSegmentationPhase2A()
{
    for (int settingsIndex = 0; settingsIndex < 2; ++settingsIndex)
    {
        const bool production = settingsIndex == 1;
        const PitchAnalysisSettings analysisSettings = production
            ? productionMappedAnalysisSettings() : PitchAnalysisSettings{};
        const NoteSegmentationSettings segmentationSettings = production
            ? productionMappedSegmentationSettings() : NoteSegmentationSettings{};
        const std::string settingsLabel = production ? "production" : "struct-defaults";

        // 1. Sustained A4, no vibrato: one note.
        {
            AudioTestCase testCase = makeNoteSequenceCase (
                "2a-stable-a4", 48000.0, 2.0, { { 440.0, 0.0, 2.0, 0.5f } });
            const LevelBResult result = runLevelB (
                testCase, analysisSettings, segmentationSettings);
            benchPrint ("BENCH B 2a-stable-a4-" + settingsLabel + ": notes="
                        + std::to_string (result.notes->notes.size()));
            CHECK (result.notes->notes.size() == 1);
            if (result.notes->notes.size() == 1)
                CHECK (result.notes->notes[0].midiNote == 69);
        }

        // 2-4. Sustained A4 with +/-25, +/-50, +/-100 cents vibrato,
        // at 4.0, 5.5 and 7.0 Hz: one note each.
        for (double depth : { 0.25, 0.5, 1.0 })
        {
            for (double rate : { 4.0, 5.5, 7.0 })
            {
                AudioTestCase testCase;
                testCase.name = "2a-vibrato";
                testCase.sampleRate = 48000.0;
                testCase.audio = makeVibratoTone (
                    48000.0, 2.0, 440.0, depth, rate, 0.5f);
                testCase.referenceFrames = vibratoReferenceFrames (
                    2.0, kDefaultHopSeconds, 440.0, depth, rate);
                ReferenceNote note;
                note.startSeconds = 0.0;
                note.endSeconds = 2.0;
                note.frequencyHz = 440.0;
                note.midiNote = 69;
                testCase.referenceNotes.push_back (note);

                const LevelBResult result = runLevelB (
                    testCase, analysisSettings, segmentationSettings);
                benchPrint ("BENCH B 2a-vibrato-" + fmt (depth, 2) + "st-"
                            + fmt (rate, 1) + "hz-" + settingsLabel + ": notes="
                            + std::to_string (result.notes->notes.size()));
                CHECK (result.notes->notes.size() == 1);
                if (result.notes->notes.size() == 1)
                    CHECK (result.notes->notes[0].midiNote == 69);
            }
        }

        // Peak-start vibrato (clip starts at the vibrato peak): the
        // leading excursion must merge too.
        {
            AudioTestCase testCase;
            testCase.name = "2a-vibrato-peak-start";
            testCase.sampleRate = 48000.0;
            testCase.audio = makeVibratoToneWithPhase (
                48000.0, 2.0, 440.0, 1.0, 5.5, std::numbers::pi / 2.0, 0.5f);
            testCase.referenceFrames = vibratoReferenceFramesWithPhase (
                2.0, kDefaultHopSeconds, 440.0, 1.0, 5.5, std::numbers::pi / 2.0);
            ReferenceNote note;
            note.startSeconds = 0.0;
            note.endSeconds = 2.0;
            note.frequencyHz = 440.0;
            note.midiNote = 69;
            testCase.referenceNotes.push_back (note);

            const LevelBResult result = runLevelB (
                testCase, analysisSettings, segmentationSettings);
            benchPrint ("BENCH B 2a-vibrato-peak-start-" + settingsLabel + ": notes="
                        + std::to_string (result.notes->notes.size()));
            CHECK (result.notes->notes.size() == 1);
            if (result.notes->notes.size() == 1)
                CHECK (result.notes->notes[0].midiNote == 69);
        }

        // Irregular vibrato (drifting depth 0.7-1.1 st, rate 4-7 Hz).
        {
            AudioTestCase testCase;
            testCase.name = "2a-vibrato-irregular";
            testCase.sampleRate = 48000.0;
            testCase.audio = makeIrregularVibratoTone (48000.0, 2.0, 440.0, 0.5f);
            testCase.referenceFrames = irregularVibratoReferenceFrames (
                2.0, kDefaultHopSeconds, 440.0);
            ReferenceNote note;
            note.startSeconds = 0.0;
            note.endSeconds = 2.0;
            note.frequencyHz = 440.0;
            note.midiNote = 69;
            testCase.referenceNotes.push_back (note);

            const LevelBResult result = runLevelB (
                testCase, analysisSettings, segmentationSettings);
            benchPrint ("BENCH B 2a-vibrato-irregular-" + settingsLabel + ": notes="
                        + std::to_string (result.notes->notes.size()));
            CHECK (result.notes->notes.size() == 1);
            if (result.notes->notes.size() == 1)
                CHECK (result.notes->notes[0].midiNote == 69);
        }

        // 5. Two genuine consecutive notes one semitone apart, both
        // directions: two notes each.
        for (const auto& stepPair : { std::pair<double, double> { 440.0, 466.16 },
                                      std::pair<double, double> { 466.16, 440.0 } })
        {
            AudioTestCase testCase = makeNoteSequenceCase (
                "2a-semitone-step", 48000.0, 1.0,
                { { stepPair.first, 0.0, 0.5, 0.5f },
                  { stepPair.second, 0.5, 0.5, 0.5f } });
            const LevelBResult result = runLevelB (
                testCase, analysisSettings, segmentationSettings);
            benchPrint ("BENCH B 2a-semitone-step-" + settingsLabel + ": notes="
                        + std::to_string (result.notes->notes.size()));
            CHECK (result.notes->notes.size() == 2);
            if (result.notes->notes.size() == 2)
            {
                CHECK (result.notes->notes[0].midiNote
                       == static_cast<int> (std::lround (
                           frequencyToMidiDouble (stepPair.first))));
                CHECK (result.notes->notes[1].midiNote
                       == static_cast<int> (std::lround (
                           frequencyToMidiDouble (stepPair.second))));
            }
        }

        // 6. Repeated notes of the same pitch with distinct attacks
        // (silence gaps): the gap is the onset evidence, and the merge
        // pass must not join them.
        for (double gap : { 0.12, 0.15 })
        {
            AudioTestCase testCase = makeNoteSequenceCase (
                "2a-repeated-a4", 48000.0, 1.0 + gap,
                { { 440.0, 0.0, 0.5, 0.5f },
                  { 440.0, 0.5 + gap, 0.5, 0.5f } });
            const LevelBResult result = runLevelB (
                testCase, analysisSettings, segmentationSettings);
            benchPrint ("BENCH B 2a-repeated-a4-gap" + fmt (gap * 1000.0, 0)
                        + "ms-" + settingsLabel + ": notes="
                        + std::to_string (result.notes->notes.size()));
            CHECK (result.notes->notes.size() == 2);
            if (result.notes->notes.size() == 2)
            {
                CHECK (result.notes->notes[0].midiNote == 69);
                CHECK (result.notes->notes[1].midiNote == 69);
            }
        }

        // 7. Fast alternating notes: genuine transitions must not merge.
        for (double step : { 0.15, 0.20 })
        {
            std::vector<SequenceNoteSpec> specs;
            const int count = static_cast<int> (std::llround (1.2 / step));
            for (int i = 0; i < count; ++i)
                specs.push_back ({ (i % 2 == 0) ? 440.0 : 466.16,
                                   static_cast<double> (i) * step, step, 0.5f });
            AudioTestCase testCase = makeNoteSequenceCase (
                "2a-alternating", 48000.0, 1.2, specs);
            const LevelBResult result = runLevelB (
                testCase, analysisSettings, segmentationSettings);
            benchPrint ("BENCH B 2a-alternating-" + fmt (step * 1000.0, 0)
                        + "ms-" + settingsLabel + ": notes="
                        + std::to_string (result.notes->notes.size())
                        + " expected=" + std::to_string (count));
            CHECK (result.notes->notes.size() == static_cast<size_t> (count));
        }
    }

    // 8. Glissando: measure and report only. The count must stay at the
    // Phase 1 baseline (11 notes, contiguous staircase): the fix must
    // not leak into slide handling, and no new glissando policy is
    // implemented here.
    {
        AudioTestCase testCase;
        testCase.name = "glissando-220-440";
        testCase.sampleRate = 48000.0;
        testCase.audio = makeGlissandoTone (48000.0, 1.0, 220.0, 440.0, 0.5f);
        testCase.referenceFrames = glissandoReferenceFrames (1.0, kDefaultHopSeconds,
                                                             220.0, 440.0);
        const LevelBResult result = runLevelB (testCase);
        std::string noteList;
        for (const auto& note : result.notes->notes)
            noteList += " [" + std::to_string (note.midiNote) + " "
                      + fmt (note.startSeconds, 2) + "-" + fmt (note.endSeconds, 2) + "s]";
        benchPrint ("BENCH B 2a-glissando: " + std::to_string (result.notes->notes.size())
                    + " notes:" + noteList);
        CHECK (result.notes->notes.size() == 11);
    }

    // 9. Short notes: report regression results against the Phase 1
    // baseline (struct: 0/4 at 40/60 ms, 4/4 at 78/100 ms;
    // production: 0/4 at 40/60/78 ms, >=3/4 at 100 ms).
    {
        const std::vector<double> frequencies { 261.63, 293.66, 329.63, 392.00 };
        for (double duration : { 0.040, 0.060, 0.078, 0.100 })
        {
            std::vector<SequenceNoteSpec> specs;
            for (size_t i = 0; i < frequencies.size(); ++i)
                specs.push_back ({ frequencies[i], static_cast<double> (i) * duration,
                                   duration, 0.5f });
            AudioTestCase testCase = makeNoteSequenceCase (
                "2a-short-notes", 48000.0, duration * 4.0, specs);
            for (int settingsIndex = 0; settingsIndex < 2; ++settingsIndex)
            {
                const bool production = settingsIndex == 1;
                const LevelBResult result = runLevelB (
                    testCase,
                    production ? productionMappedAnalysisSettings() : PitchAnalysisSettings{},
                    production ? productionMappedSegmentationSettings() : NoteSegmentationSettings{});
                benchPrint ("BENCH B 2a-short-" + std::to_string (static_cast<int> (duration * 1000))
                            + "ms-" + (production ? "production" : "struct-defaults")
                            + ": matched=" + std::to_string (result.noteMatch.matched)
                            + " missed=" + std::to_string (result.noteMatch.missed)
                            + " false=" + std::to_string (result.noteMatch.falsePositives));
                if (! production && duration >= 0.078)
                    CHECK (result.noteMatch.matched == 4);
                if (production && duration >= 0.100)
                    CHECK (result.noteMatch.matched >= 3);
                if (production && duration < 0.100)
                    CHECK (result.noteMatch.matched == 0);
                if (! production && duration < 0.078)
                    CHECK (result.noteMatch.matched == 0);
            }
        }
    }

    // 10. Silence and noise: no new false notes.
    {
        AudioTestCase silenceCase;
        silenceCase.name = "2a-silence";
        silenceCase.sampleRate = 48000.0;
        silenceCase.audio = makeSilence (48000.0, 1.0);
        silenceCase.referenceFrames = buildReferenceFrames (
            1.0, kDefaultHopSeconds,
            [] (double) { return 0.0; }, [] (double) { return false; });
        const LevelBResult silenceResult = runLevelB (silenceCase);
        benchPrint ("BENCH B 2a-silence: notes="
                    + std::to_string (silenceResult.notes->notes.size()));
        CHECK (silenceResult.notes->notes.empty());

        AudioTestCase noiseCase;
        noiseCase.name = "2a-white-noise";
        noiseCase.sampleRate = 48000.0;
        noiseCase.audio = makeWhiteNoise (48000.0, 1.0, 0.5f);
        noiseCase.referenceFrames = buildReferenceFrames (
            1.0, kDefaultHopSeconds,
            [] (double) { return 0.0; }, [] (double) { return false; });
        const LevelBResult noiseResult = runLevelB (noiseCase);
        benchPrint ("BENCH B 2a-white-noise: notes="
                    + std::to_string (noiseResult.notes->notes.size()));
        CHECK (noiseResult.notes->notes.empty());
    }
}

// ---- F3: glissando through the analyzer ----

void testBenchmarkGlissando()
{
    AudioTestCase testCase;
    testCase.name = "glissando-220-440";
    testCase.sampleRate = 48000.0;
    testCase.audio = makeGlissandoTone (48000.0, 1.0, 220.0, 440.0, 0.5f);
    testCase.sourceDescription = "synthetic linear slide 220->440 Hz in 1 s, "
                                 "continuous reference";
    testCase.referenceFrames = glissandoReferenceFrames (1.0, kDefaultHopSeconds,
                                                         220.0, 440.0);

    const LevelBResult result = runLevelB (testCase);
    CHECK (result.levelA.analysis != nullptr);
    if (result.levelA.analysis == nullptr)
        return;
    printLevelA ("glissando-220-440", result.levelA.pitchStats, result.levelA.voicing);
    // No printLevelB here: a slide has no discrete reference notes, so
    // note matching against an empty reference would print a misleading
    // "matched=0 false=N" line. The staircase properties below are the
    // Level B measurement for glissando.

    std::string noteList;
    for (const auto& note : result.notes->notes)
        noteList += " [" + std::to_string (note.midiNote) + " "
                  + fmt (note.startSeconds, 2) + "-" + fmt (note.endSeconds, 2) + "s]";
    benchPrint ("BENCH B glissando: " + std::to_string (result.notes->notes.size())
                + " notes:" + noteList);

    // Level A: the curve follows the continuous slide (window lag makes a
    // loose bound appropriate — the printed median is the measurement).
    CHECK (result.levelA.pitchStats.hasConditionalStats);
    if (result.levelA.pitchStats.hasConditionalStats)
        CHECK (result.levelA.pitchStats.medianAbsCents <= 75.0);

    // Level B: a contiguous staircase, no gaps, no wrong intermediate
    // pitches, monotonically rising. Emitted notes last >= 60 ms by
    // construction, so any out-of-range note is a genuine sustained
    // error, not a one-frame blip.
    CHECK (result.notes->notes.size() >= 2);
    CHECK (result.notes->notes.size() <= 16);
    for (size_t i = 0; i < result.notes->notes.size(); ++i)
    {
        const MelodyNote& note = result.notes->notes[i];
        CHECK (note.midiNote >= 57); // 220 Hz
        CHECK (note.midiNote <= 69); // 440 Hz
        if (i > 0)
        {
            const MelodyNote& previous = result.notes->notes[i - 1];
            CHECK (note.midiNote >= previous.midiNote);
            CHECK (note.startSeconds - previous.endSeconds <= 0.03);
        }
    }
}

// ---- Part 13 + F10: sample-rate sweep (no pitch shift) ----

void testBenchmarkSampleRates()
{
    // 440 Hz must stay ~440 Hz at every rate: the resampler must not
    // shift pitch. 12 kHz upsamples to 24 kHz (quality previously
    // unverified), so it gets a looser bound, labelled as such.
    for (double sampleRate : { 12000.0, 32000.0, 44100.0, 48000.0, 88200.0,
                               96000.0, 192000.0 })
    {
        const auto testCase = makeSineCase ("sine-440-rate", sampleRate, 1.0, 440.0, 0.5f);
        const LevelAResult result = runLevelA (testCase);
        CHECK (result.analysis != nullptr);
        if (result.analysis == nullptr)
            continue;
        const double toleranceHz = std::abs (sampleRate - 12000.0) < 1.0 ? 5.0 : 3.0;
        benchPrint ("BENCH A rate-" + std::to_string (static_cast<int> (sampleRate))
            + "Hz: median=" + fmt (result.analysis->medianFrequencyHz, 2) + "Hz"
            + " conf=" + fmt (result.analysis->medianConfidence, 3)
            + " frames=" + std::to_string (result.analysis->frames.size())
            + " RPA=" + fmt (result.pitchStats.rawPitchAccuracy * 100.0, 1) + "%");
        CHECK_NEAR (result.analysis->medianFrequencyHz, 440.0f, static_cast<float> (toleranceHz));
        CHECK (result.analysis->frames.size() >= 99);
        CHECK (result.analysis->frames.size() <= 101);
        CHECK (result.pitchStats.rawPitchAccuracy >= 0.95);
    }
}

// ---- Part 7: note timing (onset/offset/duration in ms) ----

void testBenchmarkNoteTiming()
{
    // Two notes with 200 ms of silence between them: 440 Hz 0.20-0.70 s,
    // C5 0.90-1.40 s. Both struct defaults and production-mapped settings.
    const std::vector<SequenceNoteSpec> specs {
        { 440.0, 0.20, 0.50, 0.5f },
        { 523.25, 0.90, 0.50, 0.5f },
    };
    AudioTestCase testCase = makeNoteSequenceCase (
        "note-timing", 48000.0, 1.6, specs);

    for (int settingsIndex = 0; settingsIndex < 2; ++settingsIndex)
    {
        const bool production = settingsIndex == 1;
        const PitchAnalysisSettings analysisSettings = production
            ? productionMappedAnalysisSettings() : PitchAnalysisSettings{};
        const NoteSegmentationSettings segmentationSettings = production
            ? productionMappedSegmentationSettings() : NoteSegmentationSettings{};
        const LevelBResult result = runLevelB (
            testCase, analysisSettings, segmentationSettings);
        const std::string label = std::string ("note-timing-")
            + (production ? "production" : "struct-defaults");
        printLevelB (label, result.noteMatch);
        // Print the detected note times too: the mean errors alone do
        // not say whether onsets come early or late (window smear vs
        // median shrinkage pull in opposite directions).
        std::string noteList;
        for (const auto& note : result.notes->notes)
            noteList += " [" + std::to_string (note.midiNote) + " "
                      + fmt (note.startSeconds, 2) + "-" + fmt (note.endSeconds, 2) + "s]";
        benchPrint ("BENCH B " + label + " notes:" + noteList
                    + " (reference: [69 0.20-0.70s] [72 0.90-1.40s])");
        CHECK (result.noteMatch.matched == 2);
        CHECK (result.noteMatch.missed == 0);
        CHECK (result.noteMatch.falsePositives == 0);
        // Tolerance: 10 ms frame grid + up to 20 ms median-smoothing
        // boundary shift + ~10 ms RMS-gate tail.
        CHECK (result.noteMatch.meanAbsOnsetErrorMs <= 40.0);
        CHECK (result.noteMatch.meanAbsOffsetErrorMs <= 40.0);
    }
}

// ---- Level C: MIDI generation integrity ----

void testBenchmarkMidiIntegrity()
{
    const std::vector<SequenceNoteSpec> specs {
        { 440.0, 0.20, 0.50, 0.5f },
        { 523.25, 0.90, 0.50, 0.5f },
    };
    AudioTestCase testCase = makeNoteSequenceCase (
        "midi-integrity", 48000.0, 1.6, specs);

    MidiExportSettings midiSettings;
    midiSettings.bpm = 120.0;
    midiSettings.channel = 3;
    midiSettings.ticksPerQuarterNote = 960;

    const LevelCResult result = runLevelC (testCase, {}, {}, midiSettings);
    const MidiIntegrityReport& integrity = result.integrity;
    benchPrint ("BENCH C midi-integrity: noteOns=" + std::to_string (integrity.noteOnCount)
        + " noteOffs=" + std::to_string (integrity.noteOffCount)
        + " pitchMismatch=" + std::to_string (integrity.pitchMismatches)
        + " channelMismatch=" + std::to_string (integrity.channelMismatches)
        + " velocityOutOfRange=" + std::to_string (integrity.velocityOutOfRange)
        + " timingErrTicks=" + std::to_string (integrity.timingErrorsTicks)
        + " tempoMeta=" + std::string (integrity.hasTempoMeta ? "yes" : "no")
        + " endOfTrack=" + std::string (integrity.hasEndOfTrack ? "yes" : "no")
        + " balanced=" + std::string (integrity.balanced ? "yes" : "no"));
    CHECK (integrity.balanced);
    CHECK (integrity.noteOnCount == 2);
    CHECK (integrity.noteOffCount == 2);
    CHECK (integrity.pitchMismatches == 0);
    CHECK (integrity.channelMismatches == 0);
    CHECK (integrity.velocityOutOfRange == 0);
    CHECK (integrity.timingErrorsTicks == 0);
    CHECK (integrity.hasTempoMeta);
    CHECK (integrity.hasEndOfTrack);

    // Documented exporter behaviour (audit W13): velocity is
    // lround(velocity * 126) clamped to 1..127, so it maxes at 126,
    // never 127. Assert the documented behaviour, not the ideal.
    for (int trackIndex = 0; trackIndex < result.midiFile.getNumTracks(); ++trackIndex)
    {
        const auto* track = result.midiFile.getTrack (trackIndex);
        if (track == nullptr)
            continue;
        for (int i = 0; i < track->getNumEvents(); ++i)
        {
            const auto message = track->getEventPointer (i)->message;
            if (message.isNoteOn())
            {
                CHECK (message.getVelocity() >= 1);
                CHECK (message.getVelocity() <= 126);
                CHECK (message.getChannel() == 3);
            }
        }
    }

    // Empty sequence: no notes, but a well-formed file with end-of-track.
    MelodyNoteSequence empty;
    const juce::MidiFile emptyFile = MidiExporter::createMidiFile (empty, midiSettings);
    int emptyNoteOns = 0;
    bool emptyHasEndOfTrack = false;
    for (int trackIndex = 0; trackIndex < emptyFile.getNumTracks(); ++trackIndex)
    {
        const auto* track = emptyFile.getTrack (trackIndex);
        if (track == nullptr)
            continue;
        for (int i = 0; i < track->getNumEvents(); ++i)
        {
            const auto message = track->getEventPointer (i)->message;
            if (message.isNoteOn())
                ++emptyNoteOns;
            if (message.isEndOfTrackMetaEvent())
                emptyHasEndOfTrack = true;
        }
    }
    CHECK (emptyNoteOns == 0);
    CHECK (emptyHasEndOfTrack);
}

// ---- F4: inharmonic partials (guitar/piano-like) ----

void testBenchmarkInharmonicity()
{
    AudioTestCase testCase;
    testCase.name = "inharmonic-220";
    testCase.sampleRate = 48000.0;
    // Partials stretched sharp 0.3% per partial: 2nd +0.3%, 3rd +0.6%,
    // 4th +0.9% (audit F4 range is 0.2-0.5% per partial).
    testCase.audio = makeInharmonicTone (48000.0, 1.0, 220.0, 0.003,
                                         { 0.5f, 0.3f, 0.2f, 0.1f });
    testCase.sourceDescription = "synthetic inharmonic partials, +0.3% per partial";
    testCase.referenceFrames = buildReferenceFrames (
        1.0, kDefaultHopSeconds,
        [] (double) { return 220.0; },
        [] (double) { return true; });

    const LevelAResult result = runLevelA (testCase);
    CHECK (result.analysis != nullptr);
    if (result.analysis == nullptr)
        return;
    printLevelA ("inharmonic-220Hz", result.pitchStats, result.voicing);
    CHECK (result.pitchStats.hasConditionalStats);
    if (result.pitchStats.hasConditionalStats)
        CHECK (result.pitchStats.medianAbsCents <= 30.0);
    CHECK (result.pitchStats.rawPitchAccuracy >= 0.90);
}

// ---- F5: pluck/attack transient ----

void testBenchmarkTransient()
{
    // 440 Hz sine with a 5 ms broadband burst added at the onset.
    juce::AudioBuffer<float> audio = makeSineTone (48000.0, 0.5, 440.0, 0.5f);
    juce::AudioBuffer<float> burst = makeTransientBurst (
        48000.0, 0.5, 0.0, 0.005, 0.5f, 12345);
    for (int i = 0; i < audio.getNumSamples(); ++i)
        audio.getWritePointer (0)[i] += burst.getReadPointer (0)[i];

    AudioTestCase testCase;
    testCase.name = "transient-onset";
    testCase.sampleRate = 48000.0;
    testCase.audio = std::move (audio);
    testCase.sourceDescription = "sine + 5 ms broadband burst at onset";
    testCase.referenceFrames = buildReferenceFrames (
        0.5, kDefaultHopSeconds,
        [] (double) { return 440.0; },
        [] (double) { return true; });
    ReferenceNote note;
    note.startSeconds = 0.0;
    note.endSeconds = 0.5;
    note.frequencyHz = 440.0;
    note.midiNote = 69;
    testCase.referenceNotes.push_back (note);

    const LevelBResult result = runLevelB (testCase);
    CHECK (result.levelA.analysis != nullptr);
    if (result.levelA.analysis == nullptr)
        return;

    double firstVoicedSeconds = -1.0;
    for (const auto& frame : result.levelA.analysis->frames)
    {
        if (frame.voiced)
        {
            firstVoicedSeconds = frame.timeSeconds;
            break;
        }
    }
    benchPrint ("BENCH A transient-onset: firstVoiced=" + fmt (firstVoicedSeconds, 3) + "s"
        + " noteStart=" + (result.notes->notes.empty() ? "none"
                           : fmt (result.notes->notes[0].startSeconds, 3) + "s"));
    CHECK (firstVoicedSeconds >= 0.0);
    CHECK (firstVoicedSeconds <= 0.03);
    CHECK (result.notes->notes.size() == 1);
    if (result.notes->notes.size() == 1)
        CHECK (result.notes->notes[0].startSeconds <= 0.02);
}

// ---- F6: low-volume sweep (RMS dBFS levels) ----

void testBenchmarkLowVolume()
{
    // Levels are RMS dBFS: peak amplitude = RMS * sqrt(2). The gate
    // compares frame RMS against the silence threshold, so RMS levels
    // are the honest unit here.
    const auto rmsToAmplitude = [] (double rmsDb)
    {
        return static_cast<float> (std::pow (10.0, rmsDb / 20.0) * std::sqrt (2.0));
    };

    for (double levelDb : { -50.0, -60.0, -70.0 })
    {
        const auto testCase = makeSineCase ("quiet-sine", 48000.0, 1.0, 440.0,
                                            rmsToAmplitude (levelDb));
        const LevelAResult result = runLevelA (testCase);
        CHECK (result.analysis != nullptr);
        if (result.analysis == nullptr)
            continue;
        benchPrint ("BENCH A low-volume RMS " + fmt (levelDb, 0) + "dB: voiced="
                    + std::to_string (result.analysis->voicedFrameCount) + "/100");
        if (std::abs (levelDb - -50.0) < 1e-9)
            CHECK (result.analysis->voicedFrameCount > 50);
        if (std::abs (levelDb - -70.0) < 1e-9)
            CHECK (result.analysis->voicedFrameCount == 0);
        // -60 dB RMS sits exactly on the -60 dB default gate: boundary,
        // recorded only, no CHECK.
    }

    // Gate at its extremes: -72 dB lets a -70 dB RMS tone through;
    // -35 dB cuts even a -50 dB RMS tone.
    PitchAnalysisSettings wideGate;
    wideGate.silenceThresholdDb = -72.0f;
    const auto quietCase = makeSineCase ("quiet-sine", 48000.0, 1.0, 440.0,
                                         rmsToAmplitude (-70.0));
    const LevelAResult wideResult = runLevelA (quietCase, wideGate);
    CHECK (wideResult.analysis != nullptr);
    if (wideResult.analysis != nullptr)
    {
        benchPrint ("BENCH A low-volume gate -72dB, tone RMS -70dB: voiced="
                    + std::to_string (wideResult.analysis->voicedFrameCount) + "/100");
        CHECK (wideResult.analysis->voicedFrameCount > 50);
    }

    PitchAnalysisSettings narrowGate;
    narrowGate.silenceThresholdDb = -35.0f;
    const auto loudCase = makeSineCase ("quiet-sine", 48000.0, 1.0, 440.0,
                                        rmsToAmplitude (-50.0));
    const LevelAResult narrowResult = runLevelA (loudCase, narrowGate);
    CHECK (narrowResult.analysis != nullptr);
    if (narrowResult.analysis != nullptr)
    {
        benchPrint ("BENCH A low-volume gate -35dB, tone RMS -50dB: voiced="
                    + std::to_string (narrowResult.analysis->voicedFrameCount) + "/100");
        CHECK (narrowResult.analysis->voicedFrameCount == 0);
    }
}

// ---- F7: noise floor (tone + white noise at controlled SNR) ----

void testBenchmarkNoiseFloor()
{
    juce::AudioBuffer<float> tone = makeSineTone (
        48000.0, 1.0, 440.0, static_cast<float> (0.1 * std::sqrt (2.0))); // RMS -20 dB

    for (double snrDb : { 20.0, 10.0 })
    {
        AudioTestCase testCase;
        testCase.name = "noise-floor-snr" + std::to_string (static_cast<int> (snrDb));
        testCase.sampleRate = 48000.0;
        testCase.audio = mixToneWithNoise (tone, snrDb, 12345);
        testCase.sourceDescription = "440 Hz sine RMS -20 dB + white noise, SNR "
                                   + std::to_string (static_cast<int> (snrDb)) + " dB";
        testCase.referenceFrames = buildReferenceFrames (
            1.0, kDefaultHopSeconds,
            [] (double) { return 440.0; },
            [] (double) { return true; });

        const LevelAResult result = runLevelA (testCase);
        CHECK (result.analysis != nullptr);
        if (result.analysis == nullptr)
            continue;
        printLevelA ("noise-floor-SNR" + std::to_string (static_cast<int> (snrDb)) + "dB",
                     result.pitchStats, result.voicing);
        CHECK (result.analysis->hasPitch());
        CHECK (result.pitchStats.hasConditionalStats);
        if (result.pitchStats.hasConditionalStats)
            CHECK (result.pitchStats.medianAbsCents <= 25.0);
        CHECK (result.pitchStats.rawPitchAccuracy >= 0.90);
    }
}

// ---- F9: whistle ceiling ----

void testBenchmarkWhistleCeiling()
{
    for (double frequency : { 2000.0, 2500.0, 3000.0 })
    {
        const auto testCase = makeSineCase ("whistle", 48000.0, 0.5, frequency, 0.35f);
        const LevelAResult result = runLevelA (testCase);
        CHECK (result.analysis != nullptr);
        if (result.analysis == nullptr)
            continue;
        benchPrint ("BENCH A whistle-" + std::to_string (static_cast<int> (frequency))
            + "Hz: median=" + fmt (result.analysis->medianFrequencyHz, 1) + "Hz"
            + " voiced=" + std::to_string (result.analysis->voicedFrameCount) + "/50");
        if (frequency <= 2500.0)
        {
            CHECK (result.analysis->hasPitch());
            CHECK_NEAR (result.analysis->medianFrequencyHz, static_cast<float> (frequency), 25.0f);
        }
        else
        {
            // 3000 Hz: the true period (8 samples @ 24 kHz) is shorter
            // than the high detector's minLag (9 samples), so YIN cannot
            // represent 3000 Hz at all — the validity window (<= 2625 Hz)
            // is not the operative limit here. What it does instead
            // (reject vs alias to the first representable submultiple,
            // 1500 Hz) is a MEASUREMENT, recorded in the BENCH line
            // above. The only hard assertion is that 3000 Hz is never
            // reported as 3000 Hz.
            if (result.analysis->hasPitch())
                CHECK (std::abs (result.analysis->medianFrequencyHz - 3000.0f) > 25.0f);
            benchPrint ("BENCH A whistle-3000Hz: NOT reported as 3000 Hz "
                        "(min-lag alias/reject — see report)");
        }
    }
}

// ---- F11: long-clip timing + performance (Part 14) ----

void testBenchmarkLongClipTiming()
{
    const auto testCase = makeSineCase ("long-60s", 48000.0, 60.0, 440.0, 0.5f);
    const LevelAResult result = runLevelA (testCase);
    CHECK (result.analysis != nullptr);
    if (result.analysis == nullptr)
        return;
    CHECK (result.analysis->frames.size() == 6000);
    CHECK (result.analysis->hasPitch());
    CHECK_NEAR (result.analysis->medianFrequencyHz, 440.0f, 3.0f);

    const auto printTiming = [] (const std::string& stage, const TimingSummary& timing)
    {
        benchPrint ("BENCH perf " + stage
            + ": median=" + fmt (timing.medianMs, 1) + "ms"
            + " p95=" + fmt (timing.p95Ms, 1) + "ms"
            + " mean=" + fmt (timing.meanMs, 1) + "ms"
            + " min=" + fmt (timing.minMs, 1) + "ms"
            + " max=" + fmt (timing.maxMs, 1) + "ms"
            + " RTF=" + fmt (timing.realTimeFactor, 2)
            + " (n=" + std::to_string (timing.iterations) + ")");
    };

    // 60 s of audio is slow to analyze (resampler trig + YIN): 1 warm-up
    // + 2 measured iterations for the analyze-bearing stages, 1 + 5 for
    // the fast stages. Timing never gates a CHECK.
    const TimingSummary analyzeTiming = benchmarkStage (
        [&] { MonophonicPitchAnalyzer::analyze (testCase.audio, testCase.sampleRate); },
        60.0, 1, 2);
    printTiming ("analyze-60s", analyzeTiming);

    const TimingSummary segmentTiming = benchmarkStage (
        [&] { NoteSegmenter::segment (*result.analysis); },
        60.0, 1, 5);
    printTiming ("segment-60s", segmentTiming);

    const std::shared_ptr<MelodyNoteSequence> notes =
        NoteSegmenter::segment (*result.analysis);
    const TimingSummary midiTiming = benchmarkStage (
        [&] { MidiExporter::createMidiFile (*notes); },
        60.0, 1, 5);
    printTiming ("midi-60s", midiTiming);

    const TimingSummary totalTiming = benchmarkStage (
        [&]
        {
            auto analysis = MonophonicPitchAnalyzer::analyze (
                testCase.audio, testCase.sampleRate);
            auto sequence = NoteSegmenter::segment (*analysis);
            MidiExporter::createMidiFile (*sequence);
        },
        60.0, 1, 2);
    printTiming ("total-60s", totalTiming);

    CHECK (analyzeTiming.iterations == 2);
    CHECK (segmentTiming.iterations == 5);
    CHECK (midiTiming.iterations == 5);
    CHECK (totalTiming.iterations == 2);
    CHECK (analyzeTiming.medianMs > 0.0);
    // No CHECK compares timing values against each other: medians of 2
    // iterations carry CPU noise, and timing never gates correctness.
}

// ---- F15: breathy/consonant gaps (silence gaps in a sustained tone) ----

void testBenchmarkBreathGaps()
{
    // Sustained 440 Hz, 2 s, with silence gaps of 30 ms (0.50-0.53 s),
    // 60 ms (1.00-1.06 s) and 120 ms (1.50-1.62 s). Audit F15 expects
    // the 30 ms gap bridged and the 120 ms gap split. The 60 ms gap is
    // recorded, not asserted: the 42.7 ms RMS window smears across gap
    // edges, so the EFFECTIVE unvoiced run is shorter than the acoustic
    // gap and may fall inside the 40 ms bridge — the measurement decides.
    juce::AudioBuffer<float> audio = makeSineTone (48000.0, 2.0, 440.0, 0.5f);
    const auto zeroGap = [&] (double start, double end)
    {
        const int first = static_cast<int> (std::llround (start * 48000.0));
        const int last = static_cast<int> (std::llround (end * 48000.0));
        for (int i = first; i < last && i < audio.getNumSamples(); ++i)
            audio.getWritePointer (0)[i] = 0.0f;
    };
    zeroGap (0.50, 0.53);
    zeroGap (1.00, 1.06);
    zeroGap (1.50, 1.62);

    AudioTestCase testCase;
    testCase.name = "breath-gaps";
    testCase.sampleRate = 48000.0;
    testCase.audio = std::move (audio);
    testCase.sourceDescription = "sustained 440 Hz with 30/60/120 ms silence gaps";
    testCase.referenceFrames = buildReferenceFrames (
        2.0, kDefaultHopSeconds,
        [] (double) { return 440.0; },
        [] (double time)
        {
            return ! ((time >= 0.50 && time < 0.53)
                      || (time >= 1.00 && time < 1.06)
                      || (time >= 1.50 && time < 1.62));
        });
    const std::vector<ReferenceNote> referenceNotes {
        { 0.00, 0.50, 69, 440.0 },
        { 0.53, 1.00, 69, 440.0 },
        { 1.06, 1.50, 69, 440.0 },
        { 1.62, 2.00, 69, 440.0 },
    };
    testCase.referenceNotes = referenceNotes;

    const LevelBResult result = runLevelB (testCase);
    CHECK (result.levelA.analysis != nullptr);
    if (result.levelA.analysis == nullptr)
        return;

    std::string noteList;
    for (const auto& note : result.notes->notes)
        noteList += " [" + std::to_string (note.midiNote) + " "
                  + fmt (note.startSeconds, 2) + "-" + fmt (note.endSeconds, 2) + "s]";
    benchPrint ("BENCH B breath-gaps: " + std::to_string (result.notes->notes.size())
                + " notes:" + noteList);

    // 30 ms gap bridged: some note spans across 0.50-0.53.
    bool bridges30ms = false;
    for (const auto& note : result.notes->notes)
        if (note.startSeconds <= 0.50 && note.endSeconds >= 0.53)
            bridges30ms = true;
    CHECK (bridges30ms);

    // 120 ms gap splits: some note ends at/before 1.60 and another starts
    // at/after 1.60.
    bool endsBefore120msGap = false, startsAfter120msGap = false;
    for (const auto& note : result.notes->notes)
    {
        if (note.endSeconds <= 1.60 && note.endSeconds >= 1.40)
            endsBefore120msGap = true;
        if (note.startSeconds >= 1.60 && note.startSeconds <= 1.66)
            startsAfter120msGap = true;
    }
    CHECK (endsBefore120msGap);
    CHECK (startsAfter120msGap);

    // 60 ms gap: recorded only (RMS smear may or may not bridge it).
    bool bridges60ms = false;
    for (const auto& note : result.notes->notes)
        if (note.startSeconds <= 1.00 && note.endSeconds >= 1.06)
            bridges60ms = true;
    benchPrint ("BENCH B breath-gaps: 60ms gap "
                + std::string (bridges60ms ? "BRIDGED" : "SPLIT"));
}

// ---- Part 9: voicing confusion matrix across signal types ----

void testBenchmarkVoicingMatrix()
{
    struct VoicingCase
    {
        std::string name;
        juce::AudioBuffer<float> audio;
        bool tonePresent; // reference voiced throughout when true
    };

    juce::AudioBuffer<float> quietTone = makeSineTone (
        48000.0, 1.0, 440.0,
        static_cast<float> (std::pow (10.0, -70.0 / 20.0) * std::sqrt (2.0))); // RMS -70 dB
    juce::AudioBuffer<float> loudTone = makeSineTone (
        48000.0, 1.0, 440.0,
        static_cast<float> (0.1 * std::sqrt (2.0))); // RMS -20 dB
    juce::AudioBuffer<float> noisyTone = mixToneWithNoise (loudTone, 10.0, 12345);

    const std::vector<VoicingCase> cases {
        { "silence", makeSilence (48000.0, 1.0), false },
        { "white-noise", makeWhiteNoise (48000.0, 1.0, 0.5f, 12345), false },
        { "pink-noise", makePinkNoise (48000.0, 1.0, 0.1f, 12345), false },
        { "breath-like", makeBreathLikeNoise (48000.0, 1.0, 0.5f, 12345), false },
        { "transient-only", makeTransientBurst (48000.0, 1.0, 0.5, 0.005, 0.5f, 12345), false },
        { "quiet-sine-70dB", std::move (quietTone), true },
        { "sine-plus-noise-10dB", std::move (noisyTone), true },
    };

    for (const auto& voicingCase : cases)
    {
        AudioTestCase testCase;
        testCase.name = "voicing-" + voicingCase.name;
        testCase.sampleRate = 48000.0;
        testCase.audio = voicingCase.audio;
        testCase.sourceDescription = "voicing matrix case: " + voicingCase.name;
        const bool tonePresent = voicingCase.tonePresent;
        testCase.referenceFrames = buildReferenceFrames (
            1.0, kDefaultHopSeconds,
            [] (double) { return 440.0; },
            [=] (double) { return tonePresent; });

        const LevelAResult result = runLevelA (testCase);
        CHECK (result.analysis != nullptr);
        if (result.analysis == nullptr)
            continue;
        const VoicingConfusion& confusion = result.voicing;
        benchPrint ("BENCH V " + voicingCase.name
            + ": TP=" + std::to_string (confusion.truePositives)
            + " FP=" + std::to_string (confusion.falsePositives)
            + " TN=" + std::to_string (confusion.trueNegatives)
            + " FN=" + std::to_string (confusion.falseNegatives)
            + " P=" + (confusion.precisionDefined ? fmt (confusion.precision, 3) : "undef")
            + " R=" + (confusion.recallDefined ? fmt (confusion.recall, 3) : "undef")
            + " F1=" + (confusion.f1Defined ? fmt (confusion.f1, 3) : "undef"));

        // Structural consistency for every case.
        CHECK (confusion.truePositives + confusion.falsePositives
               + confusion.trueNegatives + confusion.falseNegatives == 100);
        if (voicingCase.name == "silence")
            CHECK (confusion.falsePositives == 0);
        if (voicingCase.name == "quiet-sine-70dB")
            CHECK (result.analysis->voicedFrameCount == 0);
        if (voicingCase.name == "transient-only")
            CHECK (confusion.falsePositives <= 10);
        if (voicingCase.name == "sine-plus-noise-10dB")
        {
            CHECK (confusion.recallDefined);
            if (confusion.recallDefined)
                CHECK (confusion.recall >= 0.9);
        }
    }
}

// ---- Parts 11-12: real-audio dataset interface (interface only) ----

void writeTextFile (const juce::File& file, const std::string& content)
{
    file.deleteFile();
    file.replaceWithText (juce::String (content));
}

juce::File writeSineWav (const juce::File& dir, const char* fileName,
                         double sampleRate, double durationSeconds, double frequencyHz)
{
    const juce::File wavFile = dir.getChildFile (fileName);
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> source (1, samples);
    for (int i = 0; i < samples; ++i)
        source.getWritePointer (0)[i] = 0.5f * std::sin (
            2.0f * 3.14159265f * static_cast<float> (frequencyHz)
            * static_cast<float> (i) / static_cast<float> (sampleRate));

    std::unique_ptr<juce::FileOutputStream> stream (wavFile.createOutputStream());
    if (stream == nullptr)
        return wavFile;
    juce::WavAudioFormat format;
    std::unique_ptr<juce::AudioFormatWriter> writer (
        format.createWriterFor (stream.release(), sampleRate, 1, 16, {}, 0));
    if (writer == nullptr)
        return wavFile;
    writer->writeFromAudioSampleBuffer (source, 0, samples);
    writer.reset(); // flushes and closes the file
    return wavFile;
}

void testBenchmarkDatasetInterface()
{
    const juce::File dir = getBenchmarkTestDir();

    // A synthetic WAV + annotations, loaded THROUGH the dataset interface:
    // this exercises manifest parsing, audio decoding and annotation
    // parsing end to end without inventing any real-audio annotation.
    const juce::File wavFile = writeSineWav (dir, "synth-440.wav", 48000.0, 0.5, 440.0);
    CHECK (wavFile.existsAsFile());

    std::string framesCsv;
    for (int i = 0; i < 50; ++i)
    {
        std::ostringstream line;
        line << std::fixed << std::setprecision (2) << (i * 0.01) << ",440,1\n";
        framesCsv += line.str();
    }
    writeTextFile (dir.getChildFile ("synth-440-frames.csv"), framesCsv);
    writeTextFile (dir.getChildFile ("synth-440-notes.csv"), "0,0.5,69\n");
    writeTextFile (dir.getChildFile ("manifest.csv"),
        "audioPath,framesPath,notesPath,instrument,license,datasetVersion\n"
        "synth-440.wav,synth-440-frames.csv,synth-440-notes.csv,synth,CC-BY-4.0,test-v1\n");

    CsvAnnotationDataset dataset;
    std::string error;
    CHECK (dataset.loadManifest (dir.getChildFile ("manifest.csv"), error));
    CHECK (error.empty());
    CHECK (dataset.entries().size() == 1);

    const DatasetLoadResult loaded = dataset.loadCase (dataset.entries()[0]);
    CHECK (loaded.ok);
    CHECK (loaded.error.empty());
    if (loaded.ok)
    {
        CHECK (loaded.testCase.audio.getNumSamples() == 24000);
        CHECK_NEAR (loaded.testCase.sampleRate, 48000.0, 1.0);
        CHECK (loaded.testCase.referenceFrames.size() == 50);
        CHECK (loaded.testCase.referenceNotes.size() == 1);
        if (loaded.testCase.referenceNotes.size() == 1)
            CHECK (loaded.testCase.referenceNotes[0].midiNote == 69);
        CHECK (loaded.testCase.datasetVersion == "test-v1");
        benchPrint ("BENCH dataset: synthetic WAV + annotations round-tripped "
                    "through CsvAnnotationDataset (CC-BY-4.0, test-v1)");
    }

    // Missing manifest: clean error, no crash.
    CsvAnnotationDataset missingDataset;
    CHECK (! missingDataset.loadManifest (dir.getChildFile ("no-such-manifest.csv"), error));
    CHECK (! error.empty());

    // Unverified license: refused, never loaded.
    writeTextFile (dir.getChildFile ("manifest-unknown-license.csv"),
        "audioPath,framesPath,notesPath,instrument,license,datasetVersion\n"
        "synth-440.wav,synth-440-frames.csv,synth-440-notes.csv,synth,unknown,test-v1\n");
    CsvAnnotationDataset unknownDataset;
    CHECK (unknownDataset.loadManifest (dir.getChildFile ("manifest-unknown-license.csv"), error));
    const DatasetLoadResult refused = unknownDataset.loadCase (unknownDataset.entries()[0]);
    CHECK (! refused.ok);
    CHECK (refused.error.find ("license") != std::string::npos);
    benchPrint ("BENCH dataset: unknown-license entry refused as designed");

    // Missing audio file: clean error, no crash.
    writeTextFile (dir.getChildFile ("manifest-missing-audio.csv"),
        "audioPath,framesPath,notesPath,instrument,license,datasetVersion\n"
        "no-such-file.wav,synth-440-frames.csv,synth-440-notes.csv,synth,CC-BY-4.0,test-v1\n");
    CsvAnnotationDataset missingAudioDataset;
    CHECK (missingAudioDataset.loadManifest (dir.getChildFile ("manifest-missing-audio.csv"), error));
    const DatasetLoadResult missing = missingAudioDataset.loadCase (missingAudioDataset.entries()[0]);
    CHECK (! missing.ok);
    CHECK (missing.error.find ("not found") != std::string::npos);
}

// ---- Part 15: reproducibility record ----

void testBenchmarkReproducibilityRecord()
{
    const BenchmarkEnvironment env = collectEnvironment (
        48000.0, "synthetic-v1", PitchAnalysisSettings{}, NoteSegmentationSettings{});
    CHECK (! env.toString().empty());
    CHECK (! env.compiler.empty());
    CHECK (env.compiler != "unknown compiler");
    CHECK (env.buildConfig == "Release" || env.buildConfig == "Debug");
    CHECK (env.timestampIso.size() == 19); // YYYY-MM-DDTHH:MM:SS
    CHECK (! env.analysisConfigSummary.empty());
    CHECK (! env.segmentationConfigSummary.empty());
    benchPrint ("BENCH env:\n" + env.toString());
}

} // namespace

BenchmarkTestOutcome runPitchBenchmarkTests()
{
    testBenchmarkOctaveClassifierUnit();
    testBenchmarkSine440();
    testBenchmarkHarmonicOctaveStress();
    testBenchmarkSilence();
    testBenchmarkWhiteNoise();
    testBenchmarkShortNotes();
    testBenchmarkVibrato();
    testBenchmarkVibratoSegmentationPhase2A();
    testBenchmarkGlissando();
    testBenchmarkSampleRates();
    testBenchmarkNoteTiming();
    testBenchmarkMidiIntegrity();
    testBenchmarkInharmonicity();
    testBenchmarkTransient();
    testBenchmarkLowVolume();
    testBenchmarkNoiseFloor();
    testBenchmarkWhistleCeiling();
    testBenchmarkLongClipTiming();
    testBenchmarkBreathGaps();
    testBenchmarkVoicingMatrix();
    testBenchmarkDatasetInterface();
    testBenchmarkReproducibilityRecord();

    // Real-audio validation (separate file, own counters; folded into
    // the suite totals exactly like the benchmark block above). When no
    // dataset manifest is found it returns {0, 0}, so CI stays green.
    {
        const auto realAudioOutcome = runRealAudioValidationTests();
        checks += realAudioOutcome.checks;
        failures += realAudioOutcome.failures;
    }

    getBenchmarkTestDir().deleteRecursively();

    std::cout << "Pitch Benchmark tests: " << (checks - failures) << "/" << checks
              << " checks passed";
    if (failures == 0)
        std::cout << " - ALL GREEN\n";
    else
        std::cout << " - " << failures << " FAILED\n";

    return { checks, failures };
}

} // namespace pitchbenchmark
