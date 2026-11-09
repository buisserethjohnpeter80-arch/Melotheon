// MELOTHÉON — Real Audio Validation implementation.
//
// See RealAudioValidation.h for the scope, dataset location rules and
// CHECK discipline. Measurement only: every level calls the real
// production analysis code. No production file is modified by this
// translation unit and no threshold is changed here.

#include "RealAudioValidation.h"

#include "Analysis/MelodyEditor.h"
#include "Analysis/MonophonicPitchAnalyzer.h"
#include "Analysis/NoteSegmenter.h"
#include "Audio/AudioClip.h"
#include "Audio/AudioFileLoader.h"
#include "Midi/MidiExporter.h"
#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

std::string fmtOrUndef (double value, bool defined, int precision = 3)
{
    return defined ? fmt (value, precision) : "undef";
}

void benchPrint (const std::string& line)
{
    std::cout << line << "\n";
}

juce::File getRealAudioTestDir()
{
    juce::File dir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                         .getChildFile ("MelotheonRealAudioValidation");
    dir.createDirectory();
    return dir;
}

// ---------------------------------------------------------------------------
// Dataset location
// ---------------------------------------------------------------------------

// Resolution order: MELOTHEON_REAL_AUDIO_MANIFEST (one manifest.csv),
// then MELOTHEON_REAL_AUDIO_DIR (<DIR>/manifest.csv, else every
// immediate subdirectory's manifest.csv, sorted by directory name so
// the run order is deterministic). Returns an empty list when no
// manifest is found — the caller then skips cleanly.
std::vector<juce::File> resolveManifestFiles()
{
    std::vector<juce::File> manifests;
    if (const char* manifestEnv = std::getenv ("MELOTHEON_REAL_AUDIO_MANIFEST");
        manifestEnv != nullptr && manifestEnv[0] != '\0')
    {
        manifests.push_back (juce::File (juce::String (manifestEnv)));
        return manifests;
    }
    if (const char* dirEnv = std::getenv ("MELOTHEON_REAL_AUDIO_DIR");
        dirEnv != nullptr && dirEnv[0] != '\0')
    {
        const juce::File dir { juce::String (dirEnv) };
        const juce::File direct = dir.getChildFile ("manifest.csv");
        if (direct.existsAsFile())
        {
            manifests.push_back (direct);
            return manifests;
        }
        juce::Array<juce::File> children;
        dir.findChildFiles (children, juce::File::findDirectories, false);
        std::vector<juce::File> sorted (children.begin(), children.end());
        std::sort (sorted.begin(), sorted.end(),
                   [] (const juce::File& a, const juce::File& b)
                   { return a.getFileName() < b.getFileName(); });
        for (const auto& child : sorted)
        {
            const juce::File manifest = child.getChildFile ("manifest.csv");
            if (manifest.existsAsFile())
                manifests.push_back (manifest);
        }
    }
    return manifests;
}

// The tin-whistle smoke file lives next to the datasets, not in any
// manifest (file-level annotations only — unscored by design). Look for
// it under the datasets root: the MELOTHEON_REAL_AUDIO_DIR, or the
// parent of the first manifest's directory. Returns an empty File when
// absent; the caller then skips that leg.
juce::File findWhistleFile (const std::vector<juce::File>& manifests)
{
    const juce::String fileName ("HFA_IrishWhistleD_Tune_ShortSnippet.wav");
    if (const char* dirEnv = std::getenv ("MELOTHEON_REAL_AUDIO_DIR");
        dirEnv != nullptr && dirEnv[0] != '\0')
    {
        const juce::File candidate = juce::File (juce::String (dirEnv))
            .getChildFile ("whistling").getChildFile (fileName);
        if (candidate.existsAsFile())
            return candidate;
    }
    if (! manifests.empty())
    {
        const juce::File candidate = manifests[0].getParentDirectory()
            .getParentDirectory().getChildFile ("whistling").getChildFile (fileName);
        if (candidate.existsAsFile())
            return candidate;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Per-case printing (stable BENCH REAL lines for before/after diffing)
// ---------------------------------------------------------------------------

void printRealLevelA (const std::string& name, const std::string& settingsLabel,
                      const CentsErrorStats& stats, const VoicingConfusion& voicing)
{
    benchPrint ("BENCH REAL A " + name + " [" + settingsLabel + "]"
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
        + " voicing P=" + fmtOrUndef (voicing.precision, voicing.precisionDefined)
        + " R=" + fmtOrUndef (voicing.recall, voicing.recallDefined)
        + " F1=" + fmtOrUndef (voicing.f1, voicing.f1Defined));
}

void printRealLevelB (const std::string& name, const std::string& settingsLabel,
                      const MelodyNoteSequence& detected,
                      const NoteMatchResult& match, bool scored,
                      int longRefNotes, int overlappingDetected)
{
    std::string line = "BENCH REAL B " + name + " [" + settingsLabel + "]"
        + ": detected=" + std::to_string (static_cast<int> (detected.notes.size()));
    if (! scored)
    {
        benchPrint (line + " UNSCORED (no note reference)");
        return;
    }
    line += " matched=" + std::to_string (match.matched)
        + " missed=" + std::to_string (match.missed)
        + " false=" + std::to_string (match.falsePositives)
        + " P=" + fmtOrUndef (match.precision, match.precisionDefined)
        + " R=" + fmtOrUndef (match.recall, match.recallDefined)
        + " F1=" + fmtOrUndef (match.f1, match.f1Defined)
        + " onsetMedAbs=" + (std::isnan (match.medianAbsOnsetErrorMs) ? "n/a" : fmt (match.medianAbsOnsetErrorMs, 1) + "ms")
        + " offsetMedAbs=" + (std::isnan (match.medianAbsOffsetErrorMs) ? "n/a" : fmt (match.medianAbsOffsetErrorMs, 1) + "ms")
        + " durMeanAbs=" + (std::isnan (match.meanAbsDurationErrorMs) ? "n/a" : fmt (match.meanAbsDurationErrorMs, 1) + "ms");
    if (longRefNotes > 0)
        line += " longRefNotes=" + std::to_string (longRefNotes)
            + " fragOverlap=" + std::to_string (overlappingDetected)
            + " fragRatio=" + fmt (static_cast<double> (overlappingDetected)
                                   / static_cast<double> (longRefNotes), 2);
    benchPrint (line);
}

void printRealLevelC (const std::string& name, const std::string& settingsLabel,
                      const MidiIntegrityReport& integrity)
{
    benchPrint ("BENCH REAL C " + name + " [" + settingsLabel + "]"
        + ": noteOns=" + std::to_string (integrity.noteOnCount)
        + " noteOffs=" + std::to_string (integrity.noteOffCount)
        + " balanced=" + std::string (integrity.balanced ? "yes" : "NO")
        + " pitchMismatch=" + std::to_string (integrity.pitchMismatches)
        + " channelMismatch=" + std::to_string (integrity.channelMismatches)
        + " timingErrTicks=" + std::to_string (integrity.timingErrorsTicks)
        + " tempoMeta=" + std::string (integrity.hasTempoMeta ? "yes" : "no")
        + " endOfTrack=" + std::string (integrity.hasEndOfTrack ? "yes" : "no"));
}

// ---------------------------------------------------------------------------
// Fragmentation indicator for long (vibrato-prone) reference notes
// ---------------------------------------------------------------------------

// Counts reference notes >= minSeconds long, and how many detected notes
// overlap their time ranges. A sustained vibrato note that fragments
// into N detected notes contributes N to the overlap count, so a ratio
// above 1.0 means fragmentation on long notes. Pure measurement.
struct LongNoteFragmentation
{
    int longRefNotes = 0;
    int overlappingDetected = 0;
};

LongNoteFragmentation countLongNoteFragmentation (
    const MelodyNoteSequence& detected,
    const std::vector<ReferenceNote>& reference,
    double minSeconds = 1.0)
{
    LongNoteFragmentation result;
    for (const auto& ref : reference)
    {
        if (ref.durationSeconds() < minSeconds)
            continue;
        ++result.longRefNotes;
        for (const auto& note : detected.notes)
            if (note.startSeconds < ref.endSeconds && note.endSeconds > ref.startSeconds)
                ++result.overlappingDetected;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Per-case aggregate (for the summary lines)
// ---------------------------------------------------------------------------

struct RealAggregate
{
    int cases = 0;
    int scoredCases = 0;
    int voicedRefFrames = 0;
    int detectedVoicedFrames = 0;
    double inTuneWeighted = 0.0; // sum of RPA * voicedRefFrames
    int refNotes = 0;
    int matched = 0;
    int missed = 0;
    int falsePositives = 0;
    int longRefNotes = 0;
    int overlappingDetected = 0;

    void addCase (const LevelCResult& result, bool scored)
    {
        ++cases;
        const auto& stats = result.levelB.levelA.pitchStats;
        voicedRefFrames += stats.voicedReferenceFrames;
        detectedVoicedFrames += stats.detectedVoicedFrames;
        inTuneWeighted += stats.rawPitchAccuracy
                        * static_cast<double> (stats.voicedReferenceFrames);
        if (! scored)
            return;
        ++scoredCases;
        refNotes += result.levelB.noteMatch.matched + result.levelB.noteMatch.missed;
        matched += result.levelB.noteMatch.matched;
        missed += result.levelB.noteMatch.missed;
        falsePositives += result.levelB.noteMatch.falsePositives;
    }

    void addFragmentation (const LongNoteFragmentation& frag)
    {
        longRefNotes += frag.longRefNotes;
        overlappingDetected += frag.overlappingDetected;
    }
};

void printRealSummary (const std::string& settingsLabel, const RealAggregate& agg)
{
    std::string line = "BENCH REAL SUMMARY [" + settingsLabel + "]"
        + ": cases=" + std::to_string (agg.cases)
        + " scoredCases=" + std::to_string (agg.scoredCases)
        + " voicedRefFrames=" + std::to_string (agg.voicedRefFrames)
        + " detectedVoiced=" + std::to_string (agg.detectedVoicedFrames)
        + " meanRPA=" + (agg.voicedRefFrames > 0
            ? fmt (agg.inTuneWeighted / static_cast<double> (agg.voicedRefFrames) * 100.0, 1) + "%"
            : "n/a");
    if (agg.scoredCases > 0)
    {
        const int predicted = agg.matched + agg.falsePositives;
        const int actual = agg.matched + agg.missed;
        const double precision = predicted > 0
            ? static_cast<double> (agg.matched) / predicted
            : std::numeric_limits<double>::quiet_NaN();
        const double recall = actual > 0
            ? static_cast<double> (agg.matched) / actual
            : std::numeric_limits<double>::quiet_NaN();
        line += " refNotes=" + std::to_string (agg.refNotes)
            + " matched=" + std::to_string (agg.matched)
            + " missed=" + std::to_string (agg.missed)
            + " false=" + std::to_string (agg.falsePositives)
            + " P=" + (std::isnan (precision) ? "undef" : fmt (precision, 3))
            + " R=" + (std::isnan (recall) ? "undef" : fmt (recall, 3));
        if (! std::isnan (precision) && ! std::isnan (recall)
            && (precision + recall) > 0.0)
            line += " F1=" + fmt (2.0 * precision * recall / (precision + recall), 3);
        else
            line += " F1=undef";
    }
    if (agg.longRefNotes > 0)
        line += " longRefNotes=" + std::to_string (agg.longRefNotes)
            + " fragRatio=" + fmt (static_cast<double> (agg.overlappingDetected)
                                   / static_cast<double> (agg.longRefNotes), 2);
    benchPrint (line);
}

// ---------------------------------------------------------------------------
// One real-audio case under one settings set
// ---------------------------------------------------------------------------

// Runs Level C (which runs A and B) and prints the BENCH REAL lines.
// CHECKs assert only properties true under BOTH segmenters: the analysis
// exists, one observation per reference frame, and Level C MIDI
// integrity of whatever notes were produced. The vibrato delta is the
// DIFFERENCE in the printed numbers between the pre-2A and Phase 2A
// binaries, never a pass/fail here.
void runRealCase (const AudioTestCase& testCase, const std::string& settingsLabel,
                  const PitchAnalysisSettings& analysisSettings,
                  const NoteSegmentationSettings& segmentationSettings,
                  RealAggregate& aggregate)
{
    const LevelCResult result = runLevelC (
        testCase, analysisSettings, segmentationSettings, MidiExportSettings{});
    const bool scored = ! testCase.referenceNotes.empty();

    CHECK (result.levelB.levelA.analysis != nullptr);
    CHECK (static_cast<int> (result.levelB.levelA.observations.size())
           == static_cast<int> (testCase.referenceFrames.size()));
    CHECK (result.integrity.balanced);
    CHECK (result.integrity.pitchMismatches == 0);
    CHECK (result.integrity.channelMismatches == 0);
    CHECK (result.integrity.timingErrorsTicks == 0);

    const std::string name = testCase.name + " (" + testCase.sourceDescription + ")";
    printRealLevelA (name, settingsLabel,
                     result.levelB.levelA.pitchStats, result.levelB.levelA.voicing);
    const LongNoteFragmentation frag = countLongNoteFragmentation (
        *result.levelB.notes, testCase.referenceNotes);
    printRealLevelB (name, settingsLabel, *result.levelB.notes,
                     result.levelB.noteMatch, scored,
                     frag.longRefNotes, frag.overlappingDetected);
    printRealLevelC (name, settingsLabel, result.integrity);

    aggregate.addCase (result, scored);
    aggregate.addFragmentation (frag);
}

// ---------------------------------------------------------------------------
// Imported-audio workflow test (engine level only)
// ---------------------------------------------------------------------------

juce::File writeWavFile (const juce::File& dir, const char* fileName,
                         const juce::AudioBuffer<float>& buffer, double sampleRate)
{
    const juce::File wavFile = dir.getChildFile (fileName);
    std::unique_ptr<juce::FileOutputStream> stream (wavFile.createOutputStream());
    if (stream == nullptr)
        return wavFile;
    juce::WavAudioFormat format;
    std::unique_ptr<juce::AudioFormatWriter> writer (
        format.createWriterFor (stream.release(), sampleRate, 1, 16, {}, 0));
    if (writer == nullptr)
        return wavFile;
    writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples());
    writer.reset(); // flushes and closes the file
    return wavFile;
}

// Times come from floating-point frame arithmetic: a note's end is
// frames[last].timeSeconds + hopSeconds while the next note's start is
// frames[next].timeSeconds (index * hopSeconds), and (e-1)*hop + hop can
// differ from e*hop by ~1 ULP. On vocadito_1 this produced 7 shared
// boundaries off by < 0.001 ns — musically zero (0 MIDI ticks) but enough
// to fail a strict `<` comparison. A 1 ns tolerance is 7 orders of
// magnitude below the 10 ms frame hop, so it cannot hide a real overlap.
// The production 1-ULP boundary imprecision is documented in the
// validation report for a future decision; it is not fixed here.
bool sequenceIsSortedAndNonOverlapping (const MelodyNoteSequence& sequence)
{
    constexpr double kBoundaryToleranceSeconds = 1e-9;
    for (size_t i = 1; i < sequence.notes.size(); ++i)
        if (sequence.notes[i].startSeconds + kBoundaryToleranceSeconds
                < sequence.notes[i - 1].endSeconds)
            return false;
    return true;
}

// Loads a WAV through the PRODUCTION import path, analyzes it, segments
// it, edits the result, exports MIDI, and exercises Undo/Redo — all at
// engine level. `audioFile` is a real recording when present (first
// scored manifest entry); the synthetic 440 Hz leg always runs, so the
// workflow is covered even when no dataset is present.
void testImportedWorkflowForFile (const juce::File& audioFile, const std::string& label)
{
    AudioFileLoader loader;
    const AudioFileLoader::Result loaded = loader.loadFileSync (audioFile);
    CHECK (loaded.succeeded());
    CHECK (loaded.errorMessage.isEmpty());
    if (! loaded.succeeded() || loaded.clip == nullptr)
    {
        benchPrint ("BENCH REAL workflow " + label + ": LOAD FAILED");
        return;
    }
    const auto& clip = loaded.clip;
    CHECK (clip->importedFromFile);
    CHECK (clip->sourceSampleRate > 0.0);
    CHECK (clip->analysisMonoBuffer.getNumChannels() == 1);
    CHECK (clip->analysisMonoBuffer.getNumSamples() > 0);

    const std::shared_ptr<PitchAnalysisResult> analysis =
        MonophonicPitchAnalyzer::analyze (clip->analysisMonoBuffer,
                                          clip->sourceSampleRate,
                                          productionMappedAnalysisSettings());
    CHECK (analysis != nullptr);
    if (analysis == nullptr)
    {
        benchPrint ("BENCH REAL workflow " + label + ": ANALYZE FAILED");
        return;
    }
    CHECK (analysis->hasPitch());

    const std::shared_ptr<MelodyNoteSequence> notes = NoteSegmenter::segment (
        *analysis, productionMappedSegmentationSettings());
    CHECK (notes != nullptr);
    if (notes == nullptr)
    {
        benchPrint ("BENCH REAL workflow " + label + ": SEGMENT FAILED");
        return;
    }
    CHECK (! notes->empty());
    if (notes->empty())
    {
        benchPrint ("BENCH REAL workflow " + label + ": NO NOTES");
        return;
    }
    benchPrint ("BENCH REAL workflow " + label + ": loaded "
        + std::to_string (static_cast<int> (clip->analysisMonoBuffer.getNumSamples()))
        + " samples @ " + std::to_string (static_cast<int> (clip->sourceSampleRate))
        + " Hz, voicedFrames=" + std::to_string (analysis->voicedFrameCount)
        + ", notes=" + std::to_string (static_cast<int> (notes->notes.size())));

    // Curve edit: move the pitch at the middle of the first note up by
    // two semitones, re-segment, and check the note follows.
    const MelodyNote& firstNote = notes->notes[0];
    const double editCentre = 0.5 * (firstNote.startSeconds + firstNote.endSeconds);
    PitchAnalysisResult editedAnalysis = *analysis;
    NoteSegmenter::applyCurveEdit (editedAnalysis, editCentre,
                                   static_cast<float> (firstNote.midiNote + 2), 0.15);
    const std::shared_ptr<MelodyNoteSequence> editedNotes = NoteSegmenter::segment (
        editedAnalysis, productionMappedSegmentationSettings());
    bool foundRaised = false;
    for (const auto& note : editedNotes->notes)
        if (note.startSeconds <= editCentre && note.endSeconds >= editCentre)
            foundRaised = (note.midiNote == firstNote.midiNote + 2);
    CHECK (foundRaised);

    // MelodyEditor operations on a copy: MOVE (pitch applies), CUT,
    // STRETCH; the sequence must stay sorted and non-overlapping.
    MelodyNoteSequence editedSequence = *notes;
    const int originalMidi = editedSequence.notes[0].midiNote;
    CHECK (MelodyEditor::moveNote (editedSequence, 0, editedSequence.notes[0].startSeconds,
                                   originalMidi + 2, editedSequence.durationSeconds));
    CHECK (editedSequence.notes[0].midiNote == originalMidi + 2);
    const double cutAt = 0.5 * (editedSequence.notes[0].startSeconds
                              + editedSequence.notes[0].endSeconds);
    CHECK (MelodyEditor::splitNoteAt (editedSequence, 0, cutAt));
    CHECK (editedSequence.notes.size() == notes->notes.size() + 1);
    if (editedSequence.notes.size() < 2)
    {
        benchPrint ("BENCH REAL workflow " + label + ": SPLIT FAILED");
        return;
    }
    const double newEnd = editedSequence.notes[1].endSeconds - 0.05;
    CHECK (MelodyEditor::stretchNoteEdge (editedSequence, 1, false, newEnd,
                                          editedSequence.durationSeconds));
    CHECK_NEAR (editedSequence.notes[1].endSeconds, newEnd, 1e-9);
    CHECK (sequenceIsSortedAndNonOverlapping (editedSequence));
    // Metadata must match the notes after every edit.
    int expectedLowest = 127, expectedHighest = 0;
    for (const auto& note : editedSequence.notes)
    {
        expectedLowest = std::min (expectedLowest, note.midiNote);
        expectedHighest = std::max (expectedHighest, note.midiNote);
    }
    CHECK (editedSequence.lowestMidi == expectedLowest);
    CHECK (editedSequence.highestMidi == expectedHighest);

    // MIDI export round-trip: write, parse back, note-ons match notes.
    const juce::File midiFile = getRealAudioTestDir().getChildFile (
        "workflow-" + label + ".mid");
    juce::String exportError;
    CHECK (MidiExporter::writeToFile (*notes, midiFile, MidiExportSettings{}, exportError));
    CHECK (exportError.isEmpty());
    CHECK (midiFile.existsAsFile());
    {
        juce::FileInputStream input (midiFile);
        CHECK (input.openedOk());
        juce::MidiFile parsed;
        CHECK (parsed.readFrom (input));
        int noteOns = 0;
        for (int t = 0; t < parsed.getNumTracks(); ++t)
            for (int i = 0; i < parsed.getTrack (t)->getNumEvents(); ++i)
                if (parsed.getTrack (t)->getEventPointer (i)->message.isNoteOn())
                    ++noteOns;
        CHECK (noteOns == static_cast<int> (notes->notes.size()));
    }

    // Processor-level Undo/Redo on the real analysis.
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);
    CHECK (processor.commitPitchCurveEdit (*analysis));
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::AnalysisReady);
    CHECK (processor.hasMelodyNotes());
    const std::shared_ptr<const MelodyNoteSequence> processorMelody =
        processor.getMelodyNotes();
    if (processorMelody == nullptr || processorMelody->empty())
    {
        benchPrint ("BENCH REAL workflow " + label + ": NO PROCESSOR MELODY");
        processor.releaseResources();
        return;
    }
    const MelodyNoteSequence originalMelody = *processorMelody;
    MelodyNoteSequence movedMelody = originalMelody;
    CHECK (MelodyEditor::moveNote (movedMelody, 0, movedMelody.notes[0].startSeconds,
                                   movedMelody.notes[0].midiNote + 2,
                                   movedMelody.durationSeconds));
    CHECK (processor.commitMelodyEdit (movedMelody));
    CHECK (processor.getMelodyNotes()->notes[0].midiNote
           == originalMelody.notes[0].midiNote + 2);
    CHECK (processor.canUndoEdit());
    CHECK (processor.undoEdit());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote
           == originalMelody.notes[0].midiNote);
    CHECK (processor.canRedoEdit());
    CHECK (processor.redoEdit());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote
           == originalMelody.notes[0].midiNote + 2);
    processor.releaseResources();

    // Empty-stack behaviour on a fresh processor.
    MelotheonAudioProcessor freshProcessor;
    freshProcessor.prepareToPlay (44100.0, 512);
    CHECK (! freshProcessor.canUndoEdit());
    CHECK (! freshProcessor.canRedoEdit());
    CHECK (! freshProcessor.undoEdit());
    CHECK (! freshProcessor.redoEdit());
    freshProcessor.releaseResources();
}

void testRealAudioImportedWorkflow (const juce::File& realAudioFile,
                                    const juce::File& whistleFile)
{
    // Always-run synthetic leg: the workflow is covered even when no
    // dataset is present.
    const juce::File dir = getRealAudioTestDir();
    const juce::File synthWav = writeWavFile (
        dir, "workflow-synth-440.wav", makeSineTone (48000.0, 1.0, 440.0f, 0.5f), 48000.0);
    CHECK (synthWav.existsAsFile());
    testImportedWorkflowForFile (synthWav, "synth-440");

    // Real-audio leg: a sung melody when one was loaded (else the first
    // scored manifest entry).
    if (realAudioFile != juce::File{})
        testImportedWorkflowForFile (realAudioFile, "real-audio");

    // Tin-whistle smoke leg (unscored: file-level annotations only).
    if (whistleFile != juce::File{})
    {
        AudioFileLoader loader;
        const AudioFileLoader::Result loaded = loader.loadFileSync (whistleFile);
        CHECK (loaded.succeeded());
        if (loaded.succeeded() && loaded.clip != nullptr)
        {
            const std::shared_ptr<PitchAnalysisResult> analysis =
                MonophonicPitchAnalyzer::analyze (loaded.clip->analysisMonoBuffer,
                                                  loaded.clip->sourceSampleRate,
                                                  productionMappedAnalysisSettings());
            CHECK (analysis != nullptr);
            if (analysis != nullptr)
            {
                CHECK (analysis->hasPitch());
                const std::shared_ptr<MelodyNoteSequence> notes = NoteSegmenter::segment (
                    *analysis, productionMappedSegmentationSettings());
                CHECK (notes != nullptr);
                CHECK (! notes->empty());
                benchPrint ("BENCH REAL workflow whistle-smoke: notes="
                    + std::to_string (static_cast<int> (notes->notes.size()))
                    + " (unscored, file-level annotations only)");
            }
        }
    }
    else
    {
        benchPrint ("BENCH REAL workflow whistle-smoke: SKIPPED (file absent)");
    }

    benchPrint ("BENCH REAL workflow: engine-level only — these tests do NOT "
                "prove the FL Studio plugin interface works (no host, no "
                "processBlock audio, no message pump, no GUI, no SEND TO "
                "CHANNEL live MIDI-out). FL Studio testing on a real machine "
                "remains required before any claim about the plugin interface.");
}

// ---------------------------------------------------------------------------
// Entry point (outside the anonymous namespace: this is the function
// declared in RealAudioValidation.h and called from
// PitchBenchmarkTests.cpp).
// ---------------------------------------------------------------------------

} // namespace

BenchmarkTestOutcome runRealAudioValidationTests()
{
    const std::vector<juce::File> manifests = resolveManifestFiles();
    if (manifests.empty())
    {
        benchPrint ("BENCH REAL: SKIPPED — set MELOTHEON_REAL_AUDIO_MANIFEST "
                    "to a manifest.csv (or MELOTHEON_REAL_AUDIO_DIR to the "
                    "datasets root). Dataset absent, not a failure.");
        return { 0, 0 };
    }

    RealAggregate structAggregate, productionAggregate;
    juce::File firstScoredAudio;
    juce::File firstSingingAudio;
    int totalCases = 0;
    double totalAudioSeconds = 0.0;

    for (const auto& manifestFile : manifests)
    {
        CsvAnnotationDataset dataset;
        std::string error;
        if (! dataset.loadManifest (manifestFile, error))
        {
            benchPrint ("BENCH REAL manifest FAILED: "
                        + manifestFile.getFullPathName().toStdString()
                        + " — " + error);
            CHECK (false);
            continue;
        }
        benchPrint ("BENCH REAL manifest: "
                    + manifestFile.getFullPathName().toStdString()
                    + " entries=" + std::to_string (static_cast<int> (dataset.entries().size())));

        for (const auto& entry : dataset.entries())
        {
            const DatasetLoadResult loaded = dataset.loadCase (entry);
            if (! loaded.ok)
            {
                benchPrint ("BENCH REAL " + entry.audioPath + ": LOAD FAILED — "
                            + loaded.error);
                CHECK (false);
                continue;
            }
            if (firstScoredAudio == juce::File{}
                && ! loaded.testCase.referenceNotes.empty())
            {
                const juce::File manifestDir = manifestFile.getParentDirectory();
                firstScoredAudio = manifestDir.getChildFile (entry.audioPath);
            }
            if (firstSingingAudio == juce::File{} && entry.instrument == "singing"
                && ! loaded.testCase.referenceNotes.empty())
            {
                const juce::File manifestDir = manifestFile.getParentDirectory();
                firstSingingAudio = manifestDir.getChildFile (entry.audioPath);
            }
            ++totalCases;
            totalAudioSeconds += loaded.testCase.audio.getNumSamples()
                               / loaded.testCase.sampleRate;

            runRealCase (loaded.testCase, "struct-default",
                         PitchAnalysisSettings{}, NoteSegmentationSettings{},
                         structAggregate);
            runRealCase (loaded.testCase, "production",
                         productionMappedAnalysisSettings(),
                         productionMappedSegmentationSettings(),
                         productionAggregate);
        }
    }

    printRealSummary ("struct-default", structAggregate);
    printRealSummary ("production", productionAggregate);
    benchPrint ("BENCH REAL totals: cases=" + std::to_string (totalCases)
        + " audioSeconds=" + fmt (totalAudioSeconds, 1));

    testRealAudioImportedWorkflow (firstSingingAudio != juce::File{}
                                       ? firstSingingAudio : firstScoredAudio,
                                   findWhistleFile (manifests));

    getRealAudioTestDir().deleteRecursively();

    std::cout << "Real Audio Validation tests: " << (checks - failures) << "/" << checks
              << " checks passed";
    if (failures == 0)
        std::cout << " - ALL GREEN\n";
    else
        std::cout << " - " << failures << " FAILED\n";

    return { checks, failures };
}

} // namespace pitchbenchmark
