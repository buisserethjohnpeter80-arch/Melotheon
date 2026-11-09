// MELOTHEON Phase 1 — Professional Pitch Detection Benchmark.
//
// Measurement only. This file (and PitchBenchmark.cpp /
// PitchBenchmarkTests.cpp) contains NO production DSP: it calls the real
// production analysis code — MonophonicPitchAnalyzer::analyze(),
// NoteSegmenter::segment(), MidiExporter::createMidiFile() — and measures
// what it does. No YIN replacement, no threshold changes, no UI changes.
//
// Three separate levels, never one combined score:
//   Level A — Raw Pitch Detection: analyze() frames vs a reference
//             trajectory (cents error, RPA, RCA, octave errors).
//   Level B — Complete Melody Analysis: analyze() + segment() notes vs
//             reference notes (precision/recall/F1, timing errors).
//   Level C — MIDI Generation: createMidiFile() events vs the reference
//             notes (note on/off integrity, pitch, channel, timing).
// An error is attributed to the stage that produced it; a Level B miss
// caused by Level A is reported as a detection failure, not a
// segmentation failure, and vice versa (see attributeShortNoteFailure).
//
// CONFIDENCE — what it is and what it is not:
//   Production confidence is clamp(1 - CMND(candidate lag), 0, 1), a YIN
//   difference-derived value: 1.0 means "perfectly periodic at the chosen
//   lag". It is NOT a 0-1 probability, NOT a calibrated likelihood, and
//   must never be silently converted into one. This benchmark reports it
//   as produced and never re-interprets it.
//
// OPERATING RANGES (documented, from the production defaults):
//   Frequency: 55-2500 Hz (validity window 52.25-2625 Hz). Whistles above
//     ~2625 Hz are rejected by design, not by failure.
//   Minimum note: 0.06 s (struct default) / 0.0775 s (production-mapped
//     TRANSITION knob at 0.5). Notes shorter than the minimum are folded
//     or deleted by design; a deliberately unsupported short note is
//     reported as outside the documented operating range, NOT as a
//     crash or a test failure.
//   Hop: 10 ms frames; note starts/ends are quantised to that grid, and
//     the segmenter's 5-frame median can shift boundaries by up to 20 ms.
//
// All synthetic signals are deterministic: fixed seeds, no wall-clock or
// unseeded randomness anywhere. Offline tests only — nothing here runs in
// the real-time audio callback.

#pragma once

#include <JuceHeader.h>

#include "Analysis/MelodyNotes.h"
#include "Analysis/PitchAnalysis.h"
#include "Midi/MidiExporter.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace pitchbenchmark
{

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// A frame counts as "in tune" when its error is within +/-50 cents.
constexpr double kCentsTolerance = 50.0;
// One octave in cents.
constexpr double kOctaveCents = 1200.0;
// Octave classification window: an error is an octave error when it lies
// within +/-50 cents of a whole number of octaves (1200/2400/3600).
constexpr double kOctaveWindowCents = 50.0;
// Default analysis hop (10 ms), matching PitchAnalysisSettings.
constexpr double kDefaultHopSeconds = 0.01;

// ---------------------------------------------------------------------------
// Part 4 — data structures
// ---------------------------------------------------------------------------

// One frame of ground truth. frequencyHz is the instantaneous reference
// pitch at timeSeconds; voiced=false marks silence / aperiodic frames.
// For vibrato the reference is the instantaneous pitch trajectory, NOT
// the nominal note; for glissando it is the continuous slide.
struct ReferencePitchFrame
{
    double timeSeconds = 0.0;
    double frequencyHz = 0.0;
    bool voiced = false;
};

struct ReferenceNote
{
    double startSeconds = 0.0;
    double endSeconds = 0.0;
    int midiNote = 60;
    double frequencyHz = 440.0;

    double durationSeconds() const { return std::max (0.0, endSeconds - startSeconds); }
};

// One aligned observation: what the detector reported for one reference
// frame. observedVoiced=false means the analyzer left the frame unvoiced
// (frequencyHz/midiPitch/confidence are then 0, exactly as production
// leaves them).
struct PitchObservation
{
    double timeSeconds = 0.0;
    double referenceFrequencyHz = 0.0;
    bool referenceVoiced = false;
    float observedFrequencyHz = 0.0f;
    float observedMidiPitch = 0.0f;
    float observedConfidence = 0.0f;
    bool observedVoiced = false;
};

// A complete test case: audio + ground truth. Synthetic cases carry
// datasetVersion "synthetic-v1"; real-audio cases carry the dataset's own
// version string (see CsvAnnotationDataset).
struct AudioTestCase
{
    std::string name;
    juce::AudioBuffer<float> audio;
    double sampleRate = 48000.0;
    std::vector<ReferencePitchFrame> referenceFrames;
    std::vector<ReferenceNote> referenceNotes;
    std::string datasetVersion = "synthetic-v1";
    std::string sourceDescription;
};

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

inline double frequencyToMidiDouble (double frequencyHz)
{
    return 69.0 + 12.0 * std::log2 (frequencyHz / 440.0);
}

inline double midiToFrequencyDouble (double midiNote)
{
    return 440.0 * std::pow (2.0, (midiNote - 69.0) / 12.0);
}

// Signed error of the observed pitch against the reference, in cents.
// Positive = observed sharp. Requires both frequencies > 0.
inline double centsError (double observedHz, double referenceHz)
{
    return 1200.0 * std::log2 (observedHz / referenceHz);
}

// Chroma error: the cents error wrapped into [-600, +600] so that an
// octave error scores as 0 chroma error. Reported SEPARATELY from the
// raw cents error — a perfect chroma score must never hide octave errors.
inline double chromaCentsError (double observedHz, double referenceHz)
{
    double cents = centsError (observedHz, referenceHz);
    while (cents > 600.0)
        cents -= 1200.0;
    while (cents < -600.0)
        cents += 1200.0;
    return cents;
}

// ---------------------------------------------------------------------------
// Part 6 — octave classification (benchmark-only; no production correction)
// ---------------------------------------------------------------------------

struct OctaveClassification
{
    bool isOctaveError = false;
    int octaves = 0; // 1, 2 or 3 when isOctaveError
    bool up = false; // true = observed sharp by N octaves
};

// Classifies a signed cents error. An error is an octave error when its
// magnitude lies within +/-kOctaveWindowCents of 1200, 2400 or 3600 cents.
// Boundary is inclusive: +1150 cents counts as "1 octave up", -1150 as
// "1 octave down". Anything else (including 0 and +/-50 cents) is not an
// octave error.
OctaveClassification classifyOctaveError (double cents);

// ---------------------------------------------------------------------------
// Part 5 — Level A pitch-accuracy metrics
// ---------------------------------------------------------------------------

struct OctaveErrorBreakdown
{
    int up1 = 0, down1 = 0;
    int up2 = 0, down2 = 0;
    int up3 = 0, down3 = 0;
    int total = 0;
};

struct CentsErrorStats
{
    // Denominator discipline (Part 5): the denominator for RPA, RCA and
    // the octave error rate is ALL voiced reference frames. Failed or
    // unvoiced predictions are NOT excluded — a missed frame counts as a
    // wrong frame. Conditional statistics on detected-voiced frames only
    // are reported separately, under the "conditional" fields.
    int voicedReferenceFrames = 0;
    int detectedVoicedFrames = 0;
    int missedVoicedFrames = 0;

    // Mean / median absolute cents over DETECTED voiced frames only.
    // NaN (hasConditionalStats=false) when no voiced frame was detected.
    double meanAbsCents = std::numeric_limits<double>::quiet_NaN();
    double medianAbsCents = std::numeric_limits<double>::quiet_NaN();
    bool hasConditionalStats = false;

    // Raw Pitch Accuracy: detected-voiced frames within +/-50 cents,
    // divided by ALL voiced reference frames.
    double rawPitchAccuracy = 0.0;
    // Raw Chroma Accuracy: same, with the error wrapped to +/-600 cents.
    // Reported separately; a high RCA with a low RPA means octave errors.
    double rawChromaAccuracy = 0.0;
    // Octave-classified frames divided by ALL voiced reference frames.
    double octaveErrorRate = 0.0;
    OctaveErrorBreakdown octaveBreakdown;

    // Accuracy among detected-voiced frames only (conditional RPA).
    double conditionalRawPitchAccuracy = std::numeric_limits<double>::quiet_NaN();
};

// ---------------------------------------------------------------------------
// Part 7 — Level B note-matching metrics
// ---------------------------------------------------------------------------

struct NoteMatchResult
{
    int matched = 0;
    int missed = 0;         // reference notes with no detected match
    int falsePositives = 0; // detected notes with no reference match
    // Precision is undefined when nothing was detected (matched +
    // falsePositives == 0); recall is undefined when there is no
    // reference note (matched + missed == 0); F1 is undefined when
    // either is undefined or precision + recall == 0. Undefined is
    // reported as NaN, never as 1.0.
    double precision = std::numeric_limits<double>::quiet_NaN();
    double recall = std::numeric_limits<double>::quiet_NaN();
    double f1 = std::numeric_limits<double>::quiet_NaN();
    bool precisionDefined = false;
    bool recallDefined = false;
    bool f1Defined = false;

    // Timing errors over MATCHED pairs only, in milliseconds.
    // NaN when nothing matched.
    double meanAbsOnsetErrorMs = std::numeric_limits<double>::quiet_NaN();
    double medianAbsOnsetErrorMs = std::numeric_limits<double>::quiet_NaN();
    double meanAbsOffsetErrorMs = std::numeric_limits<double>::quiet_NaN();
    double medianAbsOffsetErrorMs = std::numeric_limits<double>::quiet_NaN();
    double meanAbsDurationErrorMs = std::numeric_limits<double>::quiet_NaN();
    double medianAbsDurationErrorMs = std::numeric_limits<double>::quiet_NaN();
};

// ---------------------------------------------------------------------------
// Part 9 — voicing metrics
// ---------------------------------------------------------------------------

struct VoicingConfusion
{
    int truePositives = 0;  // reference voiced, detected voiced
    int falsePositives = 0; // reference unvoiced, detected voiced
    int trueNegatives = 0;  // reference unvoiced, detected unvoiced
    int falseNegatives = 0; // reference voiced, detected unvoiced

    // Precision/recall/F1 are NaN with the matching flag false when their
    // denominator is zero. An undefined metric is reported as UNDEFINED,
    // never as 1.0 ("perfect") — in particular a test case with no voiced
    // reference frames has undefined recall, and a case with no detected
    // voiced frames has undefined precision.
    double precision = std::numeric_limits<double>::quiet_NaN();
    double recall = std::numeric_limits<double>::quiet_NaN();
    double f1 = std::numeric_limits<double>::quiet_NaN();
    bool precisionDefined = false;
    bool recallDefined = false;
    bool f1Defined = false;
};

// ---------------------------------------------------------------------------
// Parts 13-14 — performance metrics
// ---------------------------------------------------------------------------

struct TimingSummary
{
    double meanMs = 0.0;
    double medianMs = 0.0;
    double p95Ms = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    int iterations = 0;
    double audioDurationSeconds = 0.0;
    // Real-Time Factor = audio duration / median stage time. > 1 means
    // faster than real time. Computed from the MEDIAN, not the mean.
    double realTimeFactor = 0.0;
};

// Per-stage timings for one analysis pass. Stages are measured at the
// granularity the public API allows WITHOUT instrumenting production code:
//   loadMs     — dataset/audio loading (0 for in-memory synthetic cases)
//   analyzeMs  — MonophonicPitchAnalyzer::analyze() as a whole, which
//                internally resamples, high-passes, runs dual-resolution
//                YIN and stabilises the curve. Resample / YIN /
//                stabilise cannot be timed separately without touching
//                production code, so they are reported as one stage.
//   segmentMs  — NoteSegmenter::segment()
//   midiMs     — MidiExporter::createMidiFile()
// No timing instrumentation exists inside processBlock().
struct StageTimings
{
    TimingSummary load;
    TimingSummary analyze;
    TimingSummary segment;
    TimingSummary midi;
    TimingSummary total; // the full analyze + segment + midi pipeline
                         // timed in ONE call (not a sum of the medians),
                         // load excluded
};

// ---------------------------------------------------------------------------
// Part 15 — reproducibility record
// ---------------------------------------------------------------------------

struct BenchmarkEnvironment
{
    // Commit SHA is NOT embedded at compile time: the CMake change for
    // this phase adds test sources only, so no -D flag carries the SHA.
    // The agent-written report records the exact commit SHA; this field
    // stays empty in-process and is printed as "see report".
    std::string commitSha;
    std::string buildConfig; // "Release" / "Debug" (from JUCE_DEBUG)
    std::string compiler;    // e.g. "GCC 13.2.0" / "MSVC 19.4x"
    std::string operatingSystem; // e.g. "Linux" (+ JUCE target)
    std::string cpuModel;        // from /proc/cpuinfo when available
    std::string timestampIso;    // local time, YYYY-MM-DDTHH:MM:SS
    std::string datasetVersion;
    double sampleRate = 0.0;
    int bufferSize = 0; // host buffer size is irrelevant to offline
                        // analysis; recorded as 0 / not-applicable.
    std::string analysisConfigSummary;
    std::string segmentationConfigSummary;

    std::string toString() const;
};

BenchmarkEnvironment collectEnvironment (
    double sampleRate,
    const std::string& datasetVersion,
    const PitchAnalysisSettings& analysisSettings,
    const NoteSegmentationSettings& segmentationSettings);

// ---------------------------------------------------------------------------
// Settings helpers — struct defaults vs production-mapped settings
// ---------------------------------------------------------------------------

// Production-mapped settings at the knob defaults, derived from
// PluginProcessor::startPitchAnalysis / makeNoteSegmentationSettings:
//   noiseGate 0.3  -> jmap(0.3, -72, -35)   = -60.9 dB
//   sensitivity 0.65 -> jmap(0.65, 0.72, 0.42) = 0.525
//   transition 0.5 -> jmap(0.5, 0.035, 0.120) = 0.0775 s
//   variation 0.5  -> jmap(0.5, 0.58, 0.92)   = 0.75 st
// Tests use BOTH the struct defaults and these production-mapped values
// where relevant, and label which one each number came from.
PitchAnalysisSettings productionMappedAnalysisSettings();
NoteSegmentationSettings productionMappedSegmentationSettings();

// ---------------------------------------------------------------------------
// Part 10 — deterministic synthetic generation (fixed seeds only)
// ---------------------------------------------------------------------------

// Pure sine, phase starting at 0. amplitude is linear (0.5 = -6 dBFS peak).
juce::AudioBuffer<float> makeSineTone (double sampleRate,
                                       double durationSeconds,
                                       double frequencyHz,
                                       float amplitude = 0.5f);

// Fundamental + harmonics. harmonicAmplitudes[0] is the fundamental;
// [1] the 2nd harmonic, etc. Harmonics stay phase-locked to the
// fundamental (no independent phases), so the result is deterministic.
juce::AudioBuffer<float> makeHarmonicTone (
    double sampleRate,
    double durationSeconds,
    double frequencyHz,
    const std::vector<float>& harmonicAmplitudes);

// Fundamental + INHARMONIC partials (guitar/piano-like): partial h sits
// at h * frequencyHz * (1 + stretchPerPartial * (h - 1)), i.e. each
// partial is stretched sharp by stretchPerPartial per partial index.
// Each partial accumulates its own phase, so the result is deterministic.
// stretchPerPartial = 0.003 means the 2nd partial is +0.3% sharp, the
// 3rd +0.6%, and so on.
juce::AudioBuffer<float> makeInharmonicTone (
    double sampleRate,
    double durationSeconds,
    double frequencyHz,
    double stretchPerPartial,
    const std::vector<float>& harmonicAmplitudes);

// Sine with sinusoidal vibrato. The reference trajectory is the
// INSTANTANEOUS pitch f(t) = centre * 2^(depth*sin(2*pi*rate*t)/12),
// returned by vibratoReferenceFrames(); the audio is generated by phase
// accumulation of that same trajectory, so audio and reference agree.
juce::AudioBuffer<float> makeVibratoTone (double sampleRate,
                                          double durationSeconds,
                                          double centreFrequencyHz,
                                          double depthSemitones,
                                          double rateHz,
                                          float amplitude = 0.5f);
std::vector<ReferencePitchFrame> vibratoReferenceFrames (
    double durationSeconds,
    double hopSeconds,
    double centreFrequencyHz,
    double depthSemitones,
    double rateHz);

// Linear slide in Hz from startFrequencyHz to endFrequencyHz over the
// full duration, generated by phase accumulation. The reference is the
// same continuous function sampled at each frame centre.
juce::AudioBuffer<float> makeGlissandoTone (double sampleRate,
                                            double durationSeconds,
                                            double startFrequencyHz,
                                            double endFrequencyHz,
                                            float amplitude = 0.5f);
std::vector<ReferencePitchFrame> glissandoReferenceFrames (
    double durationSeconds,
    double hopSeconds,
    double startFrequencyHz,
    double endFrequencyHz);

// One note of a sequence. Phase resets at each note start; a short
// raised-cosine attack/release (default 5 ms) avoids click transients
// that would otherwise contaminate the pitch measurement.
struct SequenceNoteSpec
{
    double frequencyHz = 440.0;
    double startSeconds = 0.0;
    double durationSeconds = 0.5;
    float amplitude = 0.5f;
};

// Back-to-back or gapped note sequence (silence where no note covers the
// time). Also builds the matching reference frames and reference notes.
AudioTestCase makeNoteSequenceCase (const std::string& name,
                                    double sampleRate,
                                    double totalDurationSeconds,
                                    const std::vector<SequenceNoteSpec>& notes,
                                    double attackSeconds = 0.005,
                                    double releaseSeconds = 0.005);

juce::AudioBuffer<float> makeSilence (double sampleRate, double durationSeconds);

// Deterministic white noise, uniform in [-amplitude, +amplitude],
// std::mt19937 seeded with `seed`. Same seed => identical samples.
juce::AudioBuffer<float> makeWhiteNoise (double sampleRate,
                                         double durationSeconds,
                                         float amplitude,
                                         std::uint32_t seed = 12345);

// Deterministic pink noise (Paul Kellet filter, 1/f spectrum), same
// seeded generator as makeWhiteNoise. The output is normalised so its
// measured RMS equals `amplitude` exactly.
juce::AudioBuffer<float> makePinkNoise (double sampleRate,
                                        double durationSeconds,
                                        float amplitude,
                                        std::uint32_t seed = 12345);

// Breath-like noise: white noise through a one-pole low-pass (~800 Hz)
// with a slow 0.15 Hz amplitude swell. This is SYNTHETIC breath-like
// noise for voicing tests, not a recording of real breath — real breath
// behaviour can only be measured on real audio (see the dataset
// interface below).
juce::AudioBuffer<float> makeBreathLikeNoise (double sampleRate,
                                              double durationSeconds,
                                              float amplitude,
                                              std::uint32_t seed = 12345);

// A single short broadband burst (rectangular-windowed noise) at
// burstStartSeconds, `durationSeconds` long. Models a pluck/attack
// transient or a click; the reference is unvoiced during the burst.
juce::AudioBuffer<float> makeTransientBurst (double sampleRate,
                                             double totalDurationSeconds,
                                             double burstStartSeconds,
                                             double burstDurationSeconds,
                                             float amplitude,
                                             std::uint32_t seed = 12345);

// Build a reference frame grid over [0, duration) at the given hop.
// frequencyAtTime(t) gives the instantaneous reference frequency;
// voicedAtTime(t) decides voicing. Frames are centred at i*hop.
std::vector<ReferencePitchFrame> buildReferenceFrames (
    double durationSeconds,
    double hopSeconds,
    const std::function<double (double)>& frequencyAtTime,
    const std::function<bool (double)>& voicedAtTime);

// Mix a deterministic noise bed under a tone at a target SNR (dB).
// signalRmsDb / noiseRmsDb are measured, not assumed: the noise is
// scaled so that 20*log10(rms(signal)/rms(noise)) == snrDb.
juce::AudioBuffer<float> mixToneWithNoise (const juce::AudioBuffer<float>& tone,
                                           double snrDb,
                                           std::uint32_t seed = 12345);

// ---------------------------------------------------------------------------
// Level runners — thin wrappers that call the real production code
// ---------------------------------------------------------------------------

struct LevelAResult
{
    std::shared_ptr<PitchAnalysisResult> analysis;
    std::vector<PitchObservation> observations;
    CentsErrorStats pitchStats;
    VoicingConfusion voicing;
};

// Level A: analyze() only. `settings` is passed through unchanged.
LevelAResult runLevelA (const AudioTestCase& testCase,
                        const PitchAnalysisSettings& settings = {});

struct LevelBResult
{
    LevelAResult levelA;
    std::shared_ptr<MelodyNoteSequence> notes;
    NoteMatchResult noteMatch;
};

// Level B: analyze() + segment(). `segmentationSettings` is passed
// through unchanged.
LevelBResult runLevelB (const AudioTestCase& testCase,
                        const PitchAnalysisSettings& analysisSettings = {},
                        const NoteSegmentationSettings& segmentationSettings = {});

struct MidiIntegrityReport
{
    int noteOnCount = 0;
    int noteOffCount = 0;
    int pitchMismatches = 0;   // note-on pitch != expected reference pitch
    int channelMismatches = 0;
    int velocityOutOfRange = 0; // outside the documented 1..126
    int timingErrorsTicks = 0;  // events > 1 tick from the expected tick
    bool hasTempoMeta = false;
    bool hasEndOfTrack = false;
    bool balanced = false; // noteOnCount == noteOffCount == note count
};

// Level C: createMidiFile() on the Level B notes, checked back against
// the reference notes. Does NOT change MIDI generation to make tests
// pass — a mismatch here is reported, not worked around.
struct LevelCResult
{
    LevelBResult levelB;
    juce::MidiFile midiFile;
    MidiIntegrityReport integrity;
};

LevelCResult runLevelC (const AudioTestCase& testCase,
                        const PitchAnalysisSettings& analysisSettings = {},
                        const NoteSegmentationSettings& segmentationSettings = {},
                        const MidiExportSettings& midiSettings = {});

// ---------------------------------------------------------------------------
// Metric computation (pure functions over aligned observations)
// ---------------------------------------------------------------------------

// Aligns analysis frames to reference frames by time (nearest frame).
// Both grids are normally the same 10 ms hop; when they differ the
// nearest analysis frame is used and the alignment is still exact in
// time because frame times are compared, not indices.
std::vector<PitchObservation> alignFrames (
    const PitchAnalysisResult& analysis,
    const std::vector<ReferencePitchFrame>& referenceFrames);

CentsErrorStats computePitchAccuracy (const std::vector<PitchObservation>& observations);
VoicingConfusion computeVoicing (const std::vector<PitchObservation>& observations);

// One-to-one note matching (Part 7, documented rule):
//   * pitch agreement: detected midiNote == reference midiNote (exact
//     integer equality — a semitone off is a miss, not a match);
//   * onset proximity: |detected.start - reference.start| <=
//     onsetToleranceSeconds (default 50 ms: the 10 ms frame grid plus up
//     to 20 ms of median-smoothing boundary shift, with margin);
//   * greedy in reference order: each reference note takes the unmatched
//     detected note of the same pitch with the smallest onset error.
// Unmatched reference notes are MISSED; unmatched detected notes are
// FALSE POSITIVES. Timing errors are computed on matched pairs only.
NoteMatchResult matchNotes (const MelodyNoteSequence& detected,
                            const std::vector<ReferenceNote>& reference,
                            double onsetToleranceSeconds = 0.05);

// The p-th percentile of `sortedValues`, which MUST already be sorted
// ascending (the function does not sort). p is in [0, 100]. Linear
// interpolation between neighbouring ranks (nearest-rank method would
// also be defensible; the choice is documented here, not hidden).
double percentileOfSorted (const std::vector<double>& sortedValues, double percentile);

// ---------------------------------------------------------------------------
// Part 8 — short-note failure attribution
// ---------------------------------------------------------------------------

enum class ShortNoteFailureStage
{
    Detection,    // the analyzer never voiced the pitch (frames missing)
    Segmentation, // the analyzer voiced the pitch, the segmenter dropped it
    NotFailed     // the note was detected (recall for this note is 1)
};

struct ShortNoteAttribution
{
    ShortNoteFailureStage stage = ShortNoteFailureStage::NotFailed;
    double frameCoverage = 0.0; // voiced frames within +/-50 cents, as a
                                // fraction of the frames spanning the note
    int expectedFrames = 0;
    int voicedFrames = 0;
};

// Attributes one reference note's Level B miss to a stage. A note is
// "covered" when at least half of the frames spanning it are voiced and
// within +/-50 cents of the reference: then the detector saw it and the
// segmenter is responsible; otherwise the detector is responsible.
ShortNoteAttribution attributeShortNoteFailure (
    const PitchAnalysisResult& analysis,
    const ReferenceNote& referenceNote);

// ---------------------------------------------------------------------------
// Parts 11-12 — real-audio dataset interface (interface only; no real
// audio ships with the repo and none is downloaded by this code)
// ---------------------------------------------------------------------------

struct RealAudioDatasetEntry
{
    std::string audioPath;       // path to the audio file (WAV/AIFF/FLAC)
    std::string framesPath;      // CSV: timeSeconds,frequencyHz,voiced
    std::string notesPath;       // CSV: startSeconds,endSeconds,midiNote
    std::string instrument;      // singing / whistling / guitar / piano /
                                 // violin / synth
    std::string license;         // MUST name a verified license (e.g.
                                 // "CC-BY-4.0", "public domain"); an empty
                                 // or "unknown" license refuses to load.
    std::string datasetVersion;  // e.g. "mir-1k-v1.0"
};

struct DatasetLoadResult
{
    bool ok = false;
    std::string error; // empty when ok
    AudioTestCase testCase;
};

// Minimal dataset abstraction. Implementations supply entries; the loader
// turns one entry into an AudioTestCase (audio decoded to mono, reference
// frames/notes parsed from the annotation CSVs).
class IRealAudioDataset
{
public:
    virtual ~IRealAudioDataset() = default;
    virtual std::vector<RealAudioDatasetEntry> entries() const = 0;
    virtual DatasetLoadResult loadCase (const RealAudioDatasetEntry& entry) const = 0;
};

// CSV-manifest dataset. Manifest format (one entry per line, header row
// required):
//   audioPath,framesPath,notesPath,instrument,license,datasetVersion
// Paths in the manifest are resolved relative to the manifest's own
// directory. Annotation formats:
//   frames CSV: timeSeconds,frequencyHz,voiced   (voiced is 0 or 1)
//   notes CSV:  startSeconds,endSeconds,midiNote
// License rule: entries with an empty license, or a license of
// "unknown" (case-insensitive), are refused with a clear error —
// annotations must never be invented, and audio must never be pulled
// into the repo without its license being checked first.
class CsvAnnotationDataset : public IRealAudioDataset
{
public:
    // Parses the manifest. Returns false (with `error` set) when the
    // manifest cannot be read or has no valid entries.
    bool loadManifest (const juce::File& manifestFile, std::string& error);

    std::vector<RealAudioDatasetEntry> entries() const override { return entries_; }
    DatasetLoadResult loadCase (const RealAudioDatasetEntry& entry) const override;

private:
    std::vector<RealAudioDatasetEntry> entries_;
    juce::File manifestDirectory_;
};

// ---------------------------------------------------------------------------
// Part 14 — performance benchmarking (correctness is separate; these
// numbers never gate a CHECK — timing must not fail on CPU noise)
// ---------------------------------------------------------------------------

// Times `stage` (warmupIterations discarded, then measuredIterations
// timed with std::chrono::steady_clock) and returns median/P95/RTF for
// the given audio duration. The callable must be re-runnable.
TimingSummary benchmarkStage (const std::function<void()>& stage,
                              double audioDurationSeconds,
                              int warmupIterations = 1,
                              int measuredIterations = 3);

// ---------------------------------------------------------------------------
// Test entry point (implemented in PitchBenchmarkTests.cpp)
// ---------------------------------------------------------------------------

struct BenchmarkTestOutcome
{
    int checks = 0;
    int failures = 0;
};

BenchmarkTestOutcome runPitchBenchmarkTests();

} // namespace pitchbenchmark
