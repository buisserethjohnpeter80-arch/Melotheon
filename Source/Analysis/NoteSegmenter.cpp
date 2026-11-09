#include "NoteSegmenter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>

namespace
{
float median (std::vector<float> values)
{
    if (values.empty())
        return 0.0f;

    const auto middle = values.begin()
                      + static_cast<std::ptrdiff_t> (values.size() / 2);
    std::nth_element (values.begin(), middle, values.end());
    float value = *middle;
    if ((values.size() % 2) == 0)
    {
        const auto lower = std::max_element (values.begin(), middle);
        value = 0.5f * (value + *lower);
    }
    return value;
}

struct Run
{
    int label = -1;
    int start = 0;
    int end = 0; // exclusive
};

std::vector<Run> makeRuns (const std::vector<int>& labels)
{
    std::vector<Run> runs;
    if (labels.empty())
        return runs;

    int start = 0;
    int label = labels.front();
    for (int i = 1; i <= static_cast<int> (labels.size()); ++i)
    {
        if (i == static_cast<int> (labels.size()) || labels[static_cast<size_t> (i)] != label)
        {
            runs.push_back ({ label, start, i });
            if (i < static_cast<int> (labels.size()))
            {
                start = i;
                label = labels[static_cast<size_t> (i)];
            }
        }
    }
    return runs;
}

// ---- Vibrato-aware consolidation (Phase 2A) ----
//
// Root cause of vibrato fragmentation: the hysteresis quantiser above
// flips the label whenever the smoothed pitch moves further than
// pitchHysteresisSemitones from the current INTEGER label. A sustained
// note with vibrato wider than the hysteresis crosses that band twice
// per vibrato cycle, so each half-cycle becomes a run long enough to
// survive blip removal and is emitted as its own MIDI note. The
// quantiser cannot tell an excursion that RETURNS to its centre
// (vibrato) from a step to a new centre (a genuine note change).
//
// The pass below looks for the signature of vibrato in the runs
// themselves: a short excursion to a neighbouring semitone, sandwiched
// between runs of the original pitch, where the continuous pitch
// trajectory over the whole span oscillates around the original centre
// (wide range, at least one direction reversal, stable median) instead
// of settling at the new pitch. Genuine note changes do not show that
// signature: their pitch is flat inside each run (narrow span range)
// and, when a new pitch is held, the tail of the span settles at it.
// Proven vibrato excursions are relabelled to the home pitch; every
// other run is left untouched, so note timing for non-vibrato material
// is identical to before. The original pitch frames are never modified:
// only the label vector (local to segment()) changes, so the vibrato
// trajectory stays available for visualisation and future pitch-bend
// support.
struct PitchRegionFeatures
{
    // Median smoothed pitch over the span, in cents (MIDI 0 = 0).
    double centreCents = 0.0;
    // Largest |pitch - home label| over the span, in cents.
    double deviationCents = 0.0;
    // Max - min smoothed pitch over the span, in cents.
    double rangeCents = 0.0;
    double durationSeconds = 0.0;
    // Significant reversals of the pitch direction (>= 25 cents from
    // the last extreme). Vibrato reverses every half period; a genuine
    // neighbour figure reverses once at most.
    int directionChanges = 0;
    // True when the tail of the span settles at the excursion pitch:
    // the pitch has found a new stable centre instead of returning.
    bool hasStableNewCenter = false;
};

// A vibrato excursion is a short slice of one oscillation, not a note:
// it lasts at most half a slow vibrato period. Genuine notes longer
// than this are never merged by the pass.
constexpr double kMaxVibratoExcursionSeconds = 0.50;
// The pitch must travel beyond BOTH labels: a genuine step between
// two flat notes spans ~100 cents + detector noise, while vibrato
// wider than the hysteresis spans at least ~150 cents after the
// 5-frame median smoothing.
constexpr double kVibratoRangeCents = 140.0;
// The oscillation must be centred on the home pitch.
constexpr double kVibratoCentreCents = 50.0;
// ...but bounded: a two-semitone step reaches 200 cents from the
// home label and must never merge.
constexpr double kVibratoMaxDeviationCents = 160.0;
// A direction change only counts after the pitch has moved this far
// from the last extreme, so detector noise never counts as
// oscillation.
constexpr double kDirectionChangeCents = 25.0;
// Tail evidence for a stable new centre: the last 100 ms sits within
// this distance of the excursion pitch with at most this much spread.
constexpr double kStableCentreCents = 25.0;
constexpr double kStableTailRangeCents = 50.0;

PitchRegionFeatures analysePitchRegion (const std::vector<float>& smoothed,
                                        const PitchAnalysisResult& analysis,
                                        int begin, int end,
                                        int homeLabel, int excursionLabel)
{
    PitchRegionFeatures features;
    const double hop = analysis.hopSeconds;
    features.durationSeconds = std::max (0.0,
        static_cast<double> (end - begin) * hop);

    std::vector<float> values;
    values.reserve (static_cast<size_t> (std::max (0, end - begin)));
    for (int i = begin; i < end; ++i)
    {
        if (i < 0 || i >= static_cast<int> (analysis.frames.size()))
            continue;
        if (! analysis.frames[static_cast<size_t> (i)].voiced)
            continue;
        values.push_back (smoothed[static_cast<size_t> (i)] * 100.0f);
    }
    if (values.empty())
        return features;

    const float homeCents = static_cast<float> (homeLabel) * 100.0f;
    features.centreCents = static_cast<double> (median (values));
    const float low = *std::min_element (values.begin(), values.end());
    const float high = *std::max_element (values.begin(), values.end());
    features.rangeCents = static_cast<double> (high - low);
    float maxDeviation = 0.0f;
    for (float value : values)
        maxDeviation = std::max (maxDeviation, std::abs (value - homeCents));
    features.deviationCents = static_cast<double> (maxDeviation);

    // Significant reversals: track the last extreme; a reversal counts
    // only after the pitch has moved at least 25 cents away from it,
    // so detector noise never counts as oscillation. The reference
    // extreme is FIXED while no direction is established yet: if it
    // followed every frame, a slow steady move (each step below the
    // deadband, e.g. a 4 Hz vibrato rising ~25 cents per frame) would
    // never register a direction at all.
    float extreme = values.front();
    int direction = 0; // +1 rising, -1 falling, 0 not established yet
    for (size_t i = 1; i < values.size(); ++i)
    {
        const float value = values[i];
        if (direction > 0)
        {
            if (value > extreme)
                extreme = value;
            else if (value < extreme - static_cast<float> (kDirectionChangeCents))
            {
                ++features.directionChanges;
                direction = -1;
                extreme = value;
            }
        }
        else if (direction < 0)
        {
            if (value < extreme)
                extreme = value;
            else if (value > extreme + static_cast<float> (kDirectionChangeCents))
            {
                ++features.directionChanges;
                direction = 1;
                extreme = value;
            }
        }
        else if (value > extreme + static_cast<float> (kDirectionChangeCents))
        {
            direction = 1;
            extreme = value;
        }
        else if (value < extreme - static_cast<float> (kDirectionChangeCents))
        {
            direction = -1;
            extreme = value;
        }
        // Otherwise the pitch has not moved far enough from the
        // reference extreme yet: leave the extreme (and the
        // direction) alone so the move can accumulate.
    }

    // Tail: the last 100 ms of the span (or the whole span if shorter).
    const int tailFrames = std::min (static_cast<int> (values.size()),
        std::max (1, static_cast<int> (std::llround (0.10 / hop))));
    std::vector<float> tail (values.end() - tailFrames, values.end());
    const float tailMedian = median (tail);
    const float tailLow = *std::min_element (tail.begin(), tail.end());
    const float tailHigh = *std::max_element (tail.begin(), tail.end());
    features.hasStableNewCenter =
        std::abs (tailMedian - static_cast<float> (excursionLabel) * 100.0f)
            <= static_cast<float> (kStableCentreCents)
        && (tailHigh - tailLow) <= static_cast<float> (kStableTailRangeCents);

    return features;
}

// Whether the analysed region looks like vibrato rather than a
// genuine note change. A region is vibrato-like when its pitch
// range is wide enough to cross a semitone boundary (> 140 cents),
// its centre stays close to the home label (within 50 cents), its
// furthest excursion stays within 160 cents of home, and it
// contains at least one direction reversal (a genuine oscillation
// always reverses; a flat or monotone span does not). The range
// test is the decisive guard: a genuine step between two flat
// notes spans ~100 cents and never reaches 140, while vibrato
// wide enough to fragment spans ~150-175 cents.
bool isLikelyVibrato (const PitchRegionFeatures& features,
                      int homeLabel)
{
    const double homeCents = static_cast<double> (homeLabel) * 100.0;
    return features.rangeCents > kVibratoRangeCents
        && std::abs (features.centreCents - homeCents) <= kVibratoCentreCents
        && features.deviationCents <= kVibratoMaxDeviationCents
        && features.directionChanges >= 1;
}

bool isConfirmedNoteTransition (const PitchRegionFeatures& features)
{
    return features.hasStableNewCenter;
}

void mergeVibratoExcursions (std::vector<int>& labels,
                             const std::vector<float>& smoothed,
                             const PitchAnalysisResult& analysis)
{
    const double hop = analysis.hopSeconds;
    if (hop <= 0.0 || labels.empty())
        return;

    // Each merge strictly reduces the run count, so the loop terminates.
    bool changed = true;
    while (changed)
    {
        changed = false;
        auto runs = makeRuns (labels);

        // Sandwiches: a short excursion to a neighbouring semitone
        // between two runs of the original pitch.
        for (size_t i = 1; i + 1 < runs.size(); ++i)
        {
            const Run& a = runs[i - 1];
            const Run& b = runs[i];
            const Run& c = runs[i + 1];
            if (a.label < 0 || b.label < 0 || c.label < 0)
                continue;
            if (a.label != c.label || std::abs (b.label - a.label) != 1)
                continue;
            if (static_cast<double> (b.end - b.start) * hop
                    > kMaxVibratoExcursionSeconds)
                continue;

            const PitchRegionFeatures features = analysePitchRegion (
                smoothed, analysis, a.start, c.end, a.label, b.label);
            // The range test is the decisive guard for a sandwich: a
            // genuine step between two flat notes spans ~100 cents, so
            // it never reaches the 140-cent bar, while vibrato wide
            // enough to fragment spans ~150-175 cents. The direction
            // test only rejects degenerate flat spans (a sandwich
            // always contains at least one reversal by construction:
            // the pitch leaves home and returns).
            if (isLikelyVibrato (features, a.label)
                && ! isConfirmedNoteTransition (features))
            {
                std::fill (labels.begin() + b.start, labels.begin() + b.end, a.label);
                changed = true;
                break; // runs are stale; recompute
            }
        }
        if (changed)
            continue;

        // Edges: a leading or trailing excursion with no return run
        // (the clip starts or ends mid-vibrato). Same evidence, one
        // direction change is enough for a two-run span.
        if (runs.size() >= 2)
        {
            const Run& first = runs[0];
            const Run& second = runs[1];
            if (first.label >= 0 && second.label >= 0
                && std::abs (first.label - second.label) == 1
                && static_cast<double> (first.end - first.start) * hop
                       <= kMaxVibratoExcursionSeconds)
            {
                const PitchRegionFeatures features = analysePitchRegion (
                    smoothed, analysis, first.start, second.end,
                    second.label, first.label);
                if (isLikelyVibrato (features, second.label)
                    && ! isConfirmedNoteTransition (features))
                {
                    std::fill (labels.begin() + first.start,
                               labels.begin() + first.end, second.label);
                    changed = true;
                }
            }

            if (! changed)
            {
                const Run& last = runs.back();
                const Run& previous = runs[runs.size() - 2];
                if (last.label >= 0 && previous.label >= 0
                    && std::abs (last.label - previous.label) == 1
                    && static_cast<double> (last.end - last.start) * hop
                           <= kMaxVibratoExcursionSeconds)
                {
                    const PitchRegionFeatures features = analysePitchRegion (
                        smoothed, analysis, previous.start, last.end,
                        previous.label, last.label);
                    if (isLikelyVibrato (features, previous.label)
                        && ! isConfirmedNoteTransition (features))
                    {
                        std::fill (labels.begin() + last.start,
                                   labels.begin() + last.end, previous.label);
                        changed = true;
                    }
                }
            }
        }
    }
}
} // namespace

std::shared_ptr<MelodyNoteSequence> NoteSegmenter::segment (
    const PitchAnalysisResult& analysis,
    const NoteSegmentationSettings& requestedSettings)
{
    auto sequence = std::make_shared<MelodyNoteSequence>();
    sequence->durationSeconds = std::max (0.0, analysis.durationSeconds);

    if (analysis.frames.empty() || analysis.hopSeconds <= 0.0)
        return sequence;

    NoteSegmentationSettings settings = requestedSettings;
    settings.minimumNoteSeconds = std::clamp (settings.minimumNoteSeconds, 0.02, 0.50);
    settings.maximumGapSeconds = std::clamp (settings.maximumGapSeconds, 0.0, 0.20);
    settings.pitchHysteresisSemitones = std::clamp (
        settings.pitchHysteresisSemitones, 0.50f, 1.25f);

    const int frameCount = static_cast<int> (analysis.frames.size());
    const int minimumFrames = std::max (1, static_cast<int> (
        std::ceil (settings.minimumNoteSeconds / analysis.hopSeconds)));
    const int maximumGapFrames = std::max (0, static_cast<int> (
        std::llround (settings.maximumGapSeconds / analysis.hopSeconds)));

    // Five-frame median smoothing removes isolated boundary wobble while
    // leaving note attacks and intentional bends visible.
    std::vector<float> smoothed (analysis.frames.size(), 0.0f);
    for (int i = 0; i < frameCount; ++i)
    {
        if (! analysis.frames[static_cast<size_t> (i)].voiced)
            continue;

        std::vector<float> local;
        local.reserve (5);
        for (int offset = -2; offset <= 2; ++offset)
        {
            const int index = i + offset;
            if (index >= 0 && index < frameCount
                && analysis.frames[static_cast<size_t> (index)].voiced)
                local.push_back (analysis.frames[static_cast<size_t> (index)].midiPitch);
        }
        smoothed[static_cast<size_t> (i)] = median (std::move (local));
    }

    // Quantise with hysteresis. The label changes only after pitch moves far
    // enough from the current semitone centre, preventing normal vibrato from
    // producing alternating MIDI notes.
    std::vector<int> labels (analysis.frames.size(), -1);
    int currentLabel = -1;
    for (int i = 0; i < frameCount; ++i)
    {
        const auto& frame = analysis.frames[static_cast<size_t> (i)];
        if (! frame.voiced)
            continue;

        const float pitch = smoothed[static_cast<size_t> (i)];
        if (currentLabel < 0
            || std::abs (pitch - static_cast<float> (currentLabel))
               > settings.pitchHysteresisSemitones)
            currentLabel = std::clamp (static_cast<int> (std::lround (pitch)), 0, 127);

        labels[static_cast<size_t> (i)] = currentLabel;
    }

    // Bridge only short gaps whose pitches agree on both sides.
    auto runs = makeRuns (labels);
    for (size_t i = 1; i + 1 < runs.size(); ++i)
    {
        const Run& gap = runs[i];
        if (gap.label != -1 || gap.end - gap.start > maximumGapFrames)
            continue;
        const int left = runs[i - 1].label;
        const int right = runs[i + 1].label;
        if (left >= 0 && left == right)
            std::fill (labels.begin() + gap.start, labels.begin() + gap.end, left);
    }

    // Vibrato consolidation (Phase 2A): merge short excursions to a
    // neighbouring semitone back into their note when the pitch
    // trajectory over the span oscillates around the original centre
    // instead of settling at the new pitch. Genuine note changes are
    // untouched: their pitch is flat inside each run and their tail
    // settles at the new centre.
    mergeVibratoExcursions (labels, smoothed, analysis);

    // Remove detector blips shorter than the commercial minimum. If both
    // neighbours agree, fold the blip into them. Otherwise use the closest
    // adjacent pitch when it is only one semitone away; isolated blips become
    // silence. Two passes also collapse runs created by the first pass.
    for (int pass = 0; pass < 2; ++pass)
    {
        runs = makeRuns (labels);
        for (size_t i = 0; i < runs.size(); ++i)
        {
            const Run& run = runs[i];
            if (run.label < 0 || run.end - run.start >= minimumFrames)
                continue;

            const int left = i > 0 ? runs[i - 1].label : -1;
            const int right = i + 1 < runs.size() ? runs[i + 1].label : -1;
            int replacement = -1;
            if (left >= 0 && left == right)
                replacement = left;
            else if (left >= 0 && std::abs (left - run.label) <= 1)
                replacement = left;
            else if (right >= 0 && std::abs (right - run.label) <= 1)
                replacement = right;

            std::fill (labels.begin() + run.start, labels.begin() + run.end, replacement);
        }
    }

    runs = makeRuns (labels);
    sequence->lowestMidi = 127;
    sequence->highestMidi = 0;

    for (const Run& run : runs)
    {
        if (run.label < 0 || run.end - run.start < minimumFrames)
            continue;

        std::vector<float> pitches;
        std::vector<float> confidences;
        std::vector<float> rmsValues;
        pitches.reserve (static_cast<size_t> (run.end - run.start));
        confidences.reserve (pitches.capacity());
        rmsValues.reserve (pitches.capacity());

        for (int frameIndex = run.start; frameIndex < run.end; ++frameIndex)
        {
            const auto& frame = analysis.frames[static_cast<size_t> (frameIndex)];
            if (! frame.voiced)
                continue;
            pitches.push_back (frame.midiPitch);
            confidences.push_back (frame.confidence);
            rmsValues.push_back (frame.rmsDb);
        }
        if (pitches.empty())
            continue;

        MelodyNote note;
        note.startSeconds = std::max (0.0,
            analysis.frames[static_cast<size_t> (run.start)].timeSeconds);
        const int lastFrame = std::min (run.end - 1, frameCount - 1);
        note.endSeconds = std::min (
            sequence->durationSeconds,
            analysis.frames[static_cast<size_t> (lastFrame)].timeSeconds
                + analysis.hopSeconds);
        note.midiNote = std::clamp (run.label, 0, 127);
        note.sourceMidiPitch = median (std::move (pitches));
        note.confidence = std::clamp (median (std::move (confidences)), 0.0f, 1.0f);
        const float rmsDb = median (std::move (rmsValues));
        note.velocity = std::clamp ((rmsDb + 54.0f) / 48.0f, 0.25f, 1.0f);

        if (note.durationSeconds() < settings.minimumNoteSeconds * 0.95)
            continue;

        sequence->lowestMidi = std::min (sequence->lowestMidi, note.midiNote);
        sequence->highestMidi = std::max (sequence->highestMidi, note.midiNote);
        sequence->notes.push_back (note);
    }

    if (sequence->notes.empty())
    {
        sequence->lowestMidi = 60;
        sequence->highestMidi = 60;
    }

    return sequence;
}

void NoteSegmenter::applyCurveEdit (PitchAnalysisResult& analysis,
                                    double centreSeconds,
                                    float targetMidi,
                                    double windowSeconds)
{
    if (analysis.frames.empty() || analysis.hopSeconds <= 0.0
        || windowSeconds <= 0.0)
        return;

    const float clampedTarget = std::clamp (targetMidi, 24.0f, 108.0f);
    const int frameCount = static_cast<int> (analysis.frames.size());
    const int centreIndex = static_cast<int> (std::llround (
        centreSeconds / analysis.hopSeconds));
    const int radiusFrames = static_cast<int> (std::ceil (
        windowSeconds / analysis.hopSeconds));
    const int first = std::max (0, centreIndex - radiusFrames);
    const int last = std::min (frameCount - 1, centreIndex + radiusFrames);
    if (first > last || centreIndex < 0 || centreIndex >= frameCount)
        return;

    // Reference pitch: the voiced frame nearest the grab point. The
    // edit is a delta from this frame, so dragging horizontally at a
    // constant height is a no-op instead of accumulating.
    const PitchFrame* centreFrame = nullptr;
    for (int distance = 0; distance <= radiusFrames; ++distance)
    {
        const int left = centreIndex - distance;
        if (left >= 0 && analysis.frames[static_cast<size_t> (left)].voiced)
        {
            centreFrame = &analysis.frames[static_cast<size_t> (left)];
            break;
        }
        const int right = centreIndex + distance;
        if (distance != 0 && right < frameCount
            && analysis.frames[static_cast<size_t> (right)].voiced)
        {
            centreFrame = &analysis.frames[static_cast<size_t> (right)];
            break;
        }
    }
    if (centreFrame == nullptr)
        return;

    const float delta = clampedTarget - centreFrame->midiPitch;
    if (std::abs (delta) < 0.001f)
        return;

    for (int i = first; i <= last; ++i)
    {
        PitchFrame& frame = analysis.frames[static_cast<size_t> (i)];
        if (! frame.voiced)
            continue;

        const double offsetSeconds = frame.timeSeconds - centreSeconds;
        const double t = offsetSeconds / windowSeconds; // -1..1
        // Raised-cosine falloff: 1 at the grab point, 0 at the edges.
        const double weight = 0.5 * (1.0 + std::cos (std::numbers::pi * t));
        if (weight <= 0.0)
            continue;

        frame.midiPitch = std::clamp (
            frame.midiPitch + static_cast<float> (delta * weight),
            24.0f, 108.0f);
        frame.frequencyHz = static_cast<float> (
            440.0 * std::pow (2.0, (static_cast<double> (frame.midiPitch) - 69.0) / 12.0));
    }

    // Keep the range metadata honest so FIT still frames the curve.
    float low = std::numeric_limits<float>::max();
    float high = std::numeric_limits<float>::lowest();
    for (const auto& frame : analysis.frames)
    {
        if (! frame.voiced)
            continue;
        low = std::min (low, frame.midiPitch);
        high = std::max (high, frame.midiPitch);
    }
    if (low <= high)
    {
        analysis.minMidi = low;
        analysis.maxMidi = high;
    }
}
