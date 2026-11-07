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
