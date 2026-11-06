#pragma once

// Immutable Phase 3A pitch-analysis data shared between the processor
// and the editor. A frame is a 10 ms observation of one monophonic
// source. Unvoiced frames keep frequencyHz / midiPitch at zero.

#include <algorithm>
#include <cmath>
#include <vector>

struct PitchFrame
{
    double timeSeconds = 0.0;
    float frequencyHz = 0.0f;
    float midiPitch = 0.0f;
    float confidence = 0.0f;
    float rmsDb = -100.0f;
    bool voiced = false;
};

struct PitchAnalysisResult
{
    std::vector<PitchFrame> frames;
    double durationSeconds = 0.0;
    double analysisSampleRate = 24000.0;
    double hopSeconds = 0.01;
    int voicedFrameCount = 0;
    float minMidi = 0.0f;
    float maxMidi = 0.0f;
    float medianConfidence = 0.0f;
    float medianFrequencyHz = 0.0f;

    bool hasPitch() const { return voicedFrameCount > 0; }

    // Find a voiced frame near a playback position. Returning nullptr
    // during a real silence keeps the header from displaying a stale
    // pitch from another part of the clip.
    const PitchFrame* findNearestVoicedFrame (double seconds,
                                              double maxDistanceSeconds = 0.08) const
    {
        if (frames.empty() || hopSeconds <= 0.0)
            return nullptr;

        const int centre = std::clamp (
            static_cast<int> (std::llround (seconds / hopSeconds)),
            0, static_cast<int> (frames.size()) - 1);
        const int radius = std::max (0, static_cast<int> (
            std::ceil (maxDistanceSeconds / hopSeconds)));

        for (int distance = 0; distance <= radius; ++distance)
        {
            const int left = centre - distance;
            if (left >= 0 && frames[static_cast<size_t> (left)].voiced)
                return &frames[static_cast<size_t> (left)];

            const int right = centre + distance;
            if (distance != 0 && right < static_cast<int> (frames.size())
                && frames[static_cast<size_t> (right)].voiced)
                return &frames[static_cast<size_t> (right)];
        }

        return nullptr;
    }
};

struct PitchAnalysisSettings
{
    double analysisSampleRate = 24000.0;
    double hopSeconds = 0.01;
    float minimumFrequencyHz = 55.0f;
    float maximumFrequencyHz = 2500.0f;
    float silenceThresholdDb = -60.0f;
    float minimumConfidence = 0.52f;
    float yinThreshold = 0.18f;
};
