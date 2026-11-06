#pragma once

// Phase 3A offline monophonic pitch tracker.
//
// It runs on a worker thread, never in processBlock(). The input is
// resampled to 24 kHz, high-pass filtered, and inspected every 10 ms
// with two YIN resolutions: an 85 ms low-frequency window and a 21 ms
// high-frequency window. The result is a pitch curve only. Note
// segmentation and MIDI generation deliberately remain Phase 3B/3D.

#include <juce_audio_basics/juce_audio_basics.h>

#include <functional>
#include <memory>

#include "PitchAnalysis.h"

class MonophonicPitchAnalyzer
{
public:
    using CancelCallback = std::function<bool()>;
    using ProgressCallback = std::function<void (float)>;

    // Returns nullptr only when cancellation was requested or the input
    // metadata is invalid. A valid silent/noisy input returns a result
    // with frames but hasPitch() == false.
    static std::shared_ptr<PitchAnalysisResult> analyze (
        const juce::AudioBuffer<float>& monoInput,
        double sourceSampleRate,
        const PitchAnalysisSettings& settings = {},
        CancelCallback shouldCancel = {},
        ProgressCallback reportProgress = {});
};
