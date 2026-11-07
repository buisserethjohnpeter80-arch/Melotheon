#pragma once

// Owns the single Phase 3A analysis worker. Starting a newer analysis
// invalidates the older result. Progress is atomic so the editor can
// poll it without blocking. Completion always returns on JUCE's message
// thread and never touches processBlock().

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>

#include "PitchAnalysis.h"
#include "../Audio/AudioClip.h"

class PitchAnalysisEngine
{
public:
    using Callback = std::function<void (std::shared_ptr<const PitchAnalysisResult>)>;

    PitchAnalysisEngine() = default;
    ~PitchAnalysisEngine();

    void start (std::shared_ptr<const AudioClip> clip,
                PitchAnalysisSettings settings,
                Callback callback,
                std::shared_ptr<std::atomic<bool>> ownerAliveFlag);
    void cancel();

    bool isRunning() const { return running.load (std::memory_order_acquire); }
    float getProgress() const { return progress.load (std::memory_order_acquire); }

private:
    class AnalysisJob;

    juce::ThreadPool threadPool { 1 };
    std::atomic<int> generation { 0 };
    std::atomic<bool> running { false };
    std::atomic<float> progress { 0.0f };
};
