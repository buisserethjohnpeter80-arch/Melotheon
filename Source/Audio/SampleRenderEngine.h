#pragma once

#include <JuceHeader.h>

#include <atomic>
#include <functional>
#include <memory>

#include "SampleInstrumentRenderer.h"

class SampleRenderEngine
{
public:
    using Callback = std::function<void (SampleRenderResult)>;

    SampleRenderEngine() = default;
    ~SampleRenderEngine();

    void start (std::shared_ptr<const AudioClip> oneShot,
                std::shared_ptr<const MelodyNoteSequence> melody,
                double outputSampleRate,
                Callback callback,
                std::shared_ptr<std::atomic<bool>> ownerAliveFlag);
    void cancel();

    bool isRunning() const { return running.load (std::memory_order_acquire); }
    float getProgress() const { return progress.load (std::memory_order_acquire); }

private:
    class RenderJob;

    juce::ThreadPool threadPool { 1 };
    std::atomic<int> generation { 0 };
    std::atomic<bool> running { false };
    std::atomic<float> progress { 0.0f };
};
