#include "PitchAnalysisEngine.h"

#include "MonophonicPitchAnalyzer.h"

class PitchAnalysisEngine::AnalysisJob : public juce::ThreadPoolJob
{
public:
    AnalysisJob (PitchAnalysisEngine& engineToUse,
                 std::shared_ptr<const AudioClip> clipToUse,
                 PitchAnalysisSettings settingsToUse,
                 Callback callbackToUse,
                 int generationToUse,
                 std::shared_ptr<std::atomic<bool>> aliveFlagToUse)
        : juce::ThreadPoolJob ("Melotheon pitch analysis"),
          engine (engineToUse),
          clip (std::move (clipToUse)),
          settings (settingsToUse),
          callback (std::move (callbackToUse)),
          jobGeneration (generationToUse),
          aliveFlag (std::move (aliveFlagToUse))
    {
    }

    JobStatus runJob() override
    {
        auto cancelled = [this]
        {
            return shouldExit()
                || jobGeneration != engine.generation.load (std::memory_order_acquire)
                || aliveFlag == nullptr
                || ! aliveFlag->load (std::memory_order_acquire);
        };

        auto report = [this] (float value)
        {
            if (jobGeneration == engine.generation.load (std::memory_order_acquire))
                engine.progress.store (juce::jlimit (0.0f, 1.0f, value),
                                       std::memory_order_release);
        };

        auto result = MonophonicPitchAnalyzer::analyze (
            clip->analysisMonoBuffer, clip->sourceSampleRate, settings,
            cancelled, report);

        if (result == nullptr || cancelled())
            return jobHasFinished;

        juce::MessageManager::callAsync (
            [enginePtr = &engine, callback = std::move (callback),
             result = std::shared_ptr<const PitchAnalysisResult> (std::move (result)),
             generation = jobGeneration, alive = std::move (aliveFlag)] () mutable
            {
                if (alive == nullptr || ! alive->load (std::memory_order_acquire))
                    return;
                if (generation != enginePtr->generation.load (std::memory_order_acquire))
                    return;

                enginePtr->progress.store (1.0f, std::memory_order_release);
                enginePtr->running.store (false, std::memory_order_release);
                callback (std::move (result));
            });

        return jobHasFinished;
    }

private:
    PitchAnalysisEngine& engine;
    std::shared_ptr<const AudioClip> clip;
    PitchAnalysisSettings settings;
    Callback callback;
    int jobGeneration = 0;
    std::shared_ptr<std::atomic<bool>> aliveFlag;
};

PitchAnalysisEngine::~PitchAnalysisEngine()
{
    cancel();
    // Cancellation is checked every eight 10 ms frames. Waiting here
    // guarantees no worker keeps a reference to this engine.
    threadPool.removeAllJobs (true, -1);
}

void PitchAnalysisEngine::start (
    std::shared_ptr<const AudioClip> clip,
    PitchAnalysisSettings settings,
    Callback callback,
    std::shared_ptr<std::atomic<bool>> ownerAliveFlag)
{
    cancel();
    if (clip == nullptr || clip->isEmpty() || callback == nullptr
        || ownerAliveFlag == nullptr)
        return;

    const int newGeneration = generation.fetch_add (1, std::memory_order_acq_rel) + 1;
    progress.store (0.0f, std::memory_order_release);
    running.store (true, std::memory_order_release);
    threadPool.addJob (new AnalysisJob (*this, std::move (clip), settings,
                                       std::move (callback), newGeneration,
                                       std::move (ownerAliveFlag)),
                       true);
}

void PitchAnalysisEngine::cancel()
{
    generation.fetch_add (1, std::memory_order_acq_rel);
    running.store (false, std::memory_order_release);
    progress.store (0.0f, std::memory_order_release);
}
