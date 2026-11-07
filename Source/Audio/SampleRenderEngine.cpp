#include "SampleRenderEngine.h"

class SampleRenderEngine::RenderJob : public juce::ThreadPoolJob
{
public:
    RenderJob (SampleRenderEngine& engineToUse,
               std::shared_ptr<const AudioClip> oneShotToUse,
               std::shared_ptr<const MelodyNoteSequence> melodyToUse,
               double outputSampleRateToUse,
               Callback callbackToUse,
               int generationToUse,
               std::shared_ptr<std::atomic<bool>> aliveFlagToUse)
        : juce::ThreadPoolJob ("Melotheon sample render"),
          engine (engineToUse),
          oneShot (std::move (oneShotToUse)),
          melody (std::move (melodyToUse)),
          outputSampleRate (outputSampleRateToUse),
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

        auto result = SampleInstrumentRenderer::render (
            *oneShot, *melody, outputSampleRate, cancelled, report);
        if (cancelled())
            return jobHasFinished;

        juce::MessageManager::callAsync (
            [enginePtr = &engine, callback = std::move (callback),
             result = std::move (result), generation = jobGeneration,
             alive = std::move (aliveFlag)] () mutable
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
    SampleRenderEngine& engine;
    std::shared_ptr<const AudioClip> oneShot;
    std::shared_ptr<const MelodyNoteSequence> melody;
    double outputSampleRate = 44100.0;
    Callback callback;
    int jobGeneration = 0;
    std::shared_ptr<std::atomic<bool>> aliveFlag;
};

SampleRenderEngine::~SampleRenderEngine()
{
    cancel();
    threadPool.removeAllJobs (true, -1);
}

void SampleRenderEngine::start (
    std::shared_ptr<const AudioClip> oneShot,
    std::shared_ptr<const MelodyNoteSequence> melody,
    double outputSampleRate,
    Callback callback,
    std::shared_ptr<std::atomic<bool>> ownerAliveFlag)
{
    cancel();
    if (oneShot == nullptr || oneShot->isEmpty() || melody == nullptr
        || melody->empty() || outputSampleRate <= 0.0 || callback == nullptr
        || ownerAliveFlag == nullptr)
        return;

    const int nextGeneration = generation.fetch_add (1, std::memory_order_acq_rel) + 1;
    progress.store (0.0f, std::memory_order_release);
    running.store (true, std::memory_order_release);
    threadPool.addJob (new RenderJob (
        *this, std::move (oneShot), std::move (melody), outputSampleRate,
        std::move (callback), nextGeneration, std::move (ownerAliveFlag)), true);
}

void SampleRenderEngine::cancel()
{
    generation.fetch_add (1, std::memory_order_acq_rel);
    running.store (false, std::memory_order_release);
    progress.store (0.0f, std::memory_order_release);
}
