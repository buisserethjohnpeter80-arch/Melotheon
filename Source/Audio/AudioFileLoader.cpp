// AudioFileLoader implementation. See the header for the contract:
// decode off the real-time thread, validate everything, short status
// strings on failure, no modal alerts.

#include "AudioFileLoader.h"

#include <JuceHeader.h>

// Worker-thread job: decode, then post the result to the message
// thread. Nested class, so it can reach the loader's private state.
class AudioFileLoader::LoadJob : public juce::ThreadPoolJob
{
public:
    LoadJob (AudioFileLoader& loaderToUse, const juce::File& fileToLoad,
             Callback callbackToUse, int generationToUse,
             std::shared_ptr<std::atomic<bool>> aliveFlagToUse)
        : juce::ThreadPoolJob ("Melotheon file load"),
          loader (loaderToUse),
          file (fileToLoad),
          callback (std::move (callbackToUse)),
          generation (generationToUse),
          aliveFlag (std::move (aliveFlagToUse))
    {
    }

    JobStatus runJob() override
    {
        Result result = loader.loadFileSync (file);

        juce::MessageManager::callAsync (
            [callback = std::move (callback), result = std::move (result),
             generation = generation, aliveFlag = std::move (aliveFlag),
             loaderPtr = &loader]() mutable
            {
                if (aliveFlag == nullptr
                    || ! aliveFlag->load (std::memory_order_acquire))
                    return; // owner is gone: drop the result

                if (generation != loaderPtr->loadGeneration.load (std::memory_order_acquire))
                    return; // a newer load started: drop this result

                callback (std::move (result));
            });

        return jobHasFinished;
    }

private:
    AudioFileLoader& loader;
    juce::File file;
    Callback callback;
    int generation;
    std::shared_ptr<std::atomic<bool>> aliveFlag;
};

AudioFileLoader::AudioFileLoader()
{
    // WAV, AIFF, FLAC (+ MP3 / Ogg when the JUCE configuration
    // enables them). No codec dependency is added here.
    formatManager.registerBasicFormats();
}

AudioFileLoader::~AudioFileLoader()
{
    // Wait for any decode in flight (bounded work: files are capped
    // at kMaxImportSeconds) so no job can outlive the loader.
    threadPool.removeAllJobs (true, 5000);
}

AudioFileLoader::Result AudioFileLoader::loadFileSync (const juce::File& file)
{
    Result result;

    if (! file.existsAsFile())
    {
        result.errorMessage = "FILE NOT FOUND";
        return result;
    }
    if (file.getSize() <= 0)
    {
        result.errorMessage = "FILE EMPTY";
        return result;
    }

    std::unique_ptr<juce::AudioFormatReader> reader (formatManager.createReaderFor (file));
    if (reader == nullptr)
    {
        result.errorMessage = "UNSUPPORTED FILE";
        return result;
    }
    if (reader->lengthInSamples <= 0 || reader->sampleRate <= 0.0)
    {
        result.errorMessage = "FILE EMPTY";
        return result;
    }
    if (reader->numChannels < 1 || reader->numChannels > kMaxImportChannels)
    {
        result.errorMessage = "UNSUPPORTED FILE";
        return result;
    }

    const double duration = static_cast<double> (reader->lengthInSamples)
                          / static_cast<double> (reader->sampleRate);
    if (duration > kMaxImportSeconds)
    {
        result.errorMessage = "FILE TOO LONG";
        return result;
    }

    juce::AudioBuffer<float> buffer (static_cast<int> (reader->numChannels),
                                    static_cast<int> (reader->lengthInSamples));
    if (! reader->read (&buffer, 0, static_cast<int> (reader->lengthInSamples),
                        0, true, true))
    {
        result.errorMessage = "READ ERROR";
        return result;
    }

    result.clip = AudioClip::fromBuffer (buffer, reader->sampleRate,
                                         file.getFileName(), true);
    if (result.clip == nullptr)
        result.errorMessage = "READ ERROR";

    return result;
}

void AudioFileLoader::loadFileAsync (const juce::File& file, Callback callback,
                                    std::shared_ptr<std::atomic<bool>> aliveFlag)
{
    const int generation = loadGeneration.fetch_add (1, std::memory_order_acq_rel) + 1;
    threadPool.addJob (new LoadJob (*this, file, std::move (callback),
                                   generation, std::move (aliveFlag)),
                       true);
}

void AudioFileLoader::cancelPendingLoad()
{
    loadGeneration.fetch_add (1, std::memory_order_acq_rel);
}

juce::String AudioFileLoader::getFileFilter()
{
    juce::String filter = "*.wav;*.aif;*.aiff;*.flac";
#if defined (JUCE_USE_MP3AUDIOFORMAT) && JUCE_USE_MP3AUDIOFORMAT
    filter += ";*.mp3";
#endif
    return filter;
}
