#pragma once

// AudioClip: an immutable snapshot of audio captured from the host or
// imported from a file (Phase 2 audio foundation).
//
// The preview buffer keeps the ORIGINAL channels so Play Original
// reproduces the source exactly. The mono buffer is a separate copy
// for future pitch analysis (Phase 3+); it is never derived from the
// preview buffer in place and the preview audio is never modified.
//
// Stereo-to-mono rule: monoSample = 0.5f * (left + right). A
// single-channel source is copied directly.
//
// Ownership: the processor holds the active clip as
// std::atomic<std::shared_ptr<const AudioClip>>. The audio thread
// takes a local shared_ptr copy per block; the message thread swaps
// the clip in. No locks, no allocation in processBlock().

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <memory>

struct AudioClip
{
    juce::AudioBuffer<float> previewBuffer;       // original channels, never modified
    juce::AudioBuffer<float> analysisMonoBuffer;  // mono copy for future analysis
    double sourceSampleRate = 0.0;
    juce::int64 totalSamples = 0;
    double durationSeconds = 0.0;
    juce::String sourceName;
    bool importedFromFile = false;

    bool isEmpty() const { return totalSamples <= 0; }

    // Build the mono analysis copy of a source buffer. The source is
    // never modified. Stereo: 0.5f * (left + right). Mono: direct copy.
    // Sources with more than two channels use the first two.
    static void buildMonoBuffer (const juce::AudioBuffer<float>& source,
                                 juce::AudioBuffer<float>& monoDestination)
    {
        const int numSamples = source.getNumSamples();
        monoDestination.setSize (1, numSamples, false, false, true);

        if (numSamples <= 0 || source.getNumChannels() <= 0)
            return;

        if (source.getNumChannels() == 1)
        {
            monoDestination.copyFrom (0, 0, source, 0, 0, numSamples);
            return;
        }

        const float* left = source.getReadPointer (0);
        const float* right = source.getReadPointer (1);
        float* mono = monoDestination.getWritePointer (0);
        for (int i = 0; i < numSamples; ++i)
            mono[i] = 0.5f * (left[i] + right[i]);
    }

    // Deep-copy a source buffer into a clip. Returns nullptr only if
    // the sample rate is invalid.
    static std::shared_ptr<AudioClip> fromBuffer (const juce::AudioBuffer<float>& source,
                                                  double sampleRate,
                                                  const juce::String& name,
                                                  bool importedFromFileFlag)
    {
        if (sampleRate <= 0.0)
            return nullptr;

        auto clip = std::make_shared<AudioClip>();
        clip->previewBuffer = source; // AudioBuffer copy is a deep copy
        buildMonoBuffer (source, clip->analysisMonoBuffer);
        clip->sourceSampleRate = sampleRate;
        clip->totalSamples = source.getNumSamples();
        clip->durationSeconds = sampleRate > 0.0
            ? static_cast<double> (clip->totalSamples) / sampleRate : 0.0;
        clip->sourceName = name;
        clip->importedFromFile = importedFromFileFlag;
        return clip;
    }
};
