#pragma once

// AudioClip: an immutable snapshot of audio captured from the host or
// imported from a file (Phase 2 audio foundation, Phase 3A analysis input).
//
// The preview buffer keeps the ORIGINAL channels so Play Original
// reproduces the source exactly. The mono buffer is a separate copy
// for offline pitch analysis; it is never derived from the
// preview buffer in place and the preview audio is never modified.
//
// Stereo-to-mono normally uses 0.5f * (left + right). If the channels
// are strongly anti-phase, summing would erase the melody. In that
// case the louder channel is selected for analysis only. A single-
// channel source is copied directly.
//
// Ownership: the processor and preview engine hold the active clip on
// the message thread. AudioPreviewEngine publishes a stable non-owning
// pointer to processBlock() and waits for an in-flight render before it
// releases the old owner. No shared_ptr operation, lock or allocation
// occurs in processBlock().

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <cmath>
#include <memory>

struct AudioClip
{
    juce::AudioBuffer<float> previewBuffer;       // original channels, never modified
    juce::AudioBuffer<float> analysisMonoBuffer;  // cancellation-safe mono analysis copy
    double sourceSampleRate = 0.0;
    juce::int64 totalSamples = 0;
    double durationSeconds = 0.0;
    juce::String sourceName;
    bool importedFromFile = false;

    bool isEmpty() const { return totalSamples <= 0; }

    // Build the mono analysis copy of a source buffer. The source is
    // never modified. Stereo normally averages L/R, but selects the
    // louder channel when averaging would cause severe phase
    // cancellation. Mono is copied directly. Sources with more than
    // two channels use the first two.
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

        double leftEnergy = 0.0;
        double rightEnergy = 0.0;
        double crossEnergy = 0.0;
        for (int i = 0; i < numSamples; ++i)
        {
            const double l = left[i];
            const double r = right[i];
            leftEnergy += l * l;
            rightEnergy += r * r;
            crossEnergy += l * r;
        }

        const double louderEnergy = juce::jmax (leftEnergy, rightEnergy);
        const double quieterEnergy = juce::jmin (leftEnergy, rightEnergy);
        const double denominator = std::sqrt (leftEnergy * rightEnergy);
        const double correlation = denominator > 1.0e-20
            ? crossEnergy / denominator : 1.0;
        const double averagedEnergy = 0.25 * (leftEnergy + rightEnergy + 2.0 * crossEnergy);
        const bool oneChannelOnly = louderEnergy > 1.0e-20
                                 && quieterEnergy < louderEnergy * 0.01;
        const bool severeCancellation = correlation < -0.25
                                     || (louderEnergy > 1.0e-20
                                         && averagedEnergy < louderEnergy * 0.08);

        if (oneChannelOnly || severeCancellation)
        {
            const float* selected = leftEnergy >= rightEnergy ? left : right;
            juce::FloatVectorOperations::copy (mono, selected, numSamples);
            return;
        }

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
