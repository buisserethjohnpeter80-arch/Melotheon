#include "SampleInstrumentRenderer.h"

#include "../Analysis/MonophonicPitchAnalyzer.h"

#include <algorithm>
#include <cmath>

namespace
{
float readInterpolated (const float* source, int sourceSamples, double position)
{
    if (source == nullptr || sourceSamples <= 0)
        return 0.0f;

    position = std::clamp (position, 0.0, static_cast<double> (sourceSamples - 1));
    const int index0 = static_cast<int> (position);
    const int index1 = std::min (index0 + 1, sourceSamples - 1);
    const float fraction = static_cast<float> (position - static_cast<double> (index0));
    return source[index0] + (source[index1] - source[index0]) * fraction;
}

// Median pitch of an analyzed one-shot as a MIDI note. Unpitched,
// noisy or unanalyzable sounds fall back to middle C rather than
// rejecting a sound the user deliberately chose.
int rootMidiFromAnalysis (const PitchAnalysisResult* pitchResult)
{
    if (pitchResult != nullptr && pitchResult->hasPitch()
        && pitchResult->medianFrequencyHz > 0.0f)
    {
        return std::clamp (static_cast<int> (std::lround (
            69.0 + 12.0 * std::log2 (
                static_cast<double> (pitchResult->medianFrequencyHz) / 440.0))),
            0, 127);
    }
    return 60;
}
} // namespace

int SampleInstrumentRenderer::detectRootMidi (const AudioClip& oneShot)
{
    if (oneShot.isEmpty() || oneShot.sourceSampleRate <= 0.0)
        return 60;

    PitchAnalysisSettings pitchSettings;
    pitchSettings.silenceThresholdDb = -66.0f;
    pitchSettings.minimumConfidence = 0.45f;
    auto pitchResult = MonophonicPitchAnalyzer::analyze (
        oneShot.analysisMonoBuffer, oneShot.sourceSampleRate, pitchSettings);

    if (pitchResult == nullptr)
        return 60;
    return rootMidiFromAnalysis (pitchResult.get());
}

SampleRenderResult SampleInstrumentRenderer::render (
    const AudioClip& oneShot,
    const MelodyNoteSequence& melody,
    double outputSampleRate,
    CancelCallback shouldCancel,
    ProgressCallback reportProgress)
{
    SampleRenderResult result;
    if (oneShot.isEmpty() || oneShot.sourceSampleRate <= 0.0
        || oneShot.previewBuffer.getNumChannels() <= 0)
    {
        result.errorMessage = "SOUND FILE EMPTY";
        return result;
    }
    if (oneShot.durationSeconds > 15.0)
    {
        result.errorMessage = "USE A SHORT ONE-SHOT";
        return result;
    }
    if (melody.empty())
    {
        result.errorMessage = "ANALYZE FIRST";
        return result;
    }
    if (outputSampleRate <= 0.0)
    {
        result.errorMessage = "INVALID SAMPLE RATE";
        return result;
    }

    PitchAnalysisSettings pitchSettings;
    pitchSettings.silenceThresholdDb = -66.0f;
    pitchSettings.minimumConfidence = 0.45f;
    auto pitchResult = MonophonicPitchAnalyzer::analyze (
        oneShot.analysisMonoBuffer, oneShot.sourceSampleRate, pitchSettings,
        shouldCancel,
        [reportProgress] (float value)
        {
            if (reportProgress)
                reportProgress (value * 0.30f);
        });

    if (pitchResult == nullptr)
    {
        if (shouldCancel && shouldCancel())
            return result;
        result.errorMessage = "SOUND ANALYSIS ERROR";
        return result;
    }
    // Pitched sounds get automatic root-note detection. For noise,
    // percussion or very short attacks, use middle C rather than rejecting
    // a sound the user deliberately chose.
    result.rootMidi = rootMidiFromAnalysis (pitchResult.get());
    result.rootFrequencyHz = (pitchResult->hasPitch()
                              && pitchResult->medianFrequencyHz > 0.0f)
        ? pitchResult->medianFrequencyHz : 261.6256f;

    const double renderDuration = std::clamp (
        std::max (melody.durationSeconds,
                  melody.notes.back().endSeconds + 0.05), 0.01, 300.0);
    const int outputSamples = std::max (1, static_cast<int> (
        std::llround (renderDuration * outputSampleRate)));
    const int outputChannels = std::clamp (
        oneShot.previewBuffer.getNumChannels(), 1, 2);
    juce::AudioBuffer<float> output (outputChannels, outputSamples);
    output.clear();

    const int sourceSamples = oneShot.previewBuffer.getNumSamples();
    const int loopStart = std::clamp (
        static_cast<int> (oneShot.sourceSampleRate * 0.05), 0,
        std::max (0, sourceSamples - 2));
    const int loopEnd = std::max (loopStart + 1,
        sourceSamples - std::min (sourceSamples / 8,
                                  static_cast<int> (oneShot.sourceSampleRate * 0.03)));
    const int loopLength = std::max (1, loopEnd - loopStart);

    for (size_t noteIndex = 0; noteIndex < melody.notes.size(); ++noteIndex)
    {
        if (shouldCancel && shouldCancel())
            return {};

        const MelodyNote& note = melody.notes[noteIndex];
        const int startSample = std::clamp (static_cast<int> (
            std::llround (note.startSeconds * outputSampleRate)), 0, outputSamples);
        const int endSample = std::clamp (static_cast<int> (
            std::llround (note.endSeconds * outputSampleRate)), startSample, outputSamples);
        const int noteSamples = endSample - startSample;
        if (noteSamples <= 0)
            continue;

        const double semitoneShift = static_cast<double> (note.midiNote - result.rootMidi);
        const double playbackStep = std::pow (2.0, semitoneShift / 12.0)
                                  * oneShot.sourceSampleRate / outputSampleRate;
        const int attackSamples = std::max (1, std::min (
            noteSamples / 3, static_cast<int> (outputSampleRate * 0.006)));
        const int releaseSamples = std::max (1, std::min (
            noteSamples / 3, static_cast<int> (outputSampleRate * 0.030)));

        for (int local = 0; local < noteSamples; ++local)
        {
            if ((local % 8192) == 0 && shouldCancel && shouldCancel())
                return {};

            double sourcePosition = static_cast<double> (local) * playbackStep;
            if (sourcePosition >= static_cast<double> (sourceSamples - 1))
            {
                // Preserve the one-shot attack once, then loop its body so a
                // long sung note stays audible without retriggering the attack.
                sourcePosition = static_cast<double> (loopStart)
                               + std::fmod (sourcePosition - loopStart,
                                           static_cast<double> (loopLength));
            }

            float envelope = 1.0f;
            if (local < attackSamples)
                envelope *= static_cast<float> (local) / static_cast<float> (attackSamples);
            const int remaining = noteSamples - local;
            if (remaining < releaseSamples)
                envelope *= static_cast<float> (remaining) / static_cast<float> (releaseSamples);
            const float gain = envelope * std::clamp (note.velocity, 0.0f, 1.0f);

            for (int channel = 0; channel < outputChannels; ++channel)
            {
                const float* source = oneShot.previewBuffer.getReadPointer (
                    std::min (channel, oneShot.previewBuffer.getNumChannels() - 1));
                output.addSample (channel, startSample + local,
                                  readInterpolated (source, sourceSamples, sourcePosition) * gain);
            }
        }

        if (reportProgress)
            reportProgress (0.30f + 0.68f
                * static_cast<float> (noteIndex + 1)
                / static_cast<float> (melody.notes.size()));
    }

    float peak = 0.0f;
    for (int channel = 0; channel < outputChannels; ++channel)
        peak = std::max (peak, output.getMagnitude (channel, 0, outputSamples));
    if (peak > 0.98f)
        output.applyGain (0.98f / peak);

    auto rendered = AudioClip::fromBuffer (
        output, outputSampleRate, "MELOTHEON INSTRUMENT", false);
    if (rendered == nullptr || rendered->isEmpty())
    {
        result.errorMessage = "INSTRUMENT RENDER ERROR";
        return result;
    }

    result.renderedClip = std::move (rendered);
    if (reportProgress)
        reportProgress (1.0f);
    return result;
}
