#include "MonophonicPitchAnalyzer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace
{
constexpr double pi = 3.14159265358979323846;
constexpr int highWindowSize = 512; // 21.3 ms at 24 kHz
constexpr int lowWindowSize = 512;  // 85.3 ms after 4:1 downsampling
constexpr int rmsWindowSize = 1024;
constexpr int lowRateDivider = 4;

struct Estimate
{
    float frequencyHz = 0.0f;
    float confidence = 0.0f;
    bool valid = false;
};

float frequencyToMidi (float frequency)
{
    return 69.0f + 12.0f * std::log2 (frequency / 440.0f);
}

float midiToFrequency (float midi)
{
    return 440.0f * std::pow (2.0f, (midi - 69.0f) / 12.0f);
}

// Windowed-sinc offline resampling. The short anti-aliasing kernel is
// inexpensive compared with pitch tracking and keeps 48/96 kHz imports
// from folding high harmonics into false fundamentals.
std::vector<float> resample (const float* input, int inputSamples,
                             double sourceRate, double targetRate,
                             const std::function<bool()>& shouldCancel = {})
{
    if (input == nullptr || inputSamples <= 0 || sourceRate <= 0.0 || targetRate <= 0.0)
        return {};

    const int outputSamples = std::max (1, static_cast<int> (
        std::llround (static_cast<double> (inputSamples) * targetRate / sourceRate)));
    std::vector<float> output (static_cast<size_t> (outputSamples), 0.0f);

    constexpr int halfKernel = 8;
    const double cutoff = 0.94 * std::min (1.0, targetRate / sourceRate);
    const double sourcePerOutput = sourceRate / targetRate;

    for (int out = 0; out < outputSamples; ++out)
    {
        if ((out % 4096) == 0 && shouldCancel && shouldCancel())
            return {};

        const double position = static_cast<double> (out) * sourcePerOutput;
        const int centre = static_cast<int> (std::floor (position));
        double weighted = 0.0;
        double weightSum = 0.0;

        for (int tap = -halfKernel; tap <= halfKernel; ++tap)
        {
            const int sourceIndex = centre + tap;
            if (sourceIndex < 0 || sourceIndex >= inputSamples)
                continue;

            const double distance = position - static_cast<double> (sourceIndex);
            const double absoluteDistance = std::abs (distance);
            if (absoluteDistance > static_cast<double> (halfKernel))
                continue;

            const double sinc = absoluteDistance < 1.0e-9
                ? cutoff
                : std::sin (pi * cutoff * distance) / (pi * distance);
            const double window = 0.42
                                + 0.5 * std::cos (pi * distance / halfKernel)
                                + 0.08 * std::cos (2.0 * pi * distance / halfKernel);
            const double weight = sinc * window;
            weighted += static_cast<double> (input[sourceIndex]) * weight;
            weightSum += weight;
        }

        if (std::abs (weightSum) > 1.0e-12)
            output[static_cast<size_t> (out)] = static_cast<float> (weighted / weightSum);
    }

    return output;
}

void highPassInPlace (std::vector<float>& samples, double sampleRate)
{
    if (samples.empty() || sampleRate <= 0.0)
        return;

    const float coefficient = static_cast<float> (
        std::exp (-2.0 * pi * 35.0 / sampleRate));
    float previousInput = samples.front();
    float previousOutput = 0.0f;

    for (float& sample : samples)
    {
        const float output = coefficient * (previousOutput + sample - previousInput);
        previousInput = sample;
        previousOutput = output;
        sample = output;
    }
}

std::vector<float> makePadded (const std::vector<float>& source, int padding)
{
    std::vector<float> padded (source.size() + static_cast<size_t> (padding * 2), 0.0f);
    std::copy (source.begin(), source.end(), padded.begin() + padding);
    return padded;
}

Estimate estimateYin (const float* frame, int windowSize, double sampleRate,
                      float minimumFrequency, float maximumFrequency,
                      float threshold, std::vector<float>& difference,
                      std::vector<float>& cumulativeDifference)
{
    Estimate estimate;
    if (frame == nullptr || sampleRate <= 0.0 || minimumFrequency <= 0.0f
        || maximumFrequency <= minimumFrequency)
        return estimate;

    const int minimumLag = std::max (2, static_cast<int> (
        std::floor (sampleRate / static_cast<double> (maximumFrequency))));
    const int maximumLag = std::min (windowSize / 2 - 1, static_cast<int> (
        std::ceil (sampleRate / static_cast<double> (minimumFrequency))));
    if (maximumLag <= minimumLag + 1)
        return estimate;

    difference.assign (static_cast<size_t> (maximumLag + 1), 0.0f);
    cumulativeDifference.assign (static_cast<size_t> (maximumLag + 1), 1.0f);

    for (int lag = 1; lag <= maximumLag; ++lag)
    {
        double sum = 0.0;
        const int count = windowSize - lag;
        for (int sample = 0; sample < count; ++sample)
        {
            const double delta = static_cast<double> (frame[sample])
                               - static_cast<double> (frame[sample + lag]);
            sum += delta * delta;
        }
        difference[static_cast<size_t> (lag)] = static_cast<float> (sum);
    }

    double runningSum = 0.0;
    for (int lag = 1; lag <= maximumLag; ++lag)
    {
        runningSum += difference[static_cast<size_t> (lag)];
        cumulativeDifference[static_cast<size_t> (lag)] = runningSum > 1.0e-20
            ? static_cast<float> (difference[static_cast<size_t> (lag)]
                                  * static_cast<double> (lag) / runningSum)
            : 1.0f;
    }

    int candidate = -1;
    for (int lag = minimumLag; lag <= maximumLag; ++lag)
    {
        if (cumulativeDifference[static_cast<size_t> (lag)] < threshold)
        {
            candidate = lag;
            while (candidate + 1 <= maximumLag
                   && cumulativeDifference[static_cast<size_t> (candidate + 1)]
                      < cumulativeDifference[static_cast<size_t> (candidate)])
                ++candidate;
            break;
        }
    }

    if (candidate < 0)
    {
        candidate = minimumLag;
        for (int lag = minimumLag + 1; lag <= maximumLag; ++lag)
            if (cumulativeDifference[static_cast<size_t> (lag)]
                < cumulativeDifference[static_cast<size_t> (candidate)])
                candidate = lag;
    }

    double refinedLag = static_cast<double> (candidate);
    if (candidate > minimumLag && candidate < maximumLag)
    {
        const double left = cumulativeDifference[static_cast<size_t> (candidate - 1)];
        const double centre = cumulativeDifference[static_cast<size_t> (candidate)];
        const double right = cumulativeDifference[static_cast<size_t> (candidate + 1)];
        const double denominator = left - 2.0 * centre + right;
        if (std::abs (denominator) > 1.0e-12)
            refinedLag += std::clamp (0.5 * (left - right) / denominator, -0.5, 0.5);
    }

    const float frequency = static_cast<float> (sampleRate / refinedLag);
    if (! std::isfinite (frequency) || frequency < minimumFrequency * 0.95f
        || frequency > maximumFrequency * 1.05f)
        return estimate;

    estimate.frequencyHz = frequency;
    estimate.confidence = std::clamp (
        1.0f - cumulativeDifference[static_cast<size_t> (candidate)], 0.0f, 1.0f);
    estimate.valid = true;
    return estimate;
}

bool nearIntegerHarmonic (float higher, float lower)
{
    if (lower <= 0.0f || higher <= lower)
        return false;
    const float ratio = higher / lower;
    for (float harmonic : { 2.0f, 3.0f, 4.0f })
        if (std::abs (ratio - harmonic) < harmonic * 0.045f)
            return true;
    return false;
}

Estimate chooseEstimate (Estimate low, Estimate high)
{
    if (! low.valid)
        return high;
    if (! high.valid)
        return low;

    const float lowerFrequency = std::min (low.frequencyHz, high.frequencyHz);
    const float higherFrequency = std::max (low.frequencyHz, high.frequencyHz);
    const float semitoneDistance = 12.0f * std::abs (
        std::log2 (high.frequencyHz / low.frequencyHz));

    // Both windows found the same pitch. Average in log-frequency so
    // neither detector is biased by its linear-Hz scale.
    if (semitoneDistance < 0.75f)
    {
        const float total = std::max (0.001f, low.confidence + high.confidence);
        const float midi = (frequencyToMidi (low.frequencyHz) * low.confidence
                          + frequencyToMidi (high.frequencyHz) * high.confidence) / total;
        return { midiToFrequency (midi), std::max (low.confidence, high.confidence), true };
    }

    // Above the long window's reliable 900 Hz range, the short window
    // is authoritative. The low-rate detector can otherwise report a
    // mathematically clean subharmonic for whistles.
    if (high.frequencyHz > 900.0f && high.confidence >= 0.72f)
        return high;

    // Prefer a credible fundamental over its octave / higher harmonic.
    if (nearIntegerHarmonic (higherFrequency, lowerFrequency))
    {
        const Estimate& lower = low.frequencyHz < high.frequencyHz ? low : high;
        const Estimate& higher = low.frequencyHz < high.frequencyHz ? high : low;
        if (lower.confidence + 0.14f >= higher.confidence)
            return lower;
    }

    if (low.frequencyHz < 220.0f && low.confidence + 0.10f >= high.confidence)
        return low;
    // On a near tie, the longer low-frequency view is less likely to
    // lock onto a harmonic, so it wins.
    return high.confidence > low.confidence + 0.08f ? high : low;
}

float median (std::vector<float> values)
{
    if (values.empty())
        return 0.0f;
    const auto middle = values.begin() + static_cast<std::ptrdiff_t> (values.size() / 2);
    std::nth_element (values.begin(), middle, values.end());
    float value = *middle;
    if ((values.size() % 2) == 0)
    {
        const auto lower = std::max_element (values.begin(), middle);
        value = 0.5f * (value + *lower);
    }
    return value;
}

void stabiliseCurve (std::vector<PitchFrame>& frames)
{
    if (frames.size() < 3)
        return;

    // Bridge a single missing 10 ms frame only when both neighbours
    // agree within one semitone.
    for (size_t i = 1; i + 1 < frames.size(); ++i)
    {
        if (! frames[i].voiced && frames[i - 1].voiced && frames[i + 1].voiced
            && std::abs (frames[i - 1].midiPitch - frames[i + 1].midiPitch) < 1.0f)
        {
            frames[i].voiced = true;
            frames[i].midiPitch = 0.5f * (frames[i - 1].midiPitch + frames[i + 1].midiPitch);
            frames[i].frequencyHz = midiToFrequency (frames[i].midiPitch);
            frames[i].confidence = 0.8f * std::min (frames[i - 1].confidence,
                                                   frames[i + 1].confidence);
        }
    }

    const std::vector<PitchFrame> original = frames;
    for (size_t i = 1; i + 1 < frames.size(); ++i)
    {
        if (! original[i - 1].voiced || ! original[i].voiced || ! original[i + 1].voiced)
            continue;

        std::array<float, 3> pitches {
            original[i - 1].midiPitch, original[i].midiPitch, original[i + 1].midiPitch
        };
        std::sort (pitches.begin(), pitches.end());
        const float localMedian = pitches[1];

        // Replace octave glitches and lightly calm smaller detector
        // jitter without flattening ordinary vocal vibrato.
        if (std::abs (original[i].midiPitch - localMedian) > 5.0f
            || std::abs (original[i - 1].midiPitch - original[i + 1].midiPitch) < 0.8f)
        {
            frames[i].midiPitch = 0.75f * original[i].midiPitch + 0.25f * localMedian;
            if (std::abs (original[i].midiPitch - localMedian) > 5.0f)
                frames[i].midiPitch = localMedian;
            frames[i].frequencyHz = midiToFrequency (frames[i].midiPitch);
        }
    }
}
} // namespace

std::shared_ptr<PitchAnalysisResult> MonophonicPitchAnalyzer::analyze (
    const juce::AudioBuffer<float>& monoInput,
    double sourceSampleRate,
    const PitchAnalysisSettings& requestedSettings,
    CancelCallback shouldCancel,
    ProgressCallback reportProgress)
{
    if (monoInput.getNumChannels() < 1 || monoInput.getNumSamples() <= 0
        || sourceSampleRate <= 0.0)
        return nullptr;

    PitchAnalysisSettings settings = requestedSettings;
    settings.analysisSampleRate = std::clamp (settings.analysisSampleRate, 12000.0, 48000.0);
    settings.hopSeconds = std::clamp (settings.hopSeconds, 0.005, 0.05);
    settings.minimumFrequencyHz = std::clamp (settings.minimumFrequencyHz, 30.0f, 500.0f);
    settings.maximumFrequencyHz = std::clamp (
        settings.maximumFrequencyHz, settings.minimumFrequencyHz + 20.0f,
        static_cast<float> (settings.analysisSampleRate * 0.45));
    settings.minimumConfidence = std::clamp (settings.minimumConfidence, 0.2f, 0.95f);
    settings.yinThreshold = std::clamp (settings.yinThreshold, 0.05f, 0.35f);

    if (shouldCancel && shouldCancel())
        return nullptr;

    std::vector<float> analysis = resample (
        monoInput.getReadPointer (0), monoInput.getNumSamples(), sourceSampleRate,
        settings.analysisSampleRate, shouldCancel);
    if (analysis.empty())
        return nullptr;
    highPassInPlace (analysis, settings.analysisSampleRate);

    if (reportProgress)
        reportProgress (0.03f);
    if (shouldCancel && shouldCancel())
        return nullptr;

    std::vector<float> lowRate = resample (
        analysis.data(), static_cast<int> (analysis.size()), settings.analysisSampleRate,
        settings.analysisSampleRate / lowRateDivider, shouldCancel);
    if (lowRate.empty())
        return nullptr;
    const auto paddedHigh = makePadded (analysis, rmsWindowSize / 2);
    const auto paddedLow = makePadded (lowRate, lowWindowSize / 2);

    const int hopSamples = std::max (1, static_cast<int> (
        std::llround (settings.analysisSampleRate * settings.hopSeconds)));
    const int frameCount = std::max (1, static_cast<int> (
        std::ceil (static_cast<double> (analysis.size()) / hopSamples)));

    auto result = std::make_shared<PitchAnalysisResult>();
    result->durationSeconds = static_cast<double> (monoInput.getNumSamples()) / sourceSampleRate;
    result->analysisSampleRate = settings.analysisSampleRate;
    result->hopSeconds = settings.hopSeconds;
    result->frames.reserve (static_cast<size_t> (frameCount));

    std::vector<float> lowDifference, lowCumulative, highDifference, highCumulative;
    const float linearSilence = std::pow (10.0f, settings.silenceThresholdDb / 20.0f);

    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex)
    {
        if ((frameIndex % 8) == 0 && shouldCancel && shouldCancel())
            return nullptr;

        const int centreHigh = std::min (
            static_cast<int> (analysis.size()) - 1, frameIndex * hopSamples);
        const int centreLow = std::min (
            static_cast<int> (lowRate.size()) - 1,
            static_cast<int> (std::llround (
                static_cast<double> (centreHigh) / lowRateDivider)));

        const float* rmsFrame = paddedHigh.data() + centreHigh;
        double sumSquares = 0.0;
        for (int i = 0; i < rmsWindowSize; ++i)
        {
            const double sample = rmsFrame[i];
            sumSquares += sample * sample;
        }
        const float rms = static_cast<float> (std::sqrt (sumSquares / rmsWindowSize));

        PitchFrame frame;
        frame.timeSeconds = std::min (
            result->durationSeconds, static_cast<double> (frameIndex) * settings.hopSeconds);
        frame.rmsDb = rms > 1.0e-10f ? 20.0f * std::log10 (rms) : -100.0f;

        if (rms >= linearSilence)
        {
            const Estimate low = estimateYin (
                paddedLow.data() + centreLow, lowWindowSize,
                settings.analysisSampleRate / lowRateDivider,
                settings.minimumFrequencyHz,
                std::min (900.0f, settings.maximumFrequencyHz),
                settings.yinThreshold, lowDifference, lowCumulative);
            const Estimate high = estimateYin (
                paddedHigh.data() + centreHigh + (rmsWindowSize - highWindowSize) / 2,
                highWindowSize, settings.analysisSampleRate,
                std::max (220.0f, settings.minimumFrequencyHz),
                settings.maximumFrequencyHz,
                settings.yinThreshold, highDifference, highCumulative);
            const Estimate selected = chooseEstimate (low, high);

            if (selected.valid && selected.confidence >= settings.minimumConfidence)
            {
                frame.voiced = true;
                frame.frequencyHz = selected.frequencyHz;
                frame.midiPitch = frequencyToMidi (selected.frequencyHz);
                frame.confidence = selected.confidence;
            }
        }

        result->frames.push_back (frame);

        if (reportProgress && ((frameIndex % 16) == 0 || frameIndex + 1 == frameCount))
            reportProgress (0.03f + 0.97f * static_cast<float> (frameIndex + 1)
                                         / static_cast<float> (frameCount));
    }

    stabiliseCurve (result->frames);

    std::vector<float> pitches, confidences, frequencies;
    pitches.reserve (result->frames.size());
    confidences.reserve (result->frames.size());
    frequencies.reserve (result->frames.size());
    for (const auto& frame : result->frames)
    {
        if (! frame.voiced)
            continue;
        pitches.push_back (frame.midiPitch);
        confidences.push_back (frame.confidence);
        frequencies.push_back (frame.frequencyHz);
    }

    result->voicedFrameCount = static_cast<int> (pitches.size());
    if (! pitches.empty())
    {
        const auto [minimum, maximum] = std::minmax_element (pitches.begin(), pitches.end());
        result->minMidi = *minimum;
        result->maxMidi = *maximum;
        result->medianConfidence = median (std::move (confidences));
        result->medianFrequencyHz = median (std::move (frequencies));
    }

    if (reportProgress)
        reportProgress (1.0f);
    return result;
}
