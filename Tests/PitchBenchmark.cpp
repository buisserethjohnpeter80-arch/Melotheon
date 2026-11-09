// MELOTHEON Phase 1 — Professional Pitch Detection Benchmark.
// Implementation of the measurement utilities declared in PitchBenchmark.h.
// Measurement only: every level calls the real production analysis code.

#include "PitchBenchmark.h"

#include "Analysis/MonophonicPitchAnalyzer.h"
#include "Analysis/NoteSegmenter.h"
#include "Audio/AudioClip.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>

namespace pitchbenchmark
{
namespace
{

constexpr double kTwoPi = 6.28318530717958647692;

double bufferRms (const juce::AudioBuffer<float>& buffer)
{
    if (buffer.getNumSamples() <= 0)
        return 0.0;
    double sumSquares = 0.0;
    const float* data = buffer.getReadPointer (0);
    for (int i = 0; i < buffer.getNumSamples(); ++i)
        sumSquares += static_cast<double> (data[i]) * data[i];
    return std::sqrt (sumSquares / static_cast<double> (buffer.getNumSamples()));
}

float bufferPeak (const juce::AudioBuffer<float>& buffer)
{
    float peak = 0.0f;
    const float* data = buffer.getReadPointer (0);
    for (int i = 0; i < buffer.getNumSamples(); ++i)
        peak = std::max (peak, std::abs (data[i]));
    return peak;
}

double medianOf (std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort (values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 1)
        return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

double meanOf (const std::vector<double>& values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    return std::accumulate (values.begin(), values.end(), 0.0)
         / static_cast<double> (values.size());
}

std::string trim (const std::string& text)
{
    const size_t first = text.find_first_not_of (" \t\r\n");
    if (first == std::string::npos)
        return {};
    const size_t last = text.find_last_not_of (" \t\r\n");
    return text.substr (first, last - first + 1);
}

std::string lowercased (std::string text)
{
    std::transform (text.begin(), text.end(), text.begin(),
                    [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
    return text;
}

std::vector<std::string> splitCsvLine (const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream (line);
    std::string field;
    while (std::getline (stream, field, ','))
        fields.push_back (trim (field));
    return fields;
}

bool parseDoubleField (const std::string& text, double& value)
{
    try
    {
        size_t used = 0;
        value = std::stod (text, &used);
        return used == text.size();
    }
    catch (...)
    {
        return false;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Octave classification
// ---------------------------------------------------------------------------

OctaveClassification classifyOctaveError (double cents)
{
    OctaveClassification result;
    const double magnitude = std::abs (cents);
    for (int octaves = 1; octaves <= 3; ++octaves)
    {
        if (std::abs (magnitude - kOctaveCents * static_cast<double> (octaves))
            <= kOctaveWindowCents)
        {
            result.isOctaveError = true;
            result.octaves = octaves;
            result.up = cents > 0.0;
            return result;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Settings helpers
// ---------------------------------------------------------------------------

PitchAnalysisSettings productionMappedAnalysisSettings()
{
    PitchAnalysisSettings settings;
    settings.analysisSampleRate = 24000.0;
    settings.hopSeconds = 0.01;
    settings.minimumFrequencyHz = 55.0f;
    settings.maximumFrequencyHz = 2500.0f;
    settings.silenceThresholdDb = -60.9f; // noiseGate knob 0.3
    settings.minimumConfidence = 0.525f;  // sensitivity knob 0.65
    settings.yinThreshold = 0.18f;
    return settings;
}

NoteSegmentationSettings productionMappedSegmentationSettings()
{
    NoteSegmentationSettings settings;
    settings.minimumNoteSeconds = 0.0775; // transition knob 0.5
    settings.maximumGapSeconds = 0.04;
    settings.pitchHysteresisSemitones = 0.75f; // variation knob 0.5
    return settings;
}

// ---------------------------------------------------------------------------
// Synthetic generation
// ---------------------------------------------------------------------------

juce::AudioBuffer<float> makeSineTone (double sampleRate,
                                       double durationSeconds,
                                       double frequencyHz,
                                       float amplitude)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const double time = static_cast<double> (i) / sampleRate;
        buffer.getWritePointer (0)[i] = amplitude * static_cast<float> (
            std::sin (kTwoPi * frequencyHz * time));
    }
    return buffer;
}

juce::AudioBuffer<float> makeHarmonicTone (
    double sampleRate,
    double durationSeconds,
    double frequencyHz,
    const std::vector<float>& harmonicAmplitudes)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    double phase = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        double sample = 0.0;
        for (size_t h = 0; h < harmonicAmplitudes.size(); ++h)
        {
            const double harmonic = static_cast<double> (h + 1);
            sample += static_cast<double> (harmonicAmplitudes[h])
                    * std::sin (harmonic * phase);
        }
        buffer.getWritePointer (0)[i] = static_cast<float> (sample);
        phase += kTwoPi * frequencyHz / sampleRate;
        if (phase > kTwoPi)
            phase -= kTwoPi;
    }
    return buffer;
}

juce::AudioBuffer<float> makeInharmonicTone (
    double sampleRate,
    double durationSeconds,
    double frequencyHz,
    double stretchPerPartial,
    const std::vector<float>& harmonicAmplitudes)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    std::vector<double> phases (harmonicAmplitudes.size(), 0.0);
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        double sample = 0.0;
        for (size_t h = 0; h < harmonicAmplitudes.size(); ++h)
        {
            const double harmonic = static_cast<double> (h + 1);
            const double partialFrequency = frequencyHz * harmonic
                * (1.0 + stretchPerPartial * static_cast<double> (h));
            sample += static_cast<double> (harmonicAmplitudes[h])
                    * std::sin (phases[h]);
            phases[h] += kTwoPi * partialFrequency / sampleRate;
        }
        buffer.getWritePointer (0)[i] = static_cast<float> (sample);
    }
    return buffer;
}

juce::AudioBuffer<float> makeVibratoTone (double sampleRate,
                                          double durationSeconds,
                                          double centreFrequencyHz,
                                          double depthSemitones,
                                          double rateHz,
                                          float amplitude)
{
    return makeVibratoToneWithPhase (sampleRate, durationSeconds,
                                     centreFrequencyHz, depthSemitones,
                                     rateHz, 0.0, amplitude);
}

juce::AudioBuffer<float> makeVibratoToneWithPhase (double sampleRate,
                                                   double durationSeconds,
                                                   double centreFrequencyHz,
                                                   double depthSemitones,
                                                   double rateHz,
                                                   double phaseRadians,
                                                   float amplitude)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    double phase = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const double time = static_cast<double> (i) / sampleRate;
        const double instantaneous = centreFrequencyHz
            * std::pow (2.0, depthSemitones
                             * std::sin (kTwoPi * rateHz * time + phaseRadians) / 12.0);
        buffer.getWritePointer (0)[i] = amplitude * static_cast<float> (std::sin (phase));
        phase += kTwoPi * instantaneous / sampleRate;
    }
    return buffer;
}

std::vector<ReferencePitchFrame> vibratoReferenceFrames (
    double durationSeconds,
    double hopSeconds,
    double centreFrequencyHz,
    double depthSemitones,
    double rateHz)
{
    return vibratoReferenceFramesWithPhase (durationSeconds, hopSeconds,
                                            centreFrequencyHz, depthSemitones,
                                            rateHz, 0.0);
}

std::vector<ReferencePitchFrame> vibratoReferenceFramesWithPhase (
    double durationSeconds,
    double hopSeconds,
    double centreFrequencyHz,
    double depthSemitones,
    double rateHz,
    double phaseRadians)
{
    return buildReferenceFrames (
        durationSeconds, hopSeconds,
        [=] (double time)
        {
            return centreFrequencyHz
                 * std::pow (2.0, depthSemitones
                                  * std::sin (kTwoPi * rateHz * time + phaseRadians) / 12.0);
        },
        [] (double) { return true; });
}

// Irregular vibrato trajectory, shared by the audio generator and the
// reference builder so they agree exactly. The vibrato phase is the
// integral of the drifting rate:
//   phase(t) = 2*pi*(5.5*t + 1.5*(1 - cos(2*pi*0.3*t)) / (2*pi*0.3))
namespace
{
double irregularVibratoFrequency (double time, double centreFrequencyHz)
{
    const double depth = 0.9 + 0.2 * std::sin (kTwoPi * 0.5 * time);
    const double phase = kTwoPi
        * (5.5 * time + 1.5 * (1.0 - std::cos (kTwoPi * 0.3 * time))
               / (kTwoPi * 0.3));
    return centreFrequencyHz
        * std::pow (2.0, depth * std::sin (phase) / 12.0);
}
} // namespace

juce::AudioBuffer<float> makeIrregularVibratoTone (double sampleRate,
                                                   double durationSeconds,
                                                   double centreFrequencyHz,
                                                   float amplitude)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    double phase = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const double time = static_cast<double> (i) / sampleRate;
        const double instantaneous = irregularVibratoFrequency (time, centreFrequencyHz);
        buffer.getWritePointer (0)[i] = amplitude * static_cast<float> (std::sin (phase));
        phase += kTwoPi * instantaneous / sampleRate;
    }
    return buffer;
}

std::vector<ReferencePitchFrame> irregularVibratoReferenceFrames (
    double durationSeconds,
    double hopSeconds,
    double centreFrequencyHz)
{
    return buildReferenceFrames (
        durationSeconds, hopSeconds,
        [=] (double time) { return irregularVibratoFrequency (time, centreFrequencyHz); },
        [] (double) { return true; });
}

juce::AudioBuffer<float> makeGlissandoTone (double sampleRate,
                                            double durationSeconds,
                                            double startFrequencyHz,
                                            double endFrequencyHz,
                                            float amplitude)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    double phase = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const double time = static_cast<double> (i) / sampleRate;
        const double fraction = durationSeconds > 0.0
            ? std::min (1.0, time / durationSeconds) : 0.0;
        const double instantaneous = startFrequencyHz
            + (endFrequencyHz - startFrequencyHz) * fraction;
        buffer.getWritePointer (0)[i] = amplitude * static_cast<float> (std::sin (phase));
        phase += kTwoPi * instantaneous / sampleRate;
    }
    return buffer;
}

std::vector<ReferencePitchFrame> glissandoReferenceFrames (
    double durationSeconds,
    double hopSeconds,
    double startFrequencyHz,
    double endFrequencyHz)
{
    return buildReferenceFrames (
        durationSeconds, hopSeconds,
        [=] (double time)
        {
            const double fraction = durationSeconds > 0.0
                ? std::min (1.0, time / durationSeconds) : 0.0;
            return startFrequencyHz + (endFrequencyHz - startFrequencyHz) * fraction;
        },
        [] (double) { return true; });
}

AudioTestCase makeNoteSequenceCase (const std::string& name,
                                    double sampleRate,
                                    double totalDurationSeconds,
                                    const std::vector<SequenceNoteSpec>& notes,
                                    double attackSeconds,
                                    double releaseSeconds)
{
    AudioTestCase testCase;
    testCase.name = name;
    testCase.sampleRate = sampleRate;
    testCase.sourceDescription = "synthetic note sequence (deterministic, "
                                 "raised-cosine attack/release)";

    const int totalSamples = static_cast<int> (
        std::llround (sampleRate * totalDurationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, totalSamples));
    buffer.clear();

    for (const auto& note : notes)
    {
        const int startSample = static_cast<int> (
            std::llround (note.startSeconds * sampleRate));
        const int noteSamples = static_cast<int> (
            std::llround (note.durationSeconds * sampleRate));
        const int attackSamples = static_cast<int> (
            std::llround (attackSeconds * sampleRate));
        const int releaseSamples = static_cast<int> (
            std::llround (releaseSeconds * sampleRate));

        for (int i = 0; i < noteSamples; ++i)
        {
            const int index = startSample + i;
            if (index < 0 || index >= buffer.getNumSamples())
                continue;
            const double localTime = static_cast<double> (i) / sampleRate;
            double envelope = 1.0;
            if (i < attackSamples && attackSamples > 0)
            {
                const double t = static_cast<double> (i) / attackSamples;
                envelope = 0.5 * (1.0 - std::cos (kTwoPi * 0.5 * t));
            }
            else if (i >= noteSamples - releaseSamples && releaseSamples > 0)
            {
                const double t = static_cast<double> (noteSamples - 1 - i) / releaseSamples;
                envelope = 0.5 * (1.0 - std::cos (kTwoPi * 0.5 * t));
            }
            buffer.getWritePointer (0)[index] = static_cast<float> (
                note.amplitude * envelope
                * std::sin (kTwoPi * note.frequencyHz * localTime));
        }

        ReferenceNote reference;
        reference.startSeconds = note.startSeconds;
        reference.endSeconds = note.startSeconds + note.durationSeconds;
        reference.frequencyHz = note.frequencyHz;
        reference.midiNote = static_cast<int> (
            std::lround (frequencyToMidiDouble (note.frequencyHz)));
        testCase.referenceNotes.push_back (reference);
    }

    testCase.audio = std::move (buffer);
    testCase.referenceFrames = buildReferenceFrames (
        totalDurationSeconds, kDefaultHopSeconds,
        [&] (double time)
        {
            for (const auto& note : notes)
                if (time >= note.startSeconds
                    && time < note.startSeconds + note.durationSeconds)
                    return note.frequencyHz;
            return 0.0;
        },
        [&] (double time)
        {
            for (const auto& note : notes)
                if (time >= note.startSeconds
                    && time < note.startSeconds + note.durationSeconds)
                    return true;
            return false;
        });
    return testCase;
}

juce::AudioBuffer<float> makeSilence (double sampleRate, double durationSeconds)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    buffer.clear();
    return buffer;
}

juce::AudioBuffer<float> makeWhiteNoise (double sampleRate,
                                         double durationSeconds,
                                         float amplitude,
                                         std::uint32_t seed)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    std::mt19937 rng (seed);
    std::uniform_real_distribution<float> uniform (-amplitude, amplitude);
    for (int i = 0; i < buffer.getNumSamples(); ++i)
        buffer.getWritePointer (0)[i] = uniform (rng);
    return buffer;
}

juce::AudioBuffer<float> makePinkNoise (double sampleRate,
                                        double durationSeconds,
                                        float amplitude,
                                        std::uint32_t seed)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    std::mt19937 rng (seed);
    std::uniform_real_distribution<float> uniform (-1.0f, 1.0f);

    // Paul Kellet pink-noise filter state.
    double b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0, b4 = 0.0, b5 = 0.0, b6 = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const double white = uniform (rng);
        b0 = 0.99886 * b0 + white * 0.0555179;
        b1 = 0.99332 * b1 + white * 0.0750759;
        b2 = 0.98063 * b2 + white * 0.153852;
        b3 = 0.95987 * b3 + white * 0.222144;
        b4 = 0.90011 * b4 + white * 0.316653;
        b5 = 0.76165 * b5 + white * 0.418583;
        b6 = 0.50006 * b6 + white * 0.259954;
        const double pink = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11;
        buffer.getWritePointer (0)[i] = static_cast<float> (pink);
    }

    // Normalise to the requested RMS so the level is exact, not assumed.
    const double rms = bufferRms (buffer);
    if (rms > 1.0e-12)
    {
        const float gain = static_cast<float> (static_cast<double> (amplitude) / rms);
        buffer.applyGain (gain);
    }
    return buffer;
}

juce::AudioBuffer<float> makeBreathLikeNoise (double sampleRate,
                                              double durationSeconds,
                                              float amplitude,
                                              std::uint32_t seed)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    std::mt19937 rng (seed);
    std::uniform_real_distribution<float> uniform (-1.0f, 1.0f);

    // One-pole low-pass at ~800 Hz, then a slow 0.15 Hz swell.
    const double alpha = 1.0 - std::exp (-kTwoPi * 800.0 / sampleRate);
    double lowPassed = 0.0;
    for (int i = 0; i < buffer.getNumSamples(); ++i)
    {
        const double time = static_cast<double> (i) / sampleRate;
        const double white = uniform (rng);
        lowPassed += alpha * (white - lowPassed);
        const double swell = 0.5 + 0.5 * std::sin (kTwoPi * 0.15 * time);
        buffer.getWritePointer (0)[i] = static_cast<float> (lowPassed * swell);
    }

    // Scale so the peak equals the requested amplitude.
    const float peak = bufferPeak (buffer);
    if (peak > 1.0e-12f)
        buffer.applyGain (amplitude / peak);
    return buffer;
}

juce::AudioBuffer<float> makeTransientBurst (double sampleRate,
                                             double totalDurationSeconds,
                                             double burstStartSeconds,
                                             double burstDurationSeconds,
                                             float amplitude,
                                             std::uint32_t seed)
{
    const int samples = static_cast<int> (std::llround (sampleRate * totalDurationSeconds));
    juce::AudioBuffer<float> buffer (1, std::max (1, samples));
    buffer.clear();
    std::mt19937 rng (seed);
    std::uniform_real_distribution<float> uniform (-amplitude, amplitude);
    const int start = static_cast<int> (std::llround (burstStartSeconds * sampleRate));
    const int count = static_cast<int> (std::llround (burstDurationSeconds * sampleRate));
    for (int i = 0; i < count; ++i)
    {
        const int index = start + i;
        if (index >= 0 && index < buffer.getNumSamples())
            buffer.getWritePointer (0)[index] = uniform (rng);
    }
    return buffer;
}

std::vector<ReferencePitchFrame> buildReferenceFrames (
    double durationSeconds,
    double hopSeconds,
    const std::function<double (double)>& frequencyAtTime,
    const std::function<bool (double)>& voicedAtTime)
{
    std::vector<ReferencePitchFrame> frames;
    if (durationSeconds <= 0.0 || hopSeconds <= 0.0)
        return frames;
    const int count = std::max (1, static_cast<int> (
        std::ceil (durationSeconds / hopSeconds - 1.0e-9)));
    frames.reserve (static_cast<size_t> (count));
    for (int i = 0; i < count; ++i)
    {
        ReferencePitchFrame frame;
        frame.timeSeconds = static_cast<double> (i) * hopSeconds;
        frame.voiced = voicedAtTime (frame.timeSeconds);
        frame.frequencyHz = frame.voiced ? frequencyAtTime (frame.timeSeconds) : 0.0;
        frames.push_back (frame);
    }
    return frames;
}

juce::AudioBuffer<float> mixToneWithNoise (const juce::AudioBuffer<float>& tone,
                                           double snrDb,
                                           std::uint32_t seed)
{
    juce::AudioBuffer<float> mixed (1, std::max (1, tone.getNumSamples()));
    mixed.clear();
    const double toneRms = bufferRms (tone);

    // Generate the noise bed at the tone's own length, then scale it to
    // the exact RMS implied by the target SNR.
    std::mt19937 rng (seed);
    std::uniform_real_distribution<float> uniform (-1.0f, 1.0f);
    for (int i = 0; i < mixed.getNumSamples(); ++i)
        mixed.getWritePointer (0)[i] = uniform (rng);
    const double noiseRms = bufferRms (mixed);
    const double targetNoiseRms = toneRms * std::pow (10.0, -snrDb / 20.0);
    if (noiseRms > 1.0e-12 && toneRms > 1.0e-12)
        mixed.applyGain (static_cast<float> (targetNoiseRms / noiseRms));

    // Add the tone on top of the scaled bed.
    const float* toneData = tone.getReadPointer (0);
    for (int i = 0; i < mixed.getNumSamples(); ++i)
        mixed.getWritePointer (0)[i] += toneData[i];
    return mixed;
}

// ---------------------------------------------------------------------------
// Metric computation
// ---------------------------------------------------------------------------

std::vector<PitchObservation> alignFrames (
    const PitchAnalysisResult& analysis,
    const std::vector<ReferencePitchFrame>& referenceFrames)
{
    std::vector<PitchObservation> observations;
    observations.reserve (referenceFrames.size());

    std::vector<double> analysisTimes;
    analysisTimes.reserve (analysis.frames.size());
    for (const auto& frame : analysis.frames)
        analysisTimes.push_back (frame.timeSeconds);

    for (const auto& reference : referenceFrames)
    {
        PitchObservation observation;
        observation.timeSeconds = reference.timeSeconds;
        observation.referenceFrequencyHz = reference.frequencyHz;
        observation.referenceVoiced = reference.voiced;

        if (! analysisTimes.empty())
        {
            // Nearest analysis frame by time (both grids are normally the
            // same 10 ms hop, so this is usually the same index).
            auto it = std::lower_bound (analysisTimes.begin(), analysisTimes.end(),
                                        reference.timeSeconds);
            size_t index = static_cast<size_t> (it - analysisTimes.begin());
            if (index > 0
                && (index >= analysisTimes.size()
                    || std::abs (analysisTimes[index - 1] - reference.timeSeconds)
                     < std::abs (analysisTimes[index] - reference.timeSeconds)))
                --index;
            const PitchFrame& frame = analysis.frames[index];
            observation.observedVoiced = frame.voiced;
            observation.observedFrequencyHz = frame.frequencyHz;
            observation.observedMidiPitch = frame.midiPitch;
            observation.observedConfidence = frame.confidence;
        }
        observations.push_back (observation);
    }
    return observations;
}

CentsErrorStats computePitchAccuracy (const std::vector<PitchObservation>& observations)
{
    CentsErrorStats stats;
    std::vector<double> absCents;
    int inTuneCount = 0;
    int inChromaCount = 0;

    for (const auto& observation : observations)
    {
        if (! observation.referenceVoiced)
            continue;
        ++stats.voicedReferenceFrames;

        if (! observation.observedVoiced
            || observation.observedFrequencyHz <= 0.0f
            || observation.referenceFrequencyHz <= 0.0)
        {
            // Missed frame: counts in the denominator, never in the
            // conditional statistics. Nothing is excluded.
            continue;
        }

        ++stats.detectedVoicedFrames;
        const double cents = centsError (observation.observedFrequencyHz,
                                         observation.referenceFrequencyHz);
        const double absValue = std::abs (cents);
        absCents.push_back (absValue);
        if (absValue <= kCentsTolerance)
            ++inTuneCount;
        if (std::abs (chromaCentsError (observation.observedFrequencyHz,
                                        observation.referenceFrequencyHz))
            <= kCentsTolerance)
            ++inChromaCount;

        const OctaveClassification octave = classifyOctaveError (cents);
        if (octave.isOctaveError)
        {
            ++stats.octaveBreakdown.total;
            if (octave.octaves == 1) (octave.up ? stats.octaveBreakdown.up1
                                                : stats.octaveBreakdown.down1)++;
            else if (octave.octaves == 2) (octave.up ? stats.octaveBreakdown.up2
                                                     : stats.octaveBreakdown.down2)++;
            else (octave.up ? stats.octaveBreakdown.up3
                            : stats.octaveBreakdown.down3)++;
        }
    }

    stats.missedVoicedFrames = stats.voicedReferenceFrames - stats.detectedVoicedFrames;
    if (stats.voicedReferenceFrames > 0)
    {
        stats.rawPitchAccuracy = static_cast<double> (inTuneCount)
                               / stats.voicedReferenceFrames;
        stats.rawChromaAccuracy = static_cast<double> (inChromaCount)
                                / stats.voicedReferenceFrames;
        stats.octaveErrorRate = static_cast<double> (stats.octaveBreakdown.total)
                              / stats.voicedReferenceFrames;
    }
    if (stats.detectedVoicedFrames > 0)
    {
        stats.hasConditionalStats = true;
        stats.meanAbsCents = meanOf (absCents);
        stats.medianAbsCents = medianOf (absCents);
        stats.conditionalRawPitchAccuracy = static_cast<double> (inTuneCount)
                                          / stats.detectedVoicedFrames;
    }
    return stats;
}

VoicingConfusion computeVoicing (const std::vector<PitchObservation>& observations)
{
    VoicingConfusion confusion;
    for (const auto& observation : observations)
    {
        if (observation.referenceVoiced && observation.observedVoiced)
            ++confusion.truePositives;
        else if (! observation.referenceVoiced && observation.observedVoiced)
            ++confusion.falsePositives;
        else if (! observation.referenceVoiced && ! observation.observedVoiced)
            ++confusion.trueNegatives;
        else
            ++confusion.falseNegatives;
    }

    const int predictedVoiced = confusion.truePositives + confusion.falsePositives;
    const int actualVoiced = confusion.truePositives + confusion.falseNegatives;
    if (predictedVoiced > 0)
    {
        confusion.precisionDefined = true;
        confusion.precision = static_cast<double> (confusion.truePositives) / predictedVoiced;
    }
    if (actualVoiced > 0)
    {
        confusion.recallDefined = true;
        confusion.recall = static_cast<double> (confusion.truePositives) / actualVoiced;
    }
    if (confusion.precisionDefined && confusion.recallDefined
        && (confusion.precision + confusion.recall) > 0.0)
    {
        confusion.f1Defined = true;
        confusion.f1 = 2.0 * confusion.precision * confusion.recall
                     / (confusion.precision + confusion.recall);
    }
    return confusion;
}

NoteMatchResult matchNotes (const MelodyNoteSequence& detected,
                            const std::vector<ReferenceNote>& reference,
                            double onsetToleranceSeconds)
{
    NoteMatchResult result;
    std::vector<bool> used (detected.notes.size(), false);
    std::vector<double> onsetErrorsMs, offsetErrorsMs, durationErrorsMs;

    for (const auto& referenceNote : reference)
    {
        int bestIndex = -1;
        double bestOnsetError = 0.0;
        for (size_t i = 0; i < detected.notes.size(); ++i)
        {
            if (used[i])
                continue;
            const MelodyNote& candidate = detected.notes[i];
            // Pitch agreement is exact integer equality: a semitone off
            // is a miss, not a match.
            if (candidate.midiNote != referenceNote.midiNote)
                continue;
            const double onsetError = std::abs (
                candidate.startSeconds - referenceNote.startSeconds);
            if (onsetError > onsetToleranceSeconds)
                continue;
            if (bestIndex < 0 || onsetError < bestOnsetError)
            {
                bestIndex = static_cast<int> (i);
                bestOnsetError = onsetError;
            }
        }

        if (bestIndex < 0)
        {
            ++result.missed;
            continue;
        }
        used[static_cast<size_t> (bestIndex)] = true;
        ++result.matched;
        const MelodyNote& matched = detected.notes[static_cast<size_t> (bestIndex)];
        onsetErrorsMs.push_back (std::abs (
            matched.startSeconds - referenceNote.startSeconds) * 1000.0);
        offsetErrorsMs.push_back (std::abs (
            matched.endSeconds - referenceNote.endSeconds) * 1000.0);
        durationErrorsMs.push_back (std::abs (
            matched.durationSeconds() - referenceNote.durationSeconds()) * 1000.0);
    }

    for (bool wasUsed : used)
        if (! wasUsed)
            ++result.falsePositives;

    const int predicted = result.matched + result.falsePositives;
    const int actual = result.matched + result.missed;
    if (predicted > 0)
    {
        result.precisionDefined = true;
        result.precision = static_cast<double> (result.matched) / predicted;
    }
    if (actual > 0)
    {
        result.recallDefined = true;
        result.recall = static_cast<double> (result.matched) / actual;
    }
    if (result.precisionDefined && result.recallDefined
        && (result.precision + result.recall) > 0.0)
    {
        result.f1Defined = true;
        result.f1 = 2.0 * result.precision * result.recall
                  / (result.precision + result.recall);
    }
    if (! onsetErrorsMs.empty())
    {
        result.meanAbsOnsetErrorMs = meanOf (onsetErrorsMs);
        result.medianAbsOnsetErrorMs = medianOf (onsetErrorsMs);
        result.meanAbsOffsetErrorMs = meanOf (offsetErrorsMs);
        result.medianAbsOffsetErrorMs = medianOf (offsetErrorsMs);
        result.meanAbsDurationErrorMs = meanOf (durationErrorsMs);
        result.medianAbsDurationErrorMs = medianOf (durationErrorsMs);
    }
    return result;
}

double percentileOfSorted (const std::vector<double>& sortedValues, double percentile)
{
    if (sortedValues.empty())
        return std::numeric_limits<double>::quiet_NaN();
    if (sortedValues.size() == 1)
        return sortedValues[0];
    const double clamped = std::clamp (percentile, 0.0, 100.0) / 100.0;
    const double rank = clamped * static_cast<double> (sortedValues.size() - 1);
    const size_t lower = static_cast<size_t> (std::floor (rank));
    const size_t upper = std::min (sortedValues.size() - 1, lower + 1);
    const double fraction = rank - static_cast<double> (lower);
    return sortedValues[lower] * (1.0 - fraction) + sortedValues[upper] * fraction;
}

// ---------------------------------------------------------------------------
// Short-note failure attribution
// ---------------------------------------------------------------------------

ShortNoteAttribution attributeShortNoteFailure (
    const PitchAnalysisResult& analysis,
    const ReferenceNote& referenceNote)
{
    ShortNoteAttribution attribution;
    if (analysis.hopSeconds <= 0.0 || referenceNote.frequencyHz <= 0.0)
    {
        // No frames to inspect: nothing proves the detector saw the pitch.
        attribution.stage = ShortNoteFailureStage::Detection;
        return attribution;
    }

    for (const auto& frame : analysis.frames)
    {
        if (frame.timeSeconds < referenceNote.startSeconds
            || frame.timeSeconds >= referenceNote.endSeconds)
            continue;
        ++attribution.expectedFrames;
        if (frame.voiced && frame.frequencyHz > 0.0f
            && std::abs (centsError (frame.frequencyHz, referenceNote.frequencyHz))
               <= kCentsTolerance)
            ++attribution.voicedFrames;
    }

    if (attribution.expectedFrames > 0)
        attribution.frameCoverage = static_cast<double> (attribution.voicedFrames)
                                  / attribution.expectedFrames;

    // The caller only attributes MISSED notes. Coverage >= 0.5 means the
    // detector saw the pitch and the segmenter dropped it; otherwise the
    // detector never produced the pitch.
    attribution.stage = attribution.frameCoverage >= 0.5
        ? ShortNoteFailureStage::Segmentation
        : ShortNoteFailureStage::Detection;
    return attribution;
}

// ---------------------------------------------------------------------------
// Level runners
// ---------------------------------------------------------------------------

LevelAResult runLevelA (const AudioTestCase& testCase,
                        const PitchAnalysisSettings& settings)
{
    LevelAResult result;
    result.analysis = MonophonicPitchAnalyzer::analyze (
        testCase.audio, testCase.sampleRate, settings);
    if (result.analysis != nullptr)
    {
        result.observations = alignFrames (*result.analysis, testCase.referenceFrames);
    }
    else
    {
        // analyze() returns nullptr only on cancellation or invalid input;
        // the observations then record every reference frame as missed.
        for (const auto& reference : testCase.referenceFrames)
        {
            PitchObservation observation;
            observation.timeSeconds = reference.timeSeconds;
            observation.referenceFrequencyHz = reference.frequencyHz;
            observation.referenceVoiced = reference.voiced;
            result.observations.push_back (observation);
        }
    }
    result.pitchStats = computePitchAccuracy (result.observations);
    result.voicing = computeVoicing (result.observations);
    return result;
}

LevelBResult runLevelB (const AudioTestCase& testCase,
                        const PitchAnalysisSettings& analysisSettings,
                        const NoteSegmentationSettings& segmentationSettings)
{
    LevelBResult result;
    result.levelA = runLevelA (testCase, analysisSettings);
    if (result.levelA.analysis != nullptr)
        result.notes = NoteSegmenter::segment (*result.levelA.analysis, segmentationSettings);
    else
        result.notes = std::make_shared<MelodyNoteSequence>();
    result.noteMatch = matchNotes (*result.notes, testCase.referenceNotes);
    return result;
}

namespace
{

MidiIntegrityReport checkMidiIntegrity (const juce::MidiFile& midiFile,
                                        const MelodyNoteSequence& sequence,
                                        const MidiExportSettings& requestedSettings)
{
    MidiIntegrityReport report;
    MidiExportSettings settings = requestedSettings;
    settings.bpm = std::clamp (settings.bpm, 20.0, 400.0);
    settings.channel = std::clamp (settings.channel, 1, 16);
    settings.ticksPerQuarterNote = std::clamp (settings.ticksPerQuarterNote, 96, 9600);
    const double ticksPerSecond = settings.bpm
        * static_cast<double> (settings.ticksPerQuarterNote) / 60.0;

    std::vector<const MelodyNote*> exportable;
    for (const auto& note : sequence.notes)
        if (note.endSeconds > note.startSeconds)
            exportable.push_back (&note);

    struct MidiEvent
    {
        int noteNumber = 0;
        int channel = 0;
        int velocity = 0;
        double timestamp = 0.0;
        bool noteOn = false;
    };
    std::vector<MidiEvent> noteOns, noteOffs;

    for (int trackIndex = 0; trackIndex < midiFile.getNumTracks(); ++trackIndex)
    {
        const auto* track = midiFile.getTrack (trackIndex);
        if (track == nullptr)
            continue;
        for (int i = 0; i < track->getNumEvents(); ++i)
        {
            const auto message = track->getEventPointer (i)->message;
            if (message.isTempoMetaEvent())
                report.hasTempoMeta = true;
            if (message.isEndOfTrackMetaEvent())
                report.hasEndOfTrack = true;
            if (message.isNoteOn())
                noteOns.push_back ({ message.getNoteNumber(), message.getChannel(),
                                     message.getVelocity(),
                                     message.getTimeStamp(), true });
            else if (message.isNoteOff())
                noteOffs.push_back ({ message.getNoteNumber(), message.getChannel(),
                                      message.getVelocity(),
                                      message.getTimeStamp(), false });
        }
    }

    report.noteOnCount = static_cast<int> (noteOns.size());
    report.noteOffCount = static_cast<int> (noteOffs.size());

    const size_t pairs = std::min (noteOns.size(), exportable.size());
    for (size_t i = 0; i < pairs; ++i)
    {
        const MelodyNote& expected = *exportable[i];
        if (noteOns[i].noteNumber != std::clamp (expected.midiNote, 0, 127))
            ++report.pitchMismatches;
        if (noteOns[i].channel != settings.channel)
            ++report.channelMismatches;
        if (noteOns[i].velocity < 1 || noteOns[i].velocity > 126)
            ++report.velocityOutOfRange;
        const double expectedStartTick = std::max (
            0.0, expected.startSeconds * ticksPerSecond);
        if (std::abs (noteOns[i].timestamp - expectedStartTick) > 1.0)
            ++report.timingErrorsTicks;
    }
    const size_t offPairs = std::min (noteOffs.size(), exportable.size());
    for (size_t i = 0; i < offPairs; ++i)
    {
        const MelodyNote& expected = *exportable[i];
        if (noteOffs[i].noteNumber != std::clamp (expected.midiNote, 0, 127))
            ++report.pitchMismatches;
        const double expectedEndTick = std::max (
            expected.startSeconds * ticksPerSecond + 1.0,
            expected.endSeconds * ticksPerSecond);
        if (std::abs (noteOffs[i].timestamp - expectedEndTick) > 1.0)
            ++report.timingErrorsTicks;
    }

    report.balanced = report.noteOnCount == report.noteOffCount
                   && report.noteOnCount == static_cast<int> (exportable.size());
    return report;
}

} // namespace

LevelCResult runLevelC (const AudioTestCase& testCase,
                        const PitchAnalysisSettings& analysisSettings,
                        const NoteSegmentationSettings& segmentationSettings,
                        const MidiExportSettings& midiSettings)
{
    LevelCResult result;
    result.levelB = runLevelB (testCase, analysisSettings, segmentationSettings);
    result.midiFile = MidiExporter::createMidiFile (*result.levelB.notes, midiSettings);
    result.integrity = checkMidiIntegrity (result.midiFile, *result.levelB.notes, midiSettings);
    return result;
}

// ---------------------------------------------------------------------------
// Real-audio dataset interface
// ---------------------------------------------------------------------------

bool CsvAnnotationDataset::loadManifest (const juce::File& manifestFile, std::string& error)
{
    error.clear();
    entries_.clear();
    manifestDirectory_ = manifestFile.getParentDirectory();

    if (! manifestFile.existsAsFile())
    {
        error = "manifest not found: " + manifestFile.getFullPathName().toStdString();
        return false;
    }

    const juce::String content = manifestFile.loadFileAsString();
    std::stringstream stream (content.toStdString());
    std::string line;
    bool headerSeen = false;
    int lineNumber = 0;
    while (std::getline (stream, line))
    {
        ++lineNumber;
        const std::string trimmed = trim (line);
        if (trimmed.empty())
            continue;
        const std::vector<std::string> fields = splitCsvLine (trimmed);
        if (! headerSeen)
        {
            headerSeen = true;
            if (fields.size() != 6 || fields[0] != "audioPath" || fields[1] != "framesPath"
                || fields[2] != "notesPath" || fields[3] != "instrument"
                || fields[4] != "license" || fields[5] != "datasetVersion")
            {
                error = "manifest header must be: audioPath,framesPath,notesPath,"
                        "instrument,license,datasetVersion (line 1)";
                return false;
            }
            continue;
        }
        if (fields.size() != 6)
        {
            error = "manifest line " + std::to_string (lineNumber)
                  + ": expected 6 fields, got " + std::to_string (fields.size());
            return false;
        }
        RealAudioDatasetEntry entry;
        entry.audioPath = fields[0];
        entry.framesPath = fields[1];
        entry.notesPath = fields[2];
        entry.instrument = fields[3];
        entry.license = fields[4];
        entry.datasetVersion = fields[5];
        entries_.push_back (entry);
    }

    if (entries_.empty())
    {
        error = "manifest has no entries: " + manifestFile.getFullPathName().toStdString();
        return false;
    }
    return true;
}

DatasetLoadResult CsvAnnotationDataset::loadCase (const RealAudioDatasetEntry& entry) const
{
    DatasetLoadResult result;

    // License first: never load audio whose license has not been checked.
    const std::string license = lowercased (trim (entry.license));
    if (license.empty() || license == "unknown")
    {
        result.error = "refusing entry with unverified license (audio: "
                     + entry.audioPath + ")";
        return result;
    }

    const auto resolve = [&] (const std::string& path)
    {
        if (juce::File::isAbsolutePath (juce::String (path)))
            return juce::File (juce::String (path));
        return manifestDirectory_.getChildFile (juce::String (path));
    };

    const juce::File audioFile = resolve (entry.audioPath);
    if (! audioFile.existsAsFile())
    {
        result.error = "audio file not found: " + audioFile.getFullPathName().toStdString();
        return result;
    }

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (
        formatManager.createReaderFor (audioFile));
    if (reader == nullptr)
    {
        result.error = "unsupported audio format: "
                     + audioFile.getFullPathName().toStdString();
        return result;
    }
    if (reader->numChannels < 1 || reader->numChannels > 2)
    {
        result.error = "only mono/stereo audio is supported (got "
                     + std::to_string (reader->numChannels) + " channels)";
        return result;
    }
    if (reader->lengthInSamples <= 0)
    {
        result.error = "audio file is empty: " + audioFile.getFullPathName().toStdString();
        return result;
    }

    juce::AudioBuffer<float> raw (static_cast<int> (reader->numChannels),
                                  static_cast<int> (reader->lengthInSamples));
    // Mono files have no right channel to read; stereo files read both.
    reader->read (&raw, 0, raw.getNumSamples(), 0, true, reader->numChannels > 1);

    AudioTestCase testCase;
    testCase.name = audioFile.getFileNameWithoutExtension().toStdString();
    testCase.sampleRate = reader->sampleRate;
    testCase.datasetVersion = entry.datasetVersion;
    testCase.sourceDescription = entry.instrument + " / " + entry.datasetVersion
                               + " / license: " + entry.license;
    if (raw.getNumChannels() == 1)
    {
        testCase.audio = raw;
    }
    else
    {
        AudioClip::buildMonoBuffer (raw, testCase.audio);
    }

    const auto parseFrames = [&] (const juce::File& file, std::string& parseError) -> bool
    {
        if (! file.existsAsFile())
        {
            parseError = "annotation file not found: "
                       + file.getFullPathName().toStdString();
            return false;
        }
        std::stringstream stream (file.loadFileAsString().toStdString());
        std::string csvLine;
        while (std::getline (stream, csvLine))
        {
            const std::string trimmed = trim (csvLine);
            if (trimmed.empty())
                continue;
            const std::vector<std::string> fields = splitCsvLine (trimmed);
            if (fields.size() != 3)
            {
                parseError = "frames CSV line must have 3 fields (timeSeconds,"
                             "frequencyHz,voiced): " + trimmed;
                return false;
            }
            ReferencePitchFrame frame;
            if (! parseDoubleField (fields[0], frame.timeSeconds)
                || ! parseDoubleField (fields[1], frame.frequencyHz)
                || (fields[2] != "0" && fields[2] != "1"))
            {
                parseError = "frames CSV line is not numeric: " + trimmed;
                return false;
            }
            frame.voiced = fields[2] == "1";
            testCase.referenceFrames.push_back (frame);
        }
        return true;
    };

    const auto parseNotes = [&] (const juce::File& file, std::string& parseError) -> bool
    {
        if (! file.existsAsFile())
        {
            parseError = "annotation file not found: "
                       + file.getFullPathName().toStdString();
            return false;
        }
        std::stringstream stream (file.loadFileAsString().toStdString());
        std::string csvLine;
        while (std::getline (stream, csvLine))
        {
            const std::string trimmed = trim (csvLine);
            if (trimmed.empty())
                continue;
            const std::vector<std::string> fields = splitCsvLine (trimmed);
            if (fields.size() != 3)
            {
                parseError = "notes CSV line must have 3 fields (startSeconds,"
                             "endSeconds,midiNote): " + trimmed;
                return false;
            }
            ReferenceNote note;
            double midi = 0.0;
            if (! parseDoubleField (fields[0], note.startSeconds)
                || ! parseDoubleField (fields[1], note.endSeconds)
                || ! parseDoubleField (fields[2], midi))
            {
                parseError = "notes CSV line is not numeric: " + trimmed;
                return false;
            }
            note.midiNote = static_cast<int> (std::lround (midi));
            note.frequencyHz = midiToFrequencyDouble (midi);
            testCase.referenceNotes.push_back (note);
        }
        return true;
    };

    std::string parseError;
    if (! parseFrames (resolve (entry.framesPath), parseError)
        || ! parseNotes (resolve (entry.notesPath), parseError))
    {
        result.error = parseError;
        return result;
    }

    result.ok = true;
    result.testCase = std::move (testCase);
    return result;
}

// ---------------------------------------------------------------------------
// Performance benchmarking
// ---------------------------------------------------------------------------

TimingSummary benchmarkStage (const std::function<void()>& stage,
                              double audioDurationSeconds,
                              int warmupIterations,
                              int measuredIterations)
{
    TimingSummary summary;
    summary.audioDurationSeconds = audioDurationSeconds;
    summary.iterations = std::max (0, measuredIterations);
    if (measuredIterations <= 0)
        return summary;

    for (int i = 0; i < std::max (0, warmupIterations); ++i)
        stage();

    std::vector<double> timesMs;
    timesMs.reserve (static_cast<size_t> (measuredIterations));
    for (int i = 0; i < measuredIterations; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        stage();
        const auto end = std::chrono::steady_clock::now();
        timesMs.push_back (std::chrono::duration<double, std::milli> (end - start).count());
    }

    std::vector<double> sorted = timesMs;
    std::sort (sorted.begin(), sorted.end());
    summary.meanMs = meanOf (timesMs);
    summary.medianMs = medianOf (timesMs);
    summary.p95Ms = percentileOfSorted (sorted, 95.0);
    summary.minMs = sorted.front();
    summary.maxMs = sorted.back();
    if (summary.medianMs > 0.0)
        summary.realTimeFactor = audioDurationSeconds / (summary.medianMs / 1000.0);
    else
        summary.realTimeFactor = std::numeric_limits<double>::infinity();
    return summary;
}

// ---------------------------------------------------------------------------
// Reproducibility record
// ---------------------------------------------------------------------------

std::string BenchmarkEnvironment::toString() const
{
    std::ostringstream stream;
    stream << "commit: " << (commitSha.empty() ? "(see report — not embedded)" : commitSha)
           << "\nbuild: " << buildConfig
           << "\ncompiler: " << compiler
           << "\nos: " << operatingSystem
           << "\ncpu: " << cpuModel
           << "\ntimestamp: " << timestampIso
           << "\ndataset: " << datasetVersion
           << "\nsampleRate: " << sampleRate
           << "\nbufferSize: " << bufferSize << " (not applicable to offline analysis)"
           << "\nanalysis: " << analysisConfigSummary
           << "\nsegmentation: " << segmentationConfigSummary;
    return stream.str();
}

BenchmarkEnvironment collectEnvironment (
    double sampleRate,
    const std::string& datasetVersion,
    const PitchAnalysisSettings& analysisSettings,
    const NoteSegmentationSettings& segmentationSettings)
{
    BenchmarkEnvironment env;
    env.commitSha = {}; // filled in by the agent-written report
#if defined (JUCE_DEBUG) && JUCE_DEBUG
    env.buildConfig = "Debug";
#else
    env.buildConfig = "Release";
#endif
#if defined (__clang__)
    env.compiler = std::string ("Clang ") + __clang_version__;
#elif defined (__GNUC__)
    env.compiler = std::string ("GCC ") + __VERSION__;
#elif defined (_MSC_VER)
    env.compiler = "MSVC " + std::to_string (_MSC_VER);
#else
    env.compiler = "unknown compiler";
#endif
#if JUCE_LINUX
    env.operatingSystem = "Linux";
#elif JUCE_WINDOWS
    env.operatingSystem = "Windows";
#elif JUCE_MAC
    env.operatingSystem = "macOS";
#else
    env.operatingSystem = "unknown OS";
#endif

    env.cpuModel = "unknown";
#if JUCE_LINUX
    {
        std::ifstream cpuinfo ("/proc/cpuinfo");
        std::string cpuLine;
        while (std::getline (cpuinfo, cpuLine))
        {
            if (cpuLine.rfind ("model name", 0) == 0)
            {
                const size_t colon = cpuLine.find (':');
                if (colon != std::string::npos)
                {
                    env.cpuModel = trim (cpuLine.substr (colon + 1));
                    break;
                }
            }
        }
    }
#endif

    const std::time_t now = std::time (nullptr);
    const std::tm* local = std::localtime (&now);
    char timeBuffer[32] = {};
    if (local != nullptr)
        std::strftime (timeBuffer, sizeof (timeBuffer), "%Y-%m-%dT%H:%M:%S", local);
    env.timestampIso = timeBuffer;
    env.datasetVersion = datasetVersion;
    env.sampleRate = sampleRate;
    env.bufferSize = 0;

    std::ostringstream analysisSummary;
    analysisSummary << "analysisRate=" << analysisSettings.analysisSampleRate
                    << " hop=" << analysisSettings.hopSeconds << "s"
                    << " minF=" << analysisSettings.minimumFrequencyHz << "Hz"
                    << " maxF=" << analysisSettings.maximumFrequencyHz << "Hz"
                    << " silence=" << analysisSettings.silenceThresholdDb << "dB"
                    << " minConf=" << analysisSettings.minimumConfidence
                    << " yin=" << analysisSettings.yinThreshold;
    env.analysisConfigSummary = analysisSummary.str();

    std::ostringstream segmentationSummary;
    segmentationSummary << "minNote=" << segmentationSettings.minimumNoteSeconds << "s"
                        << " gap=" << segmentationSettings.maximumGapSeconds << "s"
                        << " hysteresis=" << segmentationSettings.pitchHysteresisSemitones
                        << "st";
    env.segmentationConfigSummary = segmentationSummary.str();
    return env;
}

} // namespace pitchbenchmark
