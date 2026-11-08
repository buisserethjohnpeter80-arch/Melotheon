// Melotheon Phase 3C unit tests.
//
// Console app, no GUI, no host, no MessageManager: every test calls
// the audio-foundation classes (and the processor) directly. The
// file loader is exercised through its synchronous API; the async
// wrapper only adds a worker thread + a message-thread callback.
//
// Exit code 0 = all checks passed, 1 = at least one failure.

#include <JuceHeader.h>

#include "Audio/AudioCaptureEngine.h"
#include "Audio/AudioClip.h"
#include "Audio/AudioFileLoader.h"
#include "Audio/AudioPreviewEngine.h"
#include "Audio/SampleInstrumentRenderer.h"
#include "Analysis/MelodyEditor.h"
#include "Analysis/MonophonicPitchAnalyzer.h"
#include "Analysis/NoteSegmenter.h"
#include "Analysis/PitchAnalysis.h"
#include "Analysis/TempoGrid.h"
#include "Midi/MidiExporter.h"
#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "Project/ProjectState.h"
#include "UI/PitchEditorComponent.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace
{

int checks = 0;
int failures = 0;

void check (bool condition, const char* expression, int line)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cout << "FAIL (line " << line << "): " << expression << "\n";
    }
}

bool near (float a, float b, float epsilon)
{
    return std::abs (a - b) <= epsilon;
}

#define CHECK(cond) check ((cond), #cond, __LINE__)
#define CHECK_NEAR(a, b, eps) check (near ((a), (b), (eps)), #a " ~= " #b, __LINE__)

// ---- AudioClip: stereo-to-mono, mono copy, metadata ----

void testStereoToMono()
{
    juce::AudioBuffer<float> source (2, 4);
    source.getWritePointer (0)[0] = 0.6f;
    source.getWritePointer (0)[1] = -0.2f;
    source.getWritePointer (0)[2] = 0.0f;
    source.getWritePointer (0)[3] = 1.0f;
    source.getWritePointer (1)[0] = 0.2f;
    source.getWritePointer (1)[1] = 0.4f;
    source.getWritePointer (1)[2] = 0.0f;
    source.getWritePointer (1)[3] = 0.5f;

    juce::AudioBuffer<float> mono;
    AudioClip::buildMonoBuffer (source, mono);

    CHECK (mono.getNumChannels() == 1);
    CHECK (mono.getNumSamples() == 4);
    CHECK_NEAR (mono.getReadPointer (0)[0], 0.4f, 1e-6f);  // 0.5 * (0.6 + 0.2)
    CHECK_NEAR (mono.getReadPointer (0)[1], 0.1f, 1e-6f);  // 0.5 * (-0.2 + 0.4)
    CHECK_NEAR (mono.getReadPointer (0)[2], 0.0f, 1e-6f);
    CHECK_NEAR (mono.getReadPointer (0)[3], 0.75f, 1e-6f); // 0.5 * (1.0 + 0.5)
}

void testStereoPhaseCancellationProtection()
{
    juce::AudioBuffer<float> source (2, 32);
    for (int i = 0; i < source.getNumSamples(); ++i)
    {
        const float sample = std::sin (2.0f * 3.14159265f
                                       * static_cast<float> (i) / 8.0f);
        source.getWritePointer (0)[i] = sample;
        source.getWritePointer (1)[i] = -sample;
    }

    juce::AudioBuffer<float> mono;
    AudioClip::buildMonoBuffer (source, mono);

    // A normal L+R average would be complete silence. Phase 3A keeps
    // one channel for analysis while preview remains original stereo.
    for (int i = 0; i < source.getNumSamples(); ++i)
        CHECK_NEAR (mono.getReadPointer (0)[i], source.getReadPointer (0)[i], 1e-6f);
}

void testMonoCopy()
{
    juce::AudioBuffer<float> source (1, 3);
    source.getWritePointer (0)[0] = 0.5f;
    source.getWritePointer (0)[1] = -0.5f;
    source.getWritePointer (0)[2] = 0.25f;

    juce::AudioBuffer<float> mono;
    AudioClip::buildMonoBuffer (source, mono);

    CHECK (mono.getNumChannels() == 1);
    CHECK_NEAR (mono.getReadPointer (0)[0], 0.5f, 1e-6f);
    CHECK_NEAR (mono.getReadPointer (0)[1], -0.5f, 1e-6f);
    CHECK_NEAR (mono.getReadPointer (0)[2], 0.25f, 1e-6f);

    // The preview buffer must keep the original channels untouched.
    auto clip = AudioClip::fromBuffer (source, 48000.0, "mono", false);
    CHECK (clip != nullptr);
    CHECK (clip->previewBuffer.getNumChannels() == 1);
    CHECK_NEAR (clip->previewBuffer.getReadPointer (0)[1], -0.5f, 1e-6f);
}

void testAudioClipMetadata()
{
    juce::AudioBuffer<float> source (2, 48000);
    source.clear();
    auto clip = AudioClip::fromBuffer (source, 48000.0, "take.wav", true);

    CHECK (clip != nullptr);
    CHECK (clip->sourceSampleRate == 48000.0);
    CHECK (clip->totalSamples == 48000);
    CHECK_NEAR ((float) clip->durationSeconds, 1.0f, 1e-6f);
    CHECK (clip->sourceName == "take.wav");
    CHECK (clip->importedFromFile == true);
    CHECK (clip->previewBuffer.getNumChannels() == 2);
    CHECK (clip->analysisMonoBuffer.getNumChannels() == 1);
    CHECK (! clip->isEmpty());

    // Invalid sample rate -> no clip.
    CHECK (AudioClip::fromBuffer (source, 0.0, "bad", false) == nullptr);

    // Empty source -> an empty clip (valid object, zero length).
    juce::AudioBuffer<float> empty (2, 0);
    auto emptyClip = AudioClip::fromBuffer (empty, 44100.0, "empty", false);
    CHECK (emptyClip != nullptr);
    CHECK (emptyClip->isEmpty());
    CHECK (emptyClip->durationSeconds == 0.0);
}

// ---- AudioCaptureEngine: limits, duration, state transitions ----

void testCaptureStateTransitions()
{
    AudioCaptureEngine engine;
    engine.prepare (44100.0, 2);

    CHECK (! engine.isRecording());
    CHECK (engine.stopRecording() == -1); // nothing recording

    CHECK (engine.startRecording());
    CHECK (engine.isRecording());
    CHECK (! engine.startRecording()); // already recording: no reset

    juce::AudioBuffer<float> block (2, 512);
    block.clear();
    engine.processInput (block);
    CHECK (engine.getRecordedSamples() == 512);

    const int recorded = engine.stopRecording();
    CHECK (recorded == 512);
    CHECK (! engine.isRecording());
    CHECK (engine.stopRecording() == -1);

    // A new take after stopping starts from zero again.
    CHECK (engine.startRecording());
    engine.processInput (block);
    CHECK (engine.getRecordedSamples() == 512);
    CHECK (engine.stopRecording() == 512);

    // Input while idle is ignored (no write position movement).
    engine.processInput (block);
    CHECK (engine.getRecordedSamples() == 512);

    // An armed take with no audio callback is empty and must never
    // publish a zero-length AudioClip.
    CHECK (engine.startRecording());
    CHECK (engine.stopRecording() == 0);
    CHECK (engine.createClip ("empty take") == nullptr);
    engine.processInput (block);
    CHECK (engine.getRecordedSamples() == 0);
}

void testCaptureMaxDurationAndLimit()
{
    AudioCaptureEngine engine;
    engine.prepare (44100.0, 2);
    CHECK (engine.startRecording());

    // Feed ~70 s of audio in 512-sample blocks: 60 s max = 2,646,000.
    juce::AudioBuffer<float> block (2, 512);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 512; ++i)
            block.getWritePointer (ch)[i] = 0.25f;

    const int blocksFor70Seconds = static_cast<int> (70.0 * 44100.0 / 512.0) + 1;
    for (int b = 0; b < blocksFor70Seconds; ++b)
        engine.processInput (block);

    const int maxSamples = static_cast<int> (60.0 * 44100.0);
    CHECK (engine.getRecordedSamples() == maxSamples); // clamped, never past
    CHECK (engine.limitReached());
    CHECK (! engine.isRecording()); // stopped safely at the limit

    // Further input after the limit is ignored.
    engine.processInput (block);
    CHECK (engine.getRecordedSamples() == maxSamples);

    auto clip = engine.createClip ("limit");
    CHECK (clip != nullptr);
    CHECK (clip->totalSamples == maxSamples);
    CHECK_NEAR ((float) clip->durationSeconds, 60.0f, 0.01f);
    CHECK_NEAR (clip->previewBuffer.getReadPointer (0)[1000], 0.25f, 1e-6f);
    CHECK (clip->importedFromFile == false);

    // After finalizing (stop), a new take works again.
    CHECK (engine.stopRecording() == maxSamples);
    engine.clearLimitFlag();
    CHECK (! engine.limitReached());
    CHECK (engine.startRecording());
    engine.stopRecording();
}

// ---- AudioPreviewEngine: state transitions, position, resampling ----

std::shared_ptr<AudioClip> makeTestClip (double sampleRate, int numSamples)
{
    juce::AudioBuffer<float> source (1, numSamples);
    for (int i = 0; i < numSamples; ++i)
        source.getWritePointer (0)[i] = static_cast<float> (i) / static_cast<float> (numSamples);
    return AudioClip::fromBuffer (source, sampleRate, "test", false);
}

void testPreviewStateTransitions()
{
    AudioPreviewEngine engine;

    CHECK (! engine.hasClip());
    CHECK (! engine.start()); // no clip
    CHECK (! engine.isPlaying());
    CHECK (engine.getDurationSeconds() == 0.0);
    CHECK (engine.getPositionSeconds() == 0.0);

    auto clip = makeTestClip (44100.0, 44100); // 1 s mono
    engine.setClip (clip);
    CHECK (engine.hasClip());
    CHECK_NEAR ((float) engine.getDurationSeconds(), 1.0f, 1e-6f);

    CHECK (engine.start());
    CHECK (engine.isPlaying());

    // Render one block at the clip's own rate: position advances
    // exactly one block.
    juce::AudioBuffer<float> output (2, 512);
    output.clear();
    engine.render (output, 44100.0);
    CHECK (engine.isPlaying());
    CHECK_NEAR ((float) engine.getPositionSeconds(), 512.0f / 44100.0f, 1e-4f);
    // Same rate: output equals the clip samples (stereo duplicated).
    CHECK_NEAR (output.getReadPointer (0)[0], 0.0f, 1e-6f);
    CHECK_NEAR (output.getReadPointer (1)[10], 10.0f / 44100.0f, 1e-6f);

    // Pause keeps the position; stop resets it.
    engine.pause();
    CHECK (! engine.isPlaying());
    CHECK (engine.getPositionSeconds() > 0.0);
    CHECK (engine.start()); // resumes from the kept position
    engine.stop();
    CHECK (! engine.isPlaying());
    CHECK (engine.getPositionSeconds() == 0.0);

    // Rendering past the end stops playback and flags the end.
    // (The end is detected when the next block starts, so render one
    // extra block after the exact 1 s boundary.)
    CHECK (engine.start());
    juce::AudioBuffer<float> big (2, 44100);
    engine.render (big, 44100.0); // exactly 1 s: position hits the end
    CHECK (engine.isPlaying());
    juce::AudioBuffer<float> tail (2, 64);
    engine.render (tail, 44100.0); // past the end: stops here
    CHECK (! engine.isPlaying());
    CHECK (engine.consumePlaybackEnded());
    CHECK (! engine.consumePlaybackEnded()); // consumed once
    CHECK (engine.getPositionSeconds() == 0.0);

    // A new clip resets the position.
    engine.setClip (makeTestClip (44100.0, 22050));
    CHECK (engine.getPositionSeconds() == 0.0);
    CHECK_NEAR ((float) engine.getDurationSeconds(), 0.5f, 1e-6f);
}

void testPreviewResampling()
{
    AudioPreviewEngine engine;
    // Clip at 48 kHz, host at 44.1 kHz: 4410 output samples must
    // advance the clip by 4800 samples (0.1 s), not 4410.
    engine.setClip (makeTestClip (48000.0, 48000));
    CHECK (engine.start());

    juce::AudioBuffer<float> output (2, 4410);
    engine.render (output, 44100.0);
    CHECK (engine.isPlaying());
    CHECK_NEAR ((float) engine.getPositionSeconds(), 0.1f, 0.001f);
}

void testPreviewSeek()
{
    AudioPreviewEngine engine;

    // No clip: seeking is a harmless no-op.
    engine.setPositionSeconds (0.5);
    CHECK (engine.getPositionSeconds() == 0.0);

    auto clip = makeTestClip (44100.0, 44100); // 1 s mono ramp
    engine.setClip (clip);

    // Seeking clamps to the clip range.
    engine.setPositionSeconds (-1.0);
    CHECK (engine.getPositionSeconds() == 0.0);
    engine.setPositionSeconds (2.0);
    CHECK_NEAR ((float) engine.getPositionSeconds(), 1.0f, 1e-6f);
    engine.setPositionSeconds (0.25);
    CHECK_NEAR ((float) engine.getPositionSeconds(), 0.25f, 1e-6f);

    // Seeking while playing: playback continues from the new
    // position and the rendered samples come from there.
    CHECK (engine.start());
    juce::AudioBuffer<float> output (2, 512);
    output.clear();
    engine.render (output, 44100.0);
    CHECK (engine.isPlaying());
    engine.setPositionSeconds (0.5);
    CHECK_NEAR ((float) engine.getPositionSeconds(), 0.5f, 1e-4f);
    output.clear();
    engine.render (output, 44100.0);
    CHECK (engine.isPlaying());
    CHECK_NEAR ((float) engine.getPositionSeconds(), 0.5f + 512.0f / 44100.0f, 1e-4f);
    // Ramp value at 0.5 s is 0.5.
    CHECK_NEAR (output.getReadPointer (0)[0], 0.5f, 1e-4f);

    engine.stop();
    CHECK (engine.getPositionSeconds() == 0.0);
}

// ---- AudioFileLoader: rejection + valid WAV round-trip ----

juce::File getTestDir()
{
    juce::File dir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                         .getChildFile ("MelotheonTests");
    dir.createDirectory();
    return dir;
}

void testFileLoaderRejection()
{
    AudioFileLoader loader;
    const juce::File dir = getTestDir();

    // Garbage bytes with a .wav extension.
    juce::File garbage = dir.getChildFile ("garbage.wav");
    garbage.replaceWithText ("this is not audio data at all, just text");
    auto result = loader.loadFileSync (garbage);
    CHECK (! result.succeeded());
    CHECK (result.errorMessage == "UNSUPPORTED FILE");

    // Unknown extension.
    juce::File unknown = dir.getChildFile ("weird.xyz");
    unknown.replaceWithText ("garbage");
    result = loader.loadFileSync (unknown);
    CHECK (! result.succeeded());
    CHECK (result.errorMessage == "UNSUPPORTED FILE");

    // Empty file.
    juce::File empty = dir.getChildFile ("empty.wav");
    empty.replaceWithText ("");
    result = loader.loadFileSync (empty);
    CHECK (! result.succeeded());
    CHECK (result.errorMessage == "FILE EMPTY");

    // Missing file.
    result = loader.loadFileSync (dir.getChildFile ("does-not-exist.wav"));
    CHECK (! result.succeeded());
    CHECK (result.errorMessage == "FILE NOT FOUND");
}

void testFileLoaderValidWav()
{
    const juce::File dir = getTestDir();
    juce::File wavFile = dir.getChildFile ("valid.wav");

    // Write a 1 s stereo 44.1 kHz WAV with the JUCE writer.
    juce::AudioBuffer<float> source (2, 44100);
    for (int i = 0; i < 44100; ++i)
    {
        const float s = 0.5f * std::sin (2.0f * 3.14159265f * 440.0f
                                         * static_cast<float> (i) / 44100.0f);
        source.getWritePointer (0)[i] = s;
        source.getWritePointer (1)[i] = -s;
    }
    {
        juce::WavAudioFormat format;
        // createWriterFor takes ownership of the stream: release the
        // unique_ptr into it and let the writer delete it.
        std::unique_ptr<juce::FileOutputStream> stream (wavFile.createOutputStream());
        CHECK (stream != nullptr);
        if (stream == nullptr)
            return;
        std::unique_ptr<juce::AudioFormatWriter> writer (
            format.createWriterFor (stream.release(), 44100.0, 2, 16, {}, 0));
        CHECK (writer != nullptr);
        if (writer == nullptr)
            return;
        CHECK (writer->writeFromAudioSampleBuffer (source, 0, 44100));
        writer.reset(); // flushes and closes the file
    }

    AudioFileLoader loader;
    auto result = loader.loadFileSync (wavFile);
    CHECK (result.succeeded());
    CHECK (result.errorMessage.isEmpty());
    CHECK (result.clip != nullptr);
    CHECK (result.clip->sourceSampleRate == 44100.0);
    CHECK (result.clip->totalSamples == 44100);
    CHECK_NEAR ((float) result.clip->durationSeconds, 1.0f, 1e-6f);
    CHECK (result.clip->previewBuffer.getNumChannels() == 2);
    CHECK (result.clip->importedFromFile == true);
    CHECK (result.clip->sourceName == "valid.wav");
    // 16-bit quantization: the round-trip is approximate.
    CHECK_NEAR (result.clip->previewBuffer.getReadPointer (0)[441], source.getReadPointer (0)[441], 0.001f);
    CHECK (result.clip->analysisMonoBuffer.getMagnitude (0, 0, 44100) > 0.4f);
}

// ---- Phase 3A: offline monophonic pitch analysis ----

juce::AudioBuffer<float> makeSine (double sampleRate, double durationSeconds,
                                   float frequency, float amplitude = 0.5f)
{
    const int samples = static_cast<int> (std::llround (sampleRate * durationSeconds));
    juce::AudioBuffer<float> buffer (1, samples);
    for (int i = 0; i < samples; ++i)
        buffer.getWritePointer (0)[i] = amplitude * std::sin (
            2.0f * 3.14159265f * frequency * static_cast<float> (i)
            / static_cast<float> (sampleRate));
    return buffer;
}

void testPitchAnalysisAtSourceRates()
{
    for (double sampleRate : { 44100.0, 48000.0, 96000.0 })
    {
        auto sine = makeSine (sampleRate, 0.6, 440.0f);
        auto result = MonophonicPitchAnalyzer::analyze (sine, sampleRate);

        CHECK (result != nullptr);
        if (result == nullptr)
            continue;
        CHECK (result->hasPitch());
        CHECK (result->voicedFrameCount > 35);
        CHECK_NEAR (result->medianFrequencyHz, 440.0f, 3.0f);
        CHECK (result->medianConfidence > 0.75f);
        CHECK_NEAR ((float) result->hopSeconds, 0.01f, 1e-6f);
    }
}

void testLowAndHighPitchAnalysis()
{
    auto lowSine = makeSine (48000.0, 0.7, 110.0f);
    auto low = MonophonicPitchAnalyzer::analyze (lowSine, 48000.0);
    CHECK (low != nullptr);
    CHECK (low != nullptr && low->hasPitch());
    if (low != nullptr && low->hasPitch())
        CHECK_NEAR (low->medianFrequencyHz, 110.0f, 2.0f);

    auto whistleSine = makeSine (48000.0, 0.5, 1200.0f, 0.35f);
    auto whistle = MonophonicPitchAnalyzer::analyze (whistleSine, 48000.0);
    CHECK (whistle != nullptr);
    CHECK (whistle != nullptr && whistle->hasPitch());
    if (whistle != nullptr && whistle->hasPitch())
        CHECK_NEAR (whistle->medianFrequencyHz, 1200.0f, 12.0f);
}

void testPitchAnalysisSilenceAndCancellation()
{
    juce::AudioBuffer<float> silence (1, 24000);
    silence.clear();
    auto result = MonophonicPitchAnalyzer::analyze (silence, 24000.0);
    CHECK (result != nullptr);
    CHECK (result != nullptr && ! result->hasPitch());
    CHECK (result != nullptr && result->voicedFrameCount == 0);

    auto sine = makeSine (44100.0, 0.5, 220.0f);
    auto cancelled = MonophonicPitchAnalyzer::analyze (
        sine, 44100.0, {}, [] { return true; });
    CHECK (cancelled == nullptr);
}

void testPitchCurveFollowsTwoTones()
{
    constexpr double sampleRate = 48000.0;
    juce::AudioBuffer<float> source (1, static_cast<int> (sampleRate));
    source.clear();
    for (int i = 0; i < source.getNumSamples(); ++i)
    {
        const double time = static_cast<double> (i) / sampleRate;
        float frequency = 0.0f;
        if (time < 0.35)
            frequency = 220.0f;
        else if (time >= 0.50 && time < 0.90)
            frequency = 440.0f;
        if (frequency > 0.0f)
            source.getWritePointer (0)[i] = 0.45f * std::sin (
                2.0f * 3.14159265f * frequency * static_cast<float> (time));
    }

    auto result = MonophonicPitchAnalyzer::analyze (source, sampleRate);
    CHECK (result != nullptr);
    CHECK (result != nullptr && result->hasPitch());
    if (result == nullptr)
        return;

    const PitchFrame* first = result->findNearestVoicedFrame (0.20);
    const PitchFrame* second = result->findNearestVoicedFrame (0.70);
    CHECK (first != nullptr);
    CHECK (second != nullptr);
    if (first != nullptr)
        CHECK_NEAR (first->frequencyHz, 220.0f, 5.0f);
    if (second != nullptr)
        CHECK_NEAR (second->frequencyHz, 440.0f, 7.0f);

    CHECK (result->findNearestVoicedFrame (0.425, 0.03) == nullptr);
}

// ---- Phase 3C: stable notes, MIDI export, sample instrument ----

void testNoteSegmentationRejectsVibratoAndBridgesGap()
{
    PitchAnalysisResult analysis;
    analysis.durationSeconds = 1.0;
    analysis.hopSeconds = 0.01;
    analysis.frames.resize (100);

    for (int i = 0; i < 100; ++i)
    {
        auto& frame = analysis.frames[static_cast<size_t> (i)];
        frame.timeSeconds = static_cast<double> (i) * analysis.hopSeconds;
        frame.rmsDb = -18.0f;
        frame.confidence = 0.92f;
        frame.voiced = true;

        // A4 with ordinary vocal vibrato, a 30 ms detector gap, then C5.
        if (i < 50)
            frame.midiPitch = 69.0f + 0.28f * std::sin (static_cast<float> (i) * 0.7f);
        else
            frame.midiPitch = 72.0f + 0.22f * std::sin (static_cast<float> (i) * 0.6f);
        frame.frequencyHz = 440.0f * std::pow (
            2.0f, (frame.midiPitch - 69.0f) / 12.0f);
        ++analysis.voicedFrameCount;
    }
    for (int i = 24; i < 27; ++i)
    {
        auto& frame = analysis.frames[static_cast<size_t> (i)];
        frame.voiced = false;
        frame.midiPitch = 0.0f;
        frame.frequencyHz = 0.0f;
        --analysis.voicedFrameCount;
    }

    auto notes = NoteSegmenter::segment (analysis);
    CHECK (notes != nullptr);
    CHECK (notes != nullptr && notes->notes.size() == 2);
    if (notes == nullptr || notes->notes.size() != 2)
        return;

    CHECK (notes->notes[0].midiNote == 69);
    CHECK (notes->notes[1].midiNote == 72);
    CHECK (notes->notes[0].startSeconds <= 0.01);
    CHECK (notes->notes[0].endSeconds >= 0.49);
    CHECK (notes->lowestMidi == 69);
    CHECK (notes->highestMidi == 72);
}

void testMidiExportRoundTrip()
{
    MelodyNoteSequence melody;
    melody.durationSeconds = 1.5;
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.25, 0.75, 60, 60.0f, 0.80f, 0.95f },
        { 1.00, 1.40, 64, 64.0f, 0.70f, 0.90f },
    };

    MidiExportSettings settings;
    settings.bpm = 120.0;
    settings.channel = 3;
    settings.ticksPerQuarterNote = 960;

    const juce::File midiFile = getTestDir().getChildFile ("melody.mid");
    midiFile.deleteFile();
    juce::String error;
    CHECK (MidiExporter::writeToFile (melody, midiFile, settings, error));
    CHECK (error.isEmpty());
    CHECK (midiFile.existsAsFile());
    CHECK (midiFile.getSize() > 20);

    juce::FileInputStream input (midiFile);
    CHECK (input.openedOk());
    juce::MidiFile parsed;
    CHECK (parsed.readFrom (input));
    CHECK (parsed.getTimeFormat() == 960);

    int noteOns = 0;
    int noteOffs = 0;
    for (int trackIndex = 0; trackIndex < parsed.getNumTracks(); ++trackIndex)
    {
        const auto* track = parsed.getTrack (trackIndex);
        if (track == nullptr)
            continue;
        for (int eventIndex = 0; eventIndex < track->getNumEvents(); ++eventIndex)
        {
            const auto message = track->getEventPointer (eventIndex)->message;
            if (message.isNoteOn())
            {
                ++noteOns;
                CHECK (message.getChannel() == 3);
            }
            if (message.isNoteOff())
                ++noteOffs;
        }
    }
    CHECK (noteOns == 2);
    CHECK (noteOffs == 2);
}

void testMidiDragFilePreparation()
{
    // No melody yet: preparing a drag file is a no-op with
    // ANALYZE FIRST (same status as every export entry point).
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);
    juce::File dragFile = processor.prepareMidiDragFile();
    CHECK (dragFile == juce::File{});
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::Error);
    CHECK (processor.getStatusText() == "ANALYZE FIRST");
    processor.releaseResources();

    // The drag-file lifecycle: write a melody to the temp location
    // the panel drags from, verify it is a valid .mid, then clean
    // it up (best effort, exactly as the drag row does).
    MelodyNoteSequence melody;
    melody.durationSeconds = 1.5;
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.25, 0.75, 60, 60.0f, 0.80f, 0.95f },
        { 1.00, 1.40, 64, 64.0f, 0.70f, 0.90f },
    };
    const juce::File tempFile = juce::File::getSpecialLocation (
        juce::File::tempDirectory).getChildFile ("Melotheon Melody.mid");
    tempFile.deleteFile();
    juce::String error;
    CHECK (MidiExporter::writeToFile (melody, tempFile, MidiExportSettings{}, error));
    CHECK (error.isEmpty());
    CHECK (tempFile.existsAsFile());
    CHECK (tempFile.getSize() > 20);

    juce::FileInputStream input (tempFile);
    CHECK (input.openedOk());
    juce::MidiFile parsed;
    CHECK (parsed.readFrom (input));
    int noteOns = 0;
    for (int trackIndex = 0; trackIndex < parsed.getNumTracks(); ++trackIndex)
    {
        const auto* track = parsed.getTrack (trackIndex);
        if (track == nullptr)
            continue;
        for (int eventIndex = 0; eventIndex < track->getNumEvents(); ++eventIndex)
        {
            if (track->getEventPointer (eventIndex)->message.isNoteOn())
                ++noteOns;
        }
    }
    CHECK (noteOns == 2);

    CHECK (tempFile.deleteFile());
    CHECK (! tempFile.existsAsFile());
}

void testSampleInstrumentFollowsMelody()
{
    constexpr double sampleRate = 48000.0;
    auto oneShotBuffer = makeSine (sampleRate, 0.35, 440.0f, 0.45f);
    auto oneShot = AudioClip::fromBuffer (
        oneShotBuffer, sampleRate, "A4 one-shot", true);
    CHECK (oneShot != nullptr);
    if (oneShot == nullptr)
        return;

    MelodyNoteSequence melody;
    melody.durationSeconds = 0.85;
    melody.lowestMidi = 69;
    melody.highestMidi = 81;
    melody.notes = {
        { 0.00, 0.35, 69, 69.0f, 0.85f, 0.95f },
        { 0.42, 0.78, 81, 81.0f, 0.85f, 0.95f },
    };

    float progress = 0.0f;
    auto rendered = SampleInstrumentRenderer::render (
        *oneShot, melody, sampleRate, {},
        [&progress] (float value) { progress = value; });
    CHECK (rendered.succeeded());
    CHECK (rendered.rootMidi == 69);
    CHECK_NEAR (rendered.rootFrequencyHz, 440.0f, 4.0f);
    CHECK_NEAR (progress, 1.0f, 1e-6f);
    if (! rendered.succeeded())
        return;

    const auto& output = rendered.renderedClip->analysisMonoBuffer;
    const int windowSamples = static_cast<int> (0.20 * sampleRate);
    juce::AudioBuffer<float> first (1, windowSamples);
    juce::AudioBuffer<float> second (1, windowSamples);
    first.copyFrom (0, 0, output, 0, static_cast<int> (0.06 * sampleRate), windowSamples);
    second.copyFrom (0, 0, output, 0, static_cast<int> (0.48 * sampleRate), windowSamples);

    auto firstPitch = MonophonicPitchAnalyzer::analyze (first, sampleRate);
    auto secondPitch = MonophonicPitchAnalyzer::analyze (second, sampleRate);
    CHECK (firstPitch != nullptr && firstPitch->hasPitch());
    CHECK (secondPitch != nullptr && secondPitch->hasPitch());
    if (firstPitch != nullptr && firstPitch->hasPitch())
        CHECK_NEAR (firstPitch->medianFrequencyHz, 440.0f, 6.0f);
    if (secondPitch != nullptr && secondPitch->hasPitch())
        CHECK_NEAR (secondPitch->medianFrequencyHz, 880.0f, 15.0f);
}

// ---- Processor: pass-through, meter, record -> preview loop ----

void testProcessorPassThrough()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    juce::AudioBuffer<float> buffer (2, 512);
    juce::AudioBuffer<float> original (2, 512);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 512; ++i)
        {
            const float v = 0.3f * std::sin (static_cast<float> (i + ch * 100) * 0.05f);
            buffer.getWritePointer (ch)[i] = v;
            original.getWritePointer (ch)[i] = v;
        }

    juce::MidiBuffer midi;
    processor.processBlock (buffer, midi);

    // Preview inactive: the Mixer Track audio passes through
    // bit-identical, and the meter saw the input.
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 512; ++i)
            CHECK (buffer.getReadPointer (ch)[i] == original.getReadPointer (ch)[i]);

    CHECK (processor.getInputPeakDb() > -20.0f); // 0.3 peak ~= -10.5 dB
    CHECK (processor.getInputPeakDb() <= 0.0f);
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::Ready
        || processor.getStatus() == MelotheonAudioProcessor::Status::NoInput);
}

void testProcessorRecordAndPreview()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // Default input source is LIVE INPUT.
    CHECK (processor.getInputSource() == MelotheonAudioProcessor::InputSource::Live);
    CHECK (! processor.hasActiveClip());
    CHECK (! processor.startPreview()); // nothing to play
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::EmptySource);

    // Record a short take.
    CHECK (processor.startRecording());
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::Recording);

    juce::AudioBuffer<float> input (2, 512);
    juce::MidiBuffer midi;
    for (int b = 0; b < 4; ++b)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                input.getWritePointer (ch)[i] = 0.1f * static_cast<float> (b + 1);
        processor.processBlock (input, midi);
    }

    processor.stopRecording();
    CHECK (processor.hasActiveClip());
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::AudioReady);
    CHECK (processor.getClipVersion() == 1);
    CHECK_NEAR ((float) processor.getPreviewDurationSeconds(), 2048.0f / 44100.0f, 1e-4f);

    // Pass-through still holds while preview is inactive.
    juce::AudioBuffer<float> passthrough (2, 512);
    passthrough.clear();
    passthrough.getWritePointer (0)[0] = 0.77f;
    processor.processBlock (passthrough, midi);
    CHECK (passthrough.getReadPointer (0)[0] == 0.77f);

    // Preview plays the recorded take (same rate -> same samples).
    CHECK (processor.startPreview());
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::Playing);
    juce::AudioBuffer<float> output (2, 512);
    output.clear();
    processor.processBlock (output, midi);
    CHECK_NEAR (output.getReadPointer (0)[0], 0.1f, 1e-6f);
    CHECK_NEAR (output.getReadPointer (1)[511], 0.1f, 1e-6f);

    processor.stopPreview();
    CHECK (! processor.isPreviewPlaying());
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::Stopped);
    CHECK (processor.getPreviewPositionSeconds() == 0.0);

    processor.releaseResources();
}

void testProcessorSeek()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // No clip yet: seeking is a harmless no-op.
    processor.setPreviewPositionSeconds (0.5);
    CHECK (processor.getPreviewPositionSeconds() == 0.0);

    // Record a short take (4 blocks of 512 = 2048 samples).
    CHECK (processor.startRecording());
    juce::AudioBuffer<float> input (2, 512);
    juce::MidiBuffer midi;
    for (int b = 0; b < 4; ++b)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                input.getWritePointer (ch)[i] = 0.1f * static_cast<float> (b + 1);
        processor.processBlock (input, midi);
    }
    processor.stopRecording();
    CHECK (processor.hasActiveClip());
    const double duration = processor.getPreviewDurationSeconds();
    CHECK (duration > 0.0);

    // Seeking clamps to [0, duration].
    processor.setPreviewPositionSeconds (-1.0);
    CHECK (processor.getPreviewPositionSeconds() == 0.0);
    processor.setPreviewPositionSeconds (duration + 10.0);
    CHECK_NEAR ((float) processor.getPreviewPositionSeconds(), (float) duration, 1e-6f);

    // Seek into the middle of block 2 (sample 1280, value 0.3),
    // then Play starts from there.
    const double targetSeconds = 1280.0 / 44100.0;
    processor.setPreviewPositionSeconds (targetSeconds);
    CHECK_NEAR ((float) processor.getPreviewPositionSeconds(), (float) targetSeconds, 1e-4f);
    CHECK (processor.startPreview());
    juce::AudioBuffer<float> output (2, 512);
    output.clear();
    processor.processBlock (output, midi);
    CHECK_NEAR (output.getReadPointer (0)[0], 0.3f, 1e-4f);
    // Sample 255 is the last sample of block 2 (0.3); sample 511 has
    // crossed into block 3 (0.4), proving playback continued from the seek.
    CHECK_NEAR (output.getReadPointer (1)[255], 0.3f, 1e-4f);
    CHECK_NEAR (output.getReadPointer (1)[511], 0.4f, 1e-4f);

    processor.stopPreview();
    processor.releaseResources();
}

void testCurveEditAdjustsFramesAndResegments()
{
    // Synthetic 2 s analysis: voiced frames at MIDI 60 with a small
    // vibrato, 10 ms hop, plus an unvoiced gap at 0.80-1.00 s.
    PitchAnalysisResult analysis;
    analysis.durationSeconds = 2.0;
    analysis.hopSeconds = 0.01;
    analysis.analysisSampleRate = 24000.0;
    for (int i = 0; i < 200; ++i)
    {
        PitchFrame frame;
        frame.timeSeconds = i * 0.01;
        frame.voiced = (i < 80 || i >= 100);
        frame.midiPitch = 60.0f + 0.2f * std::sin (i * 0.3f);
        frame.frequencyHz = 261.63f;
        frame.confidence = 0.9f;
        frame.rmsDb = -20.0f;
        analysis.frames.push_back (frame);
    }
    analysis.voicedFrameCount = 180;
    analysis.minMidi = 59.8f;
    analysis.maxMidi = 60.2f;

    // Edit at 1.50 s toward MIDI 64 with a 0.15 s window.
    NoteSegmenter::applyCurveEdit (analysis, 1.50, 64.0f, 0.15);
    // The centre frame moved to the target...
    CHECK_NEAR (analysis.frames[150].midiPitch, 64.0f, 0.01f);
    // ...frames far away are untouched...
    CHECK_NEAR (analysis.frames[50].midiPitch,
                60.0f + 0.2f * std::sin (50 * 0.3f), 0.01f);
    CHECK_NEAR (analysis.frames[10].midiPitch,
                60.0f + 0.2f * std::sin (10 * 0.3f), 0.01f);
    // ...the falloff is smooth: the frame 0.10 s before the centre
    // moved only part of the way...
    const float before = analysis.frames[140].midiPitch;
    CHECK (before > 60.5f && before < 63.5f);
    // ...and the unvoiced gap stayed unvoiced and untouched.
    CHECK (! analysis.frames[90].voiced);
    CHECK_NEAR (analysis.frames[90].midiPitch,
                60.0f + 0.2f * std::sin (90 * 0.3f), 0.01f);
    // Frequency follows the new pitch, and the range metadata too.
    CHECK_NEAR (analysis.frames[150].frequencyHz, 329.63f, 1.0f);
    CHECK_NEAR (analysis.maxMidi, 64.0f, 0.01f);

    // Re-segmentation follows the corrected curve: the note covering
    // 1.50 s is now at 64.
    auto notes = NoteSegmenter::segment (analysis);
    CHECK (! notes->empty());
    bool found64 = false;
    for (const auto& note : notes->notes)
        if (note.startSeconds <= 1.50 && note.endSeconds >= 1.50)
            found64 = (note.midiNote == 64);
    CHECK (found64);
}

void testAuditionSineFallback()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // No sound loaded: a key still plays (sine fallback).
    processor.startAudition (69); // A4 = 440 Hz
    CHECK (processor.isAuditioning());
    CHECK (processor.getAuditionNote() == 69);

    juce::AudioBuffer<float> output (2, 512);
    juce::MidiBuffer midi;
    int zeroCrossings = 0;
    float previous = 0.0f;
    for (int b = 0; b < 9; ++b) // ~0.1 s at 44.1 kHz
    {
        output.clear();
        processor.processBlock (output, midi);
        const float* data = output.getReadPointer (0);
        for (int i = 0; i < 512; ++i)
        {
            if ((previous < 0.0f && data[i] >= 0.0f)
                || (previous > 0.0f && data[i] <= 0.0f))
                ++zeroCrossings;
            previous = data[i];
        }
    }
    // 440 Hz over ~0.1 s crosses zero ~88 times (the attack ramp
    // lowers the count slightly, so accept a wide band).
    CHECK (zeroCrossings > 60 && zeroCrossings < 110);

    // Note-off: the release tail dies out and the output is silent.
    processor.stopAudition();
    CHECK (! processor.isAuditioning());
    for (int b = 0; b < 10; ++b)
    {
        output.clear();
        processor.processBlock (output, midi);
    }
    CHECK_NEAR (output.getMagnitude (0, 0, 512), 0.0f, 1e-6f);

    processor.releaseResources();
}

void testAuditionSamplePlayback()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // A 440 Hz one-shot (root A4) installed as the audition source.
    auto oneShotBuffer = makeSine (44100.0, 0.5, 440.0f, 0.5f);
    auto oneShot = AudioClip::fromBuffer (oneShotBuffer, 44100.0, "A4 one-shot", true);
    CHECK (oneShot != nullptr);
    if (oneShot == nullptr)
        return;
    CHECK (SampleInstrumentRenderer::detectRootMidi (*oneShot) == 69);
    processor.installAuditionSample (oneShot, 69);

    // Audition one octave up: the one-shot plays at 2x speed (880 Hz).
    processor.startAudition (81);
    juce::AudioBuffer<float> output (2, 512);
    juce::MidiBuffer midi;
    int zeroCrossings = 0;
    float previous = 0.0f;
    for (int b = 0; b < 9; ++b) // ~0.1 s
    {
        output.clear();
        processor.processBlock (output, midi);
        const float* data = output.getReadPointer (0);
        for (int i = 0; i < 512; ++i)
        {
            if ((previous < 0.0f && data[i] >= 0.0f)
                || (previous > 0.0f && data[i] <= 0.0f))
                ++zeroCrossings;
            previous = data[i];
        }
    }
    // 880 Hz over ~0.1 s: ~184 crossings.
    CHECK (zeroCrossings > 140 && zeroCrossings < 220);

    // Installing a new source while a key is held stops the old voice
    // without crashing; the new sound (220 Hz root) then plays at the
    // requested pitch (A4 = 440 Hz from root A3).
    auto otherBuffer = makeSine (44100.0, 0.5, 220.0f, 0.5f);
    auto otherShot = AudioClip::fromBuffer (otherBuffer, 44100.0, "A3 one-shot", true);
    processor.installAuditionSample (otherShot, 57);
    processor.startAudition (69);
    zeroCrossings = 0;
    previous = 0.0f;
    for (int b = 0; b < 9; ++b)
    {
        output.clear();
        processor.processBlock (output, midi);
        const float* data = output.getReadPointer (0);
        for (int i = 0; i < 512; ++i)
        {
            if ((previous < 0.0f && data[i] >= 0.0f)
                || (previous > 0.0f && data[i] <= 0.0f))
                ++zeroCrossings;
            previous = data[i];
        }
    }
    CHECK (zeroCrossings > 60 && zeroCrossings < 110);

    processor.stopAudition();
    processor.releaseResources();
}

void testCurveEditCommitAndDragFile()
{
    // Synthetic analysis: 1.5 s of MIDI 60, 10 ms hop.
    PitchAnalysisResult analysis;
    analysis.durationSeconds = 1.5;
    analysis.hopSeconds = 0.01;
    analysis.analysisSampleRate = 24000.0;
    for (int i = 0; i < 150; ++i)
    {
        PitchFrame frame;
        frame.timeSeconds = i * 0.01;
        frame.voiced = true;
        frame.midiPitch = 60.0f;
        frame.frequencyHz = 261.63f;
        frame.confidence = 0.9f;
        frame.rmsDb = -20.0f;
        analysis.frames.push_back (frame);
    }
    analysis.voicedFrameCount = 150;
    analysis.minMidi = 60.0f;
    analysis.maxMidi = 60.0f;

    // The user drags the curve at 0.75 s up to 64.
    NoteSegmenter::applyCurveEdit (analysis, 0.75, 64.0f, 0.15);

    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);
    CHECK (processor.commitPitchCurveEdit (analysis));
    CHECK (processor.isPitchAnalysisEdited());
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::AnalysisReady);
    CHECK (processor.getStatusHint() == "CURVE EDITED - RE-ANALYZE TO RESET");
    CHECK (processor.getStatusHintVersion() > 0);
    CHECK (processor.hasMelodyNotes());
    bool found64 = false;
    for (const auto& note : processor.getMelodyNotes()->notes)
        if (note.startSeconds <= 0.75 && note.endSeconds >= 0.75)
            found64 = (note.midiNote == 64);
    CHECK (found64);

    // The committed melody is now draggable: the temp file exists for
    // the drag (valid .mid) and is deleted after it.
    const juce::File dragFile = processor.prepareMidiDragFile();
    CHECK (dragFile != juce::File{});
    CHECK (dragFile.existsAsFile());
    CHECK (dragFile.getSize() > 20);
    {
        juce::FileInputStream input (dragFile);
        CHECK (input.openedOk());
        juce::MidiFile parsed;
        CHECK (parsed.readFrom (input));
    }
    CHECK (dragFile.deleteFile());
    CHECK (! dragFile.existsAsFile());

    // A second prepare re-creates the same temp path cleanly.
    const juce::File dragFile2 = processor.prepareMidiDragFile();
    CHECK (dragFile2 == dragFile);
    CHECK (dragFile2.existsAsFile());
    dragFile2.deleteFile();

    processor.releaseResources();
}

void testProcessorEmptyTake()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    CHECK (! processor.hasActiveClip());
    CHECK (processor.getClipVersion() == 0);
    CHECK (processor.startRecording());

    // Stop before processBlock has supplied a single sample.
    processor.stopRecording();
    CHECK (! processor.isRecording());
    CHECK (! processor.hasActiveClip());
    CHECK (processor.getClipVersion() == 0);
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::NoInput);
}

void testProcessorTransportExclusion()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    juce::AudioBuffer<float> block (2, 512);
    block.clear();
    block.addSample (0, 0, 0.25f);
    block.addSample (1, 0, 0.25f);
    juce::MidiBuffer midi;

    // Create a first valid clip and begin previewing it.
    CHECK (processor.startRecording());
    processor.processBlock (block, midi);
    processor.stopRecording();
    CHECK (processor.getClipVersion() == 1);
    CHECK (processor.startPreview());
    CHECK (processor.isPreviewPlaying());

    // Record owns the transport and must stop/reset preview.
    CHECK (processor.startRecording());
    CHECK (processor.isRecording());
    CHECK (! processor.isPreviewPlaying());
    CHECK (processor.getPreviewPositionSeconds() == 0.0);

    block.clear();
    block.addSample (0, 0, 0.5f);
    block.addSample (1, 0, 0.5f);
    processor.processBlock (block, midi);

    // Play while recording finalizes the new take, then previews it.
    CHECK (processor.startPreview());
    CHECK (! processor.isRecording());
    CHECK (processor.isPreviewPlaying());
    CHECK (processor.getClipVersion() == 2);
    CHECK (processor.getStatus() == MelotheonAudioProcessor::Status::Playing);

    processor.stopPreview();
}

// ---- Note editor (MOVE / CUT / STRETCH) ----

void testMelodyEditorMoveSplitStretch()
{
    MelodyNoteSequence melody;
    melody.durationSeconds = 1.6; // 16 display beats of 0.1 s
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.0, 0.4, 60, 60.0f, 0.8f, 0.95f },
        { 0.5, 0.9, 64, 64.0f, 0.7f, 0.90f },
        { 1.0, 1.4, 62, 62.0f, 0.8f, 0.92f },
    };

    // Beat snap with a 0.1 s beat (e.g. a 600 BPM grid, or the old
    // 16-beat grid over 1.6 s): 0.24 s snaps to 0.2 s, 0.26 s to 0.3 s.
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToBeat (0.24, 0.1, 1.6), 0.2f, 1e-6f);
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToBeat (0.26, 0.1, 1.6), 0.3f, 1e-6f);
    // Snap-if-close: 0.24 s is 0.4 beats from 0.2 s -> no snap;
    // 0.21 s is 0.1 beats away -> snaps to the beat.
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToBeatIfClose (0.24, 0.1, 1.6), 0.24f, 1e-6f);
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToBeatIfClose (0.21, 0.1, 1.6), 0.2f, 1e-6f);

    // MOVE note 1 (64) up 2 semitones and 0.1 s later.
    CHECK (MelodyEditor::moveNote (melody, 1, 0.6, 66, 1.6));
    CHECK (melody.notes[1].midiNote == 66);
    CHECK_NEAR ((float) melody.notes[1].startSeconds, 0.6f, 1e-6f);
    CHECK_NEAR ((float) melody.notes[1].endSeconds, 1.0f, 1e-6f);
    CHECK (melody.highestMidi == 66);

    // MOVE note 1 into its neighbour: clamped to the gap (note 2
    // starts at 1.0 s, note 1 is 0.4 s long -> start <= 0.6 s).
    CHECK (MelodyEditor::moveNote (melody, 1, 0.95, 66, 1.6));
    CHECK_NEAR ((float) melody.notes[1].startSeconds, 0.6f, 1e-6f);

    // MOVE note 0 before the start: clamped to 0.
    CHECK (MelodyEditor::moveNote (melody, 0, -0.5, 60, 1.6));
    CHECK_NEAR ((float) melody.notes[0].startSeconds, 0.0f, 1e-6f);

    // CUT note 2 (62, 1.0-1.4 s) at 1.2 s.
    CHECK (MelodyEditor::splitNoteAt (melody, 2, 1.2));
    CHECK (melody.notes.size() == 4);
    CHECK (melody.notes[2].midiNote == 62);
    CHECK_NEAR ((float) melody.notes[2].endSeconds, 1.2f, 1e-6f);
    CHECK (melody.notes[3].midiNote == 62);
    CHECK_NEAR ((float) melody.notes[3].startSeconds, 1.2f, 1e-6f);
    CHECK_NEAR ((float) melody.notes[3].endSeconds, 1.4f, 1e-6f);

    // A split too close to an edge is refused (0.01 s piece).
    CHECK (! MelodyEditor::splitNoteAt (melody, 2, 1.21));
    CHECK (melody.notes.size() == 4);

    // STRETCH note 3 (the second half, 1.2-1.4 s): right edge to 1.5 s.
    CHECK (MelodyEditor::stretchNoteEdge (melody, 3, false, 1.5, 1.6));
    CHECK_NEAR ((float) melody.notes[3].endSeconds, 1.5f, 1e-6f);
    // Left edge to 1.35 s.
    CHECK (MelodyEditor::stretchNoteEdge (melody, 3, true, 1.35, 1.6));
    CHECK_NEAR ((float) melody.notes[3].startSeconds, 1.35f, 1e-6f);
    // Minimum duration 0.03 s: the left edge cannot pass 1.47 s.
    CHECK (MelodyEditor::stretchNoteEdge (melody, 3, true, 1.49, 1.6));
    CHECK_NEAR ((float) melody.notes[3].startSeconds, 1.47f, 1e-6f);
    // The right edge cannot pass the clip end.
    CHECK (MelodyEditor::stretchNoteEdge (melody, 3, false, 2.0, 1.6));
    CHECK_NEAR ((float) melody.notes[3].endSeconds, 1.6f, 1e-6f);

    // Bad indices are refused.
    CHECK (! MelodyEditor::moveNote (melody, 9, 0.0, 60, 1.6));
    CHECK (! MelodyEditor::splitNoteAt (melody, -1, 0.5));
    CHECK (! MelodyEditor::stretchNoteEdge (melody, 9, false, 1.0, 1.6));
}

void testMelodyEditCommitUsesEditedMelody()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // An empty edit is refused and leaves no melody.
    MelodyNoteSequence empty;
    CHECK (! processor.commitMelodyEdit (empty));
    CHECK (! processor.hasMelodyNotes());

    MelodyNoteSequence melody;
    melody.durationSeconds = 1.5;
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.25, 0.75, 60, 60.0f, 0.80f, 0.95f },
        { 1.00, 1.40, 64, 64.0f, 0.70f, 0.90f },
    };
    CHECK (processor.commitMelodyEdit (melody));
    CHECK (processor.hasMelodyNotes());
    CHECK (processor.isPitchAnalysisEdited());
    CHECK (processor.getStatusHint() == "MELODY EDITED - RE-ANALYZE TO RESET");

    // Move the second note up to 66 and export: the file must carry
    // the edited pitch, proving export reads the same melody the
    // editor commits (no second melody source).
    MelodyNoteSequence edited = melody;
    CHECK (MelodyEditor::moveNote (edited, 1, 1.00, 66, 1.5));
    CHECK (processor.commitMelodyEdit (edited));
    const juce::File midiFile = getTestDir().getChildFile ("edited-melody.mid");
    midiFile.deleteFile();
    CHECK (processor.exportMidiToFile (midiFile));
    CHECK (midiFile.existsAsFile());
    juce::FileInputStream input (midiFile);
    CHECK (input.openedOk());
    juce::MidiFile parsed;
    CHECK (parsed.readFrom (input));
    int highestNoteOn = -1;
    for (int t = 0; t < parsed.getNumTracks(); ++t)
    {
        const auto* track = parsed.getTrack (t);
        if (track == nullptr)
            continue;
        for (int e = 0; e < track->getNumEvents(); ++e)
        {
            const auto message = track->getEventPointer (e)->message;
            if (message.isNoteOn())
                highestNoteOn = std::max (highestNoteOn, message.getNoteNumber());
        }
    }
    CHECK (highestNoteOn == 66);
    midiFile.deleteFile();

    processor.releaseResources();
}

void testMelodyEditPitchShiftMovesFrames()
{
    // Synthetic analysis: 1.5 s of MIDI 60, 10 ms hop.
    PitchAnalysisResult analysis;
    analysis.durationSeconds = 1.5;
    analysis.hopSeconds = 0.01;
    analysis.analysisSampleRate = 24000.0;
    for (int i = 0; i < 150; ++i)
    {
        PitchFrame frame;
        frame.timeSeconds = i * 0.01;
        frame.voiced = true;
        frame.midiPitch = 60.0f;
        frame.frequencyHz = 261.63f;
        frame.confidence = 0.9f;
        frame.rmsDb = -20.0f;
        analysis.frames.push_back (frame);
    }
    analysis.voicedFrameCount = 150;
    analysis.minMidi = 60.0f;
    analysis.maxMidi = 60.0f;

    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);
    CHECK (processor.commitPitchCurveEdit (analysis));
    CHECK (processor.hasMelodyNotes());
    const auto original = processor.getMelodyNotes();
    CHECK (original != nullptr && ! original->empty());
    const MelodyNote firstNote = original->notes.front();
    CHECK (firstNote.midiNote == 60);

    // Pitch move (+2) with an active shift: the frames at the note
    // centre move with the note, so curve and notes stay in sync.
    MelodyNoteSequence pitched = *original;
    CHECK (MelodyEditor::moveNote (pitched, 0, firstNote.startSeconds,
                                   firstNote.midiNote + 2, original->durationSeconds));
    MelodyEditor::PitchShift shift;
    shift.active = true;
    shift.centreSeconds = 0.5 * (firstNote.startSeconds + firstNote.endSeconds);
    shift.targetMidi = static_cast<float> (firstNote.midiNote + 2);
    shift.windowSeconds = firstNote.endSeconds - firstNote.startSeconds;
    CHECK (processor.commitMelodyEdit (pitched, shift));
    CHECK (processor.getMelodyNotes()->notes.front().midiNote == firstNote.midiNote + 2);
    const auto editedAnalysis = processor.getPitchAnalysis();
    CHECK (editedAnalysis != nullptr);
    const int centreFrame = static_cast<int> (
        std::llround (shift.centreSeconds / editedAnalysis->hopSeconds));
    CHECK (centreFrame >= 0
           && centreFrame < static_cast<int> (editedAnalysis->frames.size()));
    CHECK_NEAR (editedAnalysis->frames[static_cast<size_t> (centreFrame)].midiPitch,
                static_cast<float> (firstNote.midiNote + 2), 0.5f);

    // Time-only move (inactive shift): the frames stay where they
    // were — the recorded audio stays where it was.
    MelodyNoteSequence moved = *processor.getMelodyNotes();
    const double newStart = moved.notes.front().startSeconds + 0.1;
    CHECK (MelodyEditor::moveNote (moved, 0, newStart, moved.notes.front().midiNote,
                                   moved.durationSeconds));
    CHECK (processor.commitMelodyEdit (moved));
    const auto afterMove = processor.getPitchAnalysis();
    CHECK (afterMove != nullptr);
    CHECK_NEAR (afterMove->frames[static_cast<size_t> (centreFrame)].midiPitch,
                static_cast<float> (firstNote.midiNote + 2), 0.5f);

    processor.releaseResources();
}

// ---- Live MIDI out (SEND TO CHANNEL) ----

void testLiveMidiOutScheduling()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // Record a 0.51 s take (44 blocks of 512 samples).
    CHECK (processor.startRecording());
    juce::AudioBuffer<float> input (2, 512);
    juce::MidiBuffer midi;
    for (int b = 0; b < 44; ++b)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                input.getWritePointer (ch)[i] = 0.1f;
        processor.processBlock (input, midi);
    }
    processor.stopRecording();
    CHECK (processor.hasActiveClip());
    const double duration = processor.getPreviewDurationSeconds();
    CHECK (duration > 0.5);

    // Two-note melody: C4 0.05-0.25 s, E4 0.30-0.50 s.
    MelodyNoteSequence melody;
    melody.durationSeconds = duration;
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.05, 0.25, 60, 60.0f, 0.80f, 0.95f },
        { 0.30, 0.50, 64, 64.0f, 0.70f, 0.90f },
    };

    // Run one 512-sample block, collecting (absolute sample, message)
    // pairs. The sample is the block index * 512 + the event offset.
    struct MidiEvent { int sample; juce::MidiMessage message; };
    std::vector<MidiEvent> events;
    int blockIndex = 0;
    auto runBlock = [&]()
    {
        juce::AudioBuffer<float> output (2, 512);
        output.clear();
        juce::MidiBuffer blockMidi;
        processor.processBlock (output, blockMidi);
        for (const auto metadata : blockMidi)
            events.push_back ({ blockIndex * 512 + metadata.samplePosition,
                                metadata.getMessage() });
        ++blockIndex;
    };
    auto setParam = [&] (const char* id, float value)
    {
        if (auto* param = processor.apvts.getParameter (id); param != nullptr)
            param->setValueNotifyingHost (value);
    };
    auto expectEvent = [&] (size_t index, int sample, bool isOn, int note, int channel)
    {
        CHECK (index < events.size());
        if (index >= events.size())
            return;
        // One sample of slack: the event time is computed from the
        // preview position in seconds, so float rounding can shift
        // an offset by a single sample.
        CHECK (std::abs (events[index].sample - sample) <= 1);
        CHECK (events[index].message.isNoteOn() == isOn);
        CHECK (events[index].message.isNoteOff() == ! isOn);
        CHECK (events[index].message.getNoteNumber() == note);
        CHECK (events[index].message.getChannel() == channel);
    };

    // Both switches off: preview plays, no MIDI leaves.
    CHECK (processor.startPreview());
    runBlock();
    CHECK (events.empty());

    // MIDI OUT on but SEND TO CHANNEL off: still nothing.
    setParam ("midiOut", 1.0f);
    runBlock();
    CHECK (events.empty());

    // Both on, channel 3 (choice index 2 of 16): the melody streams
    // in time with the playhead. No melody is committed yet, so
    // still nothing.
    setParam ("sendToChannel", 1.0f);
    setParam ("midiChannel", 2.0f / 15.0f);
    CHECK (processor.getMidiChannel() == 3);
    runBlock();
    CHECK (events.empty());

    // Commit the melody mid-playback: the stream picks it up from
    // the current position (resync), then follows it to the end.
    CHECK (processor.commitMelodyEdit (melody));
    while (blockIndex < 44)
        runBlock();
    CHECK (events.size() == 4);
    expectEvent (0, 2205, true, 60, 3);   // 0.05 s
    expectEvent (1, 11025, false, 60, 3); // 0.25 s
    expectEvent (2, 13230, true, 64, 3);  // 0.30 s
    expectEvent (3, 22050, false, 64, 3); // 0.50 s
    processor.stopPreview();

    // Note-off hygiene on stop: restart, stop mid-note (note 60
    // sounding), and the next block must carry its note-off.
    events.clear();
    blockIndex = 0;
    CHECK (processor.startPreview());
    while (blockIndex < 10) // 10 blocks = 0.116 s, note 60 sounding
        runBlock();
    CHECK (events.size() == 1);
    expectEvent (0, 2205, true, 60, 3);
    processor.stopPreview();
    events.clear();
    runBlock();
    CHECK (events.size() == 1);
    CHECK (events[0].message.isNoteOff());
    CHECK (events[0].message.getNoteNumber() == 60);
    CHECK (events[0].message.getChannel() == 3);

    // Channel change mid-send: the sounding note is stopped on its
    // old channel before the new channel takes over.
    events.clear();
    blockIndex = 0;
    CHECK (processor.startPreview());
    while (blockIndex < 10)
        runBlock();
    CHECK (events.size() == 1);
    setParam ("midiChannel", 4.0f / 15.0f); // channel 5
    CHECK (processor.getMidiChannel() == 5);
    events.clear();
    runBlock(); // block 10: note-off ch 3 + note-on ch 5, same offset
    CHECK (events.size() == 2);
    bool sawOffOld = false, sawOnNew = false;
    for (const auto& event : events)
    {
        if (event.message.isNoteOff() && event.message.getNoteNumber() == 60
            && event.message.getChannel() == 3)
            sawOffOld = true;
        if (event.message.isNoteOn() && event.message.getNoteNumber() == 60
            && event.message.getChannel() == 5)
            sawOnNew = true;
    }
    CHECK (sawOffOld);
    CHECK (sawOnNew);
    processor.stopPreview();

    processor.releaseResources();
}

void testLiveMidiOutResyncProcessesWholeBlock()
{
    constexpr double sampleRate = 44100.0;
    constexpr int blockSize = 512;

    MelotheonAudioProcessor processor;
    processor.prepareToPlay (sampleRate, blockSize);

    // Build a clip long enough for the initial block and a replacement
    // melody in the following block.
    CHECK (processor.startRecording());
    juce::AudioBuffer<float> input (2, blockSize);
    juce::MidiBuffer ignoredMidi;
    input.clear();
    for (int block = 0; block < 12; ++block)
        processor.processBlock (input, ignoredMidi);
    processor.stopRecording();
    CHECK (processor.getPreviewDurationSeconds() > 0.1);

    auto setParam = [&] (const char* id, float value)
    {
        if (auto* param = processor.apvts.getParameter (id); param != nullptr)
            param->setValueNotifyingHost (value);
    };
    setParam ("midiOut", 1.0f);
    setParam ("sendToChannel", 1.0f);

    // Both notes live entirely inside the first 11.61 ms block. The first
    // is already active at block start; the second starts later in the same
    // block. A resync must emit all four events, not return after note 60 on.
    MelodyNoteSequence initial;
    initial.durationSeconds = processor.getPreviewDurationSeconds();
    initial.lowestMidi = 60;
    initial.highestMidi = 62;
    initial.notes = {
        { 0.000, 0.004, 60, 60.0f, 0.80f, 0.95f },
        { 0.006, 0.009, 62, 62.0f, 0.75f, 0.94f },
    };
    CHECK (processor.commitMelodyEdit (initial));
    CHECK (processor.startPreview());

    juce::AudioBuffer<float> output (2, blockSize);
    juce::MidiBuffer midi;
    output.clear();
    processor.processBlock (output, midi);

    struct Event { int sample; juce::MidiMessage message; };
    std::vector<Event> events;
    for (const auto metadata : midi)
        events.push_back ({ metadata.samplePosition, metadata.getMessage() });

    CHECK (events.size() == 4);
    if (events.size() == 4)
    {
        CHECK (events[0].message.isNoteOn());
        CHECK (events[0].message.getNoteNumber() == 60);
        CHECK (events[0].sample == 0);
        CHECK (events[1].message.isNoteOff());
        CHECK (events[1].message.getNoteNumber() == 60);
        CHECK (std::abs (events[1].sample - 176) <= 1); // 0.004 s
        CHECK (events[2].message.isNoteOn());
        CHECK (events[2].message.getNoteNumber() == 62);
        CHECK (std::abs (events[2].sample - 264) <= 1); // 0.006 s
        CHECK (events[3].message.isNoteOff());
        CHECK (events[3].message.getNoteNumber() == 62);
        CHECK (std::abs (events[3].sample - 396) <= 1); // 0.009 s
    }

    // Replace the melody while playback is running. The new short note
    // starts and ends inside the next block. A generation resync must emit
    // both events in that block.
    MelodyNoteSequence replacement;
    replacement.durationSeconds = initial.durationSeconds;
    replacement.lowestMidi = 67;
    replacement.highestMidi = 67;
    replacement.notes = {
        { 0.015, 0.020, 67, 67.0f, 0.85f, 0.96f },
    };
    CHECK (processor.commitMelodyEdit (replacement));
    midi.clear();
    output.clear();
    processor.processBlock (output, midi);

    events.clear();
    for (const auto metadata : midi)
        events.push_back ({ metadata.samplePosition, metadata.getMessage() });
    CHECK (events.size() == 2);
    if (events.size() == 2)
    {
        CHECK (events[0].message.isNoteOn());
        CHECK (events[0].message.getNoteNumber() == 67);
        CHECK (std::abs (events[0].sample - 149) <= 1);
        CHECK (events[1].message.isNoteOff());
        CHECK (events[1].message.getNoteNumber() == 67);
        CHECK (std::abs (events[1].sample - 370) <= 1);
    }

    processor.stopPreview();
    processor.releaseResources();
}

void testLiveMidiOutSeekResyncClearsOldNote()
{
    constexpr double sampleRate = 44100.0;
    constexpr int blockSize = 512;

    MelotheonAudioProcessor processor;
    processor.prepareToPlay (sampleRate, blockSize);

    CHECK (processor.startRecording());
    juce::AudioBuffer<float> input (2, blockSize);
    juce::MidiBuffer midi;
    input.clear();
    for (int block = 0; block < 12; ++block)
        processor.processBlock (input, midi);
    processor.stopRecording();

    MelodyNoteSequence melody;
    melody.durationSeconds = processor.getPreviewDurationSeconds();
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.000, 0.040, 60, 60.0f, 0.80f, 0.95f },
        { 0.050, 0.060, 64, 64.0f, 0.80f, 0.95f },
    };
    CHECK (processor.commitMelodyEdit (melody));
    if (auto* param = processor.apvts.getParameter ("midiOut"); param != nullptr)
        param->setValueNotifyingHost (1.0f);
    if (auto* param = processor.apvts.getParameter ("sendToChannel"); param != nullptr)
        param->setValueNotifyingHost (1.0f);

    CHECK (processor.startPreview());
    juce::AudioBuffer<float> output (2, blockSize);
    output.clear();
    midi.clear();
    processor.processBlock (output, midi);

    // Note 60 is now sounding. Seek into note 64, close enough to its end
    // that the end also falls in the destination block.
    processor.setPreviewPositionSeconds (0.052);
    output.clear();
    midi.clear();
    processor.processBlock (output, midi);

    // Exactly three events, in order: the old note off at offset 0,
    // the destination note on at offset 0, its off near sample 352.
    // A bare "saw" flag per event would let extra or spurious events
    // pass silently, so the count and the order are checked too.
    std::vector<std::pair<int, juce::MidiMessage>> events;
    for (const auto metadata : midi)
        events.push_back ({ metadata.samplePosition, metadata.getMessage() });
    CHECK (events.size() == 3);
    if (events.size() == 3)
    {
        CHECK (events[0].second.isNoteOff());
        CHECK (events[0].second.getNoteNumber() == 60);
        CHECK (events[0].first == 0);
        CHECK (events[1].second.isNoteOn());
        CHECK (events[1].second.getNoteNumber() == 64);
        CHECK (events[1].first == 0);
        CHECK (events[2].second.isNoteOff());
        CHECK (events[2].second.getNoteNumber() == 64);
        CHECK (std::abs (events[2].first - 352) <= 1);
    }

    processor.stopPreview();
    processor.releaseResources();
}

void testLiveMidiOutSnapshotPublicationStress()
{
    constexpr double sampleRate = 44100.0;
    constexpr int blockSize = 64;

    MelotheonAudioProcessor processor;
    processor.prepareToPlay (sampleRate, blockSize);

    // About six seconds gives both threads enough overlap without making
    // the test depend on real-time sleeping.
    CHECK (processor.startRecording());
    juce::AudioBuffer<float> input (2, blockSize);
    juce::MidiBuffer midi;
    input.clear();
    for (int block = 0; block < 4096; ++block)
        processor.processBlock (input, midi);
    processor.stopRecording();

    MelodyNoteSequence melody;
    melody.durationSeconds = processor.getPreviewDurationSeconds();
    melody.lowestMidi = 60;
    melody.highestMidi = 60;
    melody.notes = {
        { 0.0, melody.durationSeconds - 0.01, 60, 60.0f, 0.8f, 0.95f },
    };
    CHECK (processor.commitMelodyEdit (melody));
    if (auto* param = processor.apvts.getParameter ("midiOut"); param != nullptr)
        param->setValueNotifyingHost (1.0f);
    if (auto* param = processor.apvts.getParameter ("sendToChannel"); param != nullptr)
        param->setValueNotifyingHost (1.0f);
    CHECK (processor.startPreview());

    std::atomic<bool> begin { false };
    std::atomic<bool> invalidMidi { false };
    std::thread audioThread ([&]
    {
        while (! begin.load (std::memory_order_acquire))
            std::this_thread::yield();

        for (int block = 0; block < 4096 && processor.isPreviewPlaying(); ++block)
        {
            juce::AudioBuffer<float> output (2, blockSize);
            juce::MidiBuffer blockMidi;
            output.clear();
            processor.processBlock (output, blockMidi);
            for (const auto metadata : blockMidi)
            {
                const int note = metadata.getMessage().getNoteNumber();
                if (note < 60 || note > 71)
                    invalidMidi.store (true, std::memory_order_release);
            }
            std::this_thread::yield();
        }
    });

    begin.store (true, std::memory_order_release);
    for (int edit = 0; edit < 1000; ++edit)
    {
        MelodyNoteSequence edited = melody;
        const int note = 60 + (edit % 12);
        edited.notes.front().midiNote = note;
        edited.notes.front().sourceMidiPitch = static_cast<float> (note);
        edited.lowestMidi = note;
        edited.highestMidi = note;
        CHECK (processor.commitMelodyEdit (edited));
        std::this_thread::yield();
    }

    audioThread.join();
    CHECK (! invalidMidi.load (std::memory_order_acquire));
    processor.stopPreview();
    processor.releaseResources();
}

// ---- Live MIDI out: P0-4 natural end + rate/block matrix + hygiene ----

struct MidiOutHarness
{
    MelotheonAudioProcessor processor;
    double sampleRate = 44100.0;
    int blockSize = 512;
    int lastBlockStartSample = 0;
    int runningSample = 0; // monotonic block counter (see runBlock)

    struct MidiEvent { int sample; juce::MidiMessage message; };
    std::vector<MidiEvent> events;

    explicit MidiOutHarness (double rate, int block)
        : sampleRate (rate), blockSize (block)
    {
        processor.prepareToPlay (rate, block);
    }

    // Record a take of at least durationSeconds by feeding constant
    // input blocks. The recording block size can differ from the
    // playback block size (the P0-4 test needs that: a clip whose
    // end falls mid-block on playback).
    void recordTake (double durationSeconds, int recordBlockSize = 0)
    {
        const int recordBlock = recordBlockSize > 0 ? recordBlockSize : blockSize;
        const int recordBlocks = static_cast<int> (
            std::ceil (durationSeconds * sampleRate / recordBlock));
        CHECK (processor.startRecording());
        juce::AudioBuffer<float> input (2, recordBlock);
        juce::MidiBuffer midi;
        for (int b = 0; b < recordBlocks; ++b)
        {
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < recordBlock; ++i)
                    input.getWritePointer (ch)[i] = 0.1f;
            processor.processBlock (input, midi);
        }
        processor.stopRecording();
        CHECK (processor.hasActiveClip());
        CHECK (processor.getPreviewDurationSeconds() >= durationSeconds);
    }

    void setParam (const char* id, float value)
    {
        if (auto* param = processor.apvts.getParameter (id); param != nullptr)
            param->setValueNotifyingHost (value);
    }

    void enableMidiOut (int channel) // 1-based
    {
        setParam ("midiOut", 1.0f);
        setParam ("sendToChannel", 1.0f);
        setParam ("midiChannel", static_cast<float> (channel - 1) / 15.0f);
        CHECK (processor.getMidiChannel() == channel);
    }

    void runBlock()
    {
        // The absolute sample comes from the preview position, not
        // the block index, so it stays correct across seeks. After
        // the preview stops the position resets to 0, so a stopped
        // block (e.g. a drain block turning off a sounding note)
        // keeps advancing from the last block instead of jumping
        // back to 0.
        if (processor.isPreviewPlaying() || processor.isInstrumentPreviewPlaying())
            lastBlockStartSample = static_cast<int> (std::lround (
                processor.getDisplayPreviewPositionSeconds() * sampleRate));
        else
            lastBlockStartSample = runningSample;
        juce::AudioBuffer<float> output (2, blockSize);
        output.clear();
        juce::MidiBuffer blockMidi;
        processor.processBlock (output, blockMidi);
        for (const auto metadata : blockMidi)
            events.push_back ({ lastBlockStartSample + metadata.samplePosition,
                                metadata.getMessage() });
        runningSample = lastBlockStartSample + blockSize;
    }

    void runBlocks (int count) { for (int i = 0; i < count; ++i) runBlock(); }

    // Absolute sample for a time in seconds at this sample rate.
    int sampleAt (double seconds) const
    {
        return static_cast<int> (std::lround (seconds * sampleRate));
    }

    void expectEvent (size_t index, int sample, bool isOn, int note, int channel, int slack = 1)
    {
        CHECK (index < events.size());
        if (index >= events.size())
            return;
        // One or two samples of slack: event times are computed from
        // the preview position in seconds, so float rounding can
        // shift an offset by a sample.
        CHECK (std::abs (events[index].sample - sample) <= slack);
        CHECK (events[index].message.isNoteOn() == isOn);
        CHECK (events[index].message.isNoteOff() == ! isOn);
        CHECK (events[index].message.getNoteNumber() == note);
        CHECK (events[index].message.getChannel() == channel);
    }

    // Every note-on must have a matching note-off and vice versa.
    // Call after a drain (preview stopped + a few extra blocks).
    bool eventsAreBalanced() const
    {
        // Tracked per (note, channel): a note-on on channel 3 and
        // a note-off on channel 5 for the same pitch must not
        // balance — that pair is a stuck note on 3 and a stray off
        // on 5.
        int sounding[128][16] = {};
        for (const auto& event : events)
        {
            const int note = event.message.getNoteNumber();
            const int channel = event.message.getChannel() - 1;
            if (note < 0 || note > 127 || channel < 0 || channel > 15)
                return false;
            if (event.message.isNoteOn())
                ++sounding[note][channel];
            else if (event.message.isNoteOff())
            {
                if (sounding[note][channel] <= 0)
                    return false; // note-off without a note-on
                --sounding[note][channel];
            }
        }
        for (int i = 0; i < 128; ++i)
            for (int ch = 0; ch < 16; ++ch)
                if (sounding[i][ch] != 0)
                    return false; // note-on without a note-off
        return true;
    }
};

void testLiveMidiOutPlaybackEndsMidBlock()
{
    // P0-4: clip 44 blocks x 512 = 22528 samples (0.5109 s at
    // 44.1 kHz). Played back in 1000-sample blocks, the clip ends
    // at sample 22528, inside block 22 (22000-22999) — a mid-block
    // end. The last note ends at 0.505 s (sample 22270), inside the
    // same block. The old code cut it at the block start (22000);
    // the fix emits the note-off at its true end sample.
    MelodyNoteSequence melody;
    melody.durationSeconds = 0.5109;
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.05, 0.25, 60, 60.0f, 0.80f, 0.95f },
        { 0.30, 0.505, 64, 64.0f, 0.70f, 0.90f },
    };

    MidiOutHarness harness (44100.0, 1000);
    harness.recordTake (0.51, 512);
    harness.enableMidiOut (3);
    CHECK (harness.processor.commitMelodyEdit (melody));
    CHECK (harness.processor.startPreview());
    harness.runBlocks (26); // past the clip end (block 22)
    CHECK (harness.events.size() == 4);
    harness.expectEvent (0, harness.sampleAt (0.05), true, 60, 3);
    harness.expectEvent (1, harness.sampleAt (0.25), false, 60, 3);
    harness.expectEvent (2, harness.sampleAt (0.30), true, 64, 3);
    // The note-off lands at its true end sample: 0.505 s is sample
    // 22270 (floor of 22270.5), not 22271 and not the block start
    // (22000). Exact (slack 0): the floor computation is
    // deterministic IEEE754 double arithmetic.
    harness.expectEvent (3, 22270, false, 64, 3, 0);
    // No event may land at the block start where the old code cut
    // the last note.
    for (const auto& event : harness.events)
        CHECK (event.sample != 22000);
    CHECK (harness.eventsAreBalanced());
    harness.processor.stopPreview();
    harness.processor.releaseResources();
}

void testLiveMidiOutSchedulingAcrossRatesAndBlocks()
{
    // P3-15: the same two-note melody must schedule exactly at
    // 44.1 / 48 / 96 kHz and at every block size 64..1024.
    const double rates[] = { 44100.0, 48000.0, 96000.0 };
    const int blocks[] = { 64, 128, 256, 512, 1024 };
    for (double rate : rates)
    {
        for (int blockSize : blocks)
        {
            MelodyNoteSequence melody;
            melody.durationSeconds = 0.51;
            melody.lowestMidi = 60;
            melody.highestMidi = 64;
            melody.notes = {
                { 0.05, 0.25, 60, 60.0f, 0.80f, 0.95f },
                { 0.30, 0.50, 64, 64.0f, 0.70f, 0.90f },
            };
            MidiOutHarness harness (rate, blockSize);
            harness.recordTake (0.51, blockSize);
            harness.enableMidiOut (3);
            CHECK (harness.processor.commitMelodyEdit (melody));
            CHECK (harness.processor.startPreview());
            const int totalBlocks = static_cast<int> (
                std::ceil (0.55 * rate / blockSize)) + 2;
            harness.runBlocks (totalBlocks);
            CHECK (harness.events.size() == 4);
            harness.expectEvent (0, harness.sampleAt (0.05), true, 60, 3, 2);
            harness.expectEvent (1, harness.sampleAt (0.25), false, 60, 3, 2);
            harness.expectEvent (2, harness.sampleAt (0.30), true, 64, 3, 2);
            harness.expectEvent (3, harness.sampleAt (0.50), false, 64, 3, 2);
            CHECK (harness.eventsAreBalanced());
            harness.processor.stopPreview();
            harness.processor.releaseResources();
        }
    }
}

void testLiveMidiOutMelodyChangeAndToggleHygiene()
{
    // Two-note melody: C4 0.05-0.25 s, E4 0.30-0.50 s.
    MelodyNoteSequence melody;
    melody.durationSeconds = 0.51;
    melody.lowestMidi = 60;
    melody.highestMidi = 64;
    melody.notes = {
        { 0.05, 0.25, 60, 60.0f, 0.80f, 0.95f },
        { 0.30, 0.50, 64, 64.0f, 0.70f, 0.90f },
    };

    // (i) First note starts mid-resync-block via a mid-playback
    // commit: commit when the playhead is at 0.046 s, so the resync
    // block (0.046-0.058 s) contains the 0.05 s note-on.
    {
        MidiOutHarness harness (44100.0, 512);
        harness.recordTake (0.51);
        harness.enableMidiOut (3);
        CHECK (harness.processor.startPreview());
        harness.runBlocks (4); // playhead at 0.046 s
        harness.events.clear();
        CHECK (harness.processor.commitMelodyEdit (melody));
        harness.runBlocks (40); // to the end of the clip
        CHECK (harness.events.size() == 4);
        harness.expectEvent (0, harness.sampleAt (0.05), true, 60, 3);
        harness.expectEvent (1, harness.sampleAt (0.25), false, 60, 3);
        harness.expectEvent (2, harness.sampleAt (0.30), true, 64, 3);
        harness.expectEvent (3, harness.sampleAt (0.50), false, 64, 3);
        harness.processor.stopPreview();
        harness.processor.releaseResources();
    }

    // (ii) Melody change while MIDI is playing: the old note stops
    // on its original channel and the new melody resyncs from the
    // current position, with no stuck note at the end.
    {
        MelodyNoteSequence melodyB;
        melodyB.durationSeconds = 0.51;
        melodyB.lowestMidi = 64;
        melodyB.highestMidi = 64;
        melodyB.notes = { { 0.05, 0.45, 64, 64.0f, 0.70f, 0.90f } };

        MidiOutHarness harness (44100.0, 512);
        harness.recordTake (0.51);
        harness.enableMidiOut (3);
        CHECK (harness.processor.commitMelodyEdit (melody));
        CHECK (harness.processor.startPreview());
        harness.runBlocks (9); // playhead ~0.104 s, note 60 sounding
        CHECK (harness.events.size() == 1);
        harness.expectEvent (0, harness.sampleAt (0.05), true, 60, 3);
        // No clear here: the balance check below needs the whole
        // stream (note 60 was already turned on before the change).
        CHECK (harness.processor.commitMelodyEdit (melodyB));
        harness.runBlock(); // resync block: off 60 + on 64, both at offset 0
        CHECK (harness.events.size() == 3);
        const int resyncStart = harness.lastBlockStartSample;
        harness.expectEvent (1, resyncStart, false, 60, 3);
        harness.expectEvent (2, resyncStart, true, 64, 3);
        harness.runBlocks (30); // past the 0.45 s note end
        CHECK (harness.events.size() == 4);
        harness.expectEvent (3, harness.sampleAt (0.45), false, 64, 3);
        CHECK (harness.eventsAreBalanced());
        harness.processor.stopPreview();
        harness.processor.releaseResources();
    }

    // (iii) MIDI toggle change while a note is active: turning SEND
    // TO CHANNEL off stops the sounding note immediately, with no
    // stuck note.
    {
        MidiOutHarness harness (44100.0, 512);
        harness.recordTake (0.51);
        harness.enableMidiOut (3);
        CHECK (harness.processor.commitMelodyEdit (melody));
        CHECK (harness.processor.startPreview());
        harness.runBlocks (9); // note 60 sounding
        CHECK (harness.events.size() == 1);
        harness.setParam ("sendToChannel", 0.0f); // toggle off mid-note
        // No clear here: the balance check below needs the whole
        // stream (note 60 was already turned on).
        harness.runBlock();
        CHECK (harness.events.size() == 2);
        harness.expectEvent (1, harness.lastBlockStartSample, false, 60, 3);
        CHECK (harness.eventsAreBalanced());
        harness.processor.stopPreview();
        harness.processor.releaseResources();
    }

    // (iv) The MIDI OUT master off mid-note: same hygiene as SEND
    // off — the sounding note stops immediately, no stuck note.
    {
        MidiOutHarness harness (44100.0, 512);
        harness.recordTake (0.51);
        harness.enableMidiOut (3);
        CHECK (harness.processor.commitMelodyEdit (melody));
        CHECK (harness.processor.startPreview());
        harness.runBlocks (9); // note 60 sounding
        CHECK (harness.events.size() == 1);
        harness.setParam ("midiOut", 0.0f); // master off mid-note
        harness.runBlock();
        CHECK (harness.events.size() == 2);
        harness.expectEvent (1, harness.lastBlockStartSample, false, 60, 3);
        CHECK (harness.eventsAreBalanced());
        harness.processor.stopPreview();
        harness.processor.releaseResources();
    }

    // (v) SEND TO CHANNEL on mid-note: the note active at the
    // current position starts sounding at the next block (resync),
    // as if sending had been on from the start.
    {
        MidiOutHarness harness (44100.0, 512);
        harness.recordTake (0.51);
        harness.setParam ("midiOut", 1.0f); // master on, SEND off
        harness.setParam ("midiChannel", 2.0f / 15.0f); // channel 3
        CHECK (harness.processor.commitMelodyEdit (melody));
        CHECK (harness.processor.startPreview());
        harness.runBlocks (9); // playhead ~0.104 s, note 60 active
        CHECK (harness.events.empty()); // SEND off: nothing streamed
        harness.setParam ("sendToChannel", 1.0f); // on mid-note
        harness.runBlock(); // resync block: on 60 at offset 0
        CHECK (harness.events.size() == 1);
        harness.expectEvent (0, harness.lastBlockStartSample, true, 60, 3);
        // 12 blocks past the resync (playhead 0.255 s): past the
        // 0.25 s note end, before the 0.30 s next note.
        harness.runBlocks (12);
        CHECK (harness.events.size() == 2);
        harness.expectEvent (1, harness.sampleAt (0.25), false, 60, 3);
        CHECK (harness.eventsAreBalanced());
        harness.processor.stopPreview();
        harness.processor.releaseResources();
    }
}

void testLiveMidiOutEventBalanceUnderConcurrentEdits()
{
    // The branch's publication stress test checks for invalid notes;
    // this one checks the pairing: under concurrent publishes every
    // note-on must have its note-off (no stuck notes) and no
    // note-off may arrive without a note-on.
    MelodyNoteSequence melodyA;
    melodyA.durationSeconds = 0.51;
    melodyA.lowestMidi = 60;
    melodyA.highestMidi = 64;
    melodyA.notes = {
        { 0.05, 0.25, 60, 60.0f, 0.80f, 0.95f },
        { 0.30, 0.50, 64, 64.0f, 0.70f, 0.90f },
    };
    MelodyNoteSequence melodyB;
    melodyB.durationSeconds = 0.51;
    melodyB.lowestMidi = 65;
    melodyB.highestMidi = 69;
    melodyB.notes = {
        { 0.05, 0.25, 65, 65.0f, 0.80f, 0.95f },
        { 0.30, 0.50, 69, 69.0f, 0.70f, 0.90f },
    };

    MidiOutHarness harness (44100.0, 512);
    harness.recordTake (0.51);
    harness.enableMidiOut (3);
    CHECK (harness.processor.commitMelodyEdit (melodyA));
    CHECK (harness.processor.startPreview());

    std::atomic<bool> stopPublishing { false };
    std::thread publisher ([&]()
    {
        for (int i = 0; i < 200 && ! stopPublishing.load (std::memory_order_acquire); ++i)
        {
            harness.processor.commitMelodyEdit (i % 2 == 0 ? melodyB : melodyA);
            // Pace the publishes so they interleave with the audio
            // blocks instead of all landing before the first one.
            std::this_thread::sleep_for (std::chrono::microseconds (100));
        }
    });

    harness.runBlocks (400);
    stopPublishing.store (true, std::memory_order_release);
    publisher.join();

    harness.processor.stopPreview();
    harness.runBlocks (4); // drain: any sounding note gets its note-off
    // A balance check on zero events passes vacuously, so the event
    // count is asserted first: both melodies have two notes, so at
    // least two on/off pairs must have streamed.
    CHECK (harness.events.size() >= 4);
    CHECK (harness.eventsAreBalanced());

    // Phase 2: publish while stopped, then resume — the stream must
    // stay balanced.
    for (int i = 0; i < 20; ++i)
        harness.processor.commitMelodyEdit (i % 2 == 0 ? melodyA : melodyB);
    harness.events.clear();
    CHECK (harness.processor.startPreview());
    harness.runBlocks (44);
    harness.processor.stopPreview();
    harness.runBlocks (4);
    CHECK (harness.events.size() >= 4);
    CHECK (harness.eventsAreBalanced());
    harness.processor.releaseResources();
}

// ---- MIDI buffer pre-allocation (P3-13) + non-blocking snapshot handoff (P3-14) ----

void testMidiOutCapacityBound()
{
    // P3-13: the bound is derived, not magic — 3 events per note
    // (note-on + note-off + one defensive re-trigger note-off) plus
    // one all-notes-off pass (128), at 9 bytes per event.
    CHECK (MelotheonAudioProcessor::midiOutCapacityBytesForSequence (0) == 128 * 9);
    CHECK (MelotheonAudioProcessor::midiOutCapacityBytesForSequence (1) == 131 * 9);
    CHECK (MelotheonAudioProcessor::midiOutCapacityBytesForSequence (2) == 134 * 9);
    CHECK (MelotheonAudioProcessor::midiOutCapacityBytesForSequence (17) == 179 * 9);
    CHECK (MelotheonAudioProcessor::midiOutCapacityBytesForSequence (2000) == 6128 * 9);
    // The practical ceiling for a 60 s clip (shortest note 0.03 s)
    // is ~2000 notes, so one block's pre-allocation stays near 55 KB.
    CHECK (MelotheonAudioProcessor::midiOutCapacityBytesForSequence (2000) < 64 * 1024);
}

void testFindFreeMidiOutSlot()
{
    // P3-14, deterministic: with three slots there is always a free
    // one, even with the published slot AND a second slot held. With
    // two slots the third case below would have no free slot and the
    // old code would spin forever — that is exactly the bug.
    const int allFree[3] = { 0, 0, 0 };
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (0, allFree) == 1);
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (1, allFree) == 0);
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (2, allFree) == 0);

    const int publishedHeld[3] = { 1, 0, 0 };
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (0, publishedHeld) == 1);

    // The key case: published slot 0 held AND slot 1 held — the
    // publish must still return promptly, using slot 2.
    const int twoHeld[3] = { 1, 1, 0 };
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (0, twoHeld) == 2);
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (1, twoHeld) == 2);
    const int twoHeldC[3] = { 0, 1, 1 };
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (2, twoHeldC) == 0);

    // All three held: -1 (unreachable in practice — the audio
    // thread holds at most one slot — and publish drops the update
    // rather than spinning).
    const int allHeld[3] = { 1, 1, 1 };
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (0, allHeld) == -1);
    CHECK (MelotheonAudioProcessor::findFreeMidiOutSlot (2, allHeld) == -1);
}

void testLiveMidiOutDenseBlockNoEventLost()
{
    // P3-13: worst-case event density — 60 one-millisecond notes in
    // a single 4096-sample block (120 events). Without the
    // pre-allocation the host buffer would grow repeatedly while
    // events are written; with it, every event lands at its exact
    // sample offset and none is lost.
    MelodyNoteSequence melody;
    melody.durationSeconds = 0.10;
    melody.lowestMidi = 60;
    melody.highestMidi = 71;
    for (int i = 0; i < 60; ++i)
    {
        const double start = i * 0.001;
        const int pitch = 60 + (i % 12);
        melody.notes.push_back ({ start, start + 0.001, pitch,
                                  static_cast<float> (pitch), 0.80f, 0.95f });
    }

    // The bound must cover this block: 120 events x 9 bytes.
    const auto capacity = MelotheonAudioProcessor::midiOutCapacityBytesForSequence (
        melody.notes.size());
    CHECK (capacity == (3 * 60 + 128) * 9);
    CHECK (static_cast<size_t> (120 * 9) <= capacity);

    MidiOutHarness harness (44100.0, 4096);
    harness.recordTake (0.10);
    harness.enableMidiOut (3);
    CHECK (harness.processor.commitMelodyEdit (melody));
    CHECK (harness.processor.startPreview());
    harness.runBlocks (3); // the whole 0.06 s melody is inside block 0
    CHECK (harness.events.size() == 120);
    if (harness.events.size() == 120)
    {
        for (int i = 0; i < 60; ++i)
        {
            const size_t onIndex = static_cast<size_t> (2 * i);
            const size_t offIndex = static_cast<size_t> (2 * i + 1);
            CHECK (std::abs (harness.events[onIndex].sample - harness.sampleAt (i * 0.001)) <= 1);
            CHECK (harness.events[onIndex].message.isNoteOn());
            CHECK (harness.events[onIndex].message.getNoteNumber() == 60 + (i % 12));
            CHECK (std::abs (harness.events[offIndex].sample - harness.sampleAt ((i + 1) * 0.001)) <= 1);
            CHECK (harness.events[offIndex].message.isNoteOff());
            CHECK (harness.events[offIndex].message.getNoteNumber() == 60 + (i % 12));
        }
    }
    harness.processor.stopPreview();
    harness.runBlocks (2); // drain: no stuck note may remain
    CHECK (harness.eventsAreBalanced());
    harness.processor.releaseResources();
}

void testLiveMidiOutPublishNeverBlocksUnderLoad()
{
    // P3-14 (behavioral): publishes from a second thread while
    // blocks render must always complete — the old two-slot design
    // spun on the message thread whenever the inactive slot was
    // held. The deterministic proof is testFindFreeMidiOutSlot;
    // this test proves the real publish path stays live under load
    // (tight loop, no pacing: maximum interleaving with the audio
    // blocks) and that the stream stays balanced afterwards.
    MelodyNoteSequence melodyA;
    melodyA.durationSeconds = 0.51;
    melodyA.lowestMidi = 60;
    melodyA.highestMidi = 64;
    melodyA.notes = {
        { 0.05, 0.25, 60, 60.0f, 0.80f, 0.95f },
        { 0.30, 0.50, 64, 64.0f, 0.70f, 0.90f },
    };
    MelodyNoteSequence melodyB;
    melodyB.durationSeconds = 0.51;
    melodyB.lowestMidi = 65;
    melodyB.highestMidi = 69;
    melodyB.notes = {
        { 0.05, 0.25, 65, 65.0f, 0.80f, 0.95f },
        { 0.30, 0.50, 69, 69.0f, 0.70f, 0.90f },
    };

    MidiOutHarness harness (44100.0, 512);
    harness.recordTake (0.51);
    harness.enableMidiOut (3);
    CHECK (harness.processor.commitMelodyEdit (melodyA));
    CHECK (harness.processor.startPreview());

    const int versionBefore = harness.processor.getMelodyNotesVersion();
    std::atomic<bool> publisherDone { false };
    std::thread publisher ([&]()
    {
        for (int i = 0; i < 500; ++i)
            harness.processor.commitMelodyEdit (i % 2 == 0 ? melodyB : melodyA);
        publisherDone.store (true, std::memory_order_release);
    });
    harness.runBlocks (200); // ~2.3 s of audio while publishes land
    // Bounded wait so a stalled publish fails the test instead of
    // hanging CI: 30 s is far above the ~1 s this takes when
    // publishes are non-blocking. The join below still terminates:
    // a publish can only stall until the current audio block
    // finishes, and no blocks run after runBlocks returns.
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds (30);
    while (! publisherDone.load (std::memory_order_acquire)
           && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for (std::chrono::milliseconds (1));
    CHECK (publisherDone.load (std::memory_order_acquire));
    publisher.join(); // returns only if every publish completed

    // The publishes must have landed, not just returned: the version
    // advanced by all 500, and the last publish (i = 499, melodyA)
    // is the current melody.
    CHECK (harness.processor.getMelodyNotesVersion() == versionBefore + 500);
    if (auto current = harness.processor.getMelodyNotes(); current != nullptr)
        CHECK (current->lowestMidi == 60);
    else
        CHECK (false);

    harness.processor.stopPreview();
    harness.runBlocks (4); // drain: any sounding note gets its note-off
    CHECK (harness.eventsAreBalanced());
    harness.processor.releaseResources();
}

// ---- Host-BPM grid (P1-5) ----

void testTempoGridHostBpm()
{
    // The user's example: a 20 s clip at 120 BPM has a 0.5 s beat
    // (40 beats, 10 bars at 4/4) — not the old 1.25 s "beat" from
    // dividing the clip into 16 equal parts.
    TempoGrid grid120;
    grid120.bpm = 120.0;
    CHECK_NEAR ((float) grid120.beatSeconds(), 0.5f, 1e-9f);
    CHECK (grid120.totalBeats (20.0) == 40);
    CHECK (grid120.totalBars (20.0) == 10);
    CHECK_NEAR ((float) grid120.beatsPerBar(), 4.0f, 1e-9f);
    CHECK_NEAR ((float) grid120.barSeconds(), 2.0f, 1e-9f);

    TempoGrid grid90;
    grid90.bpm = 90.0;
    CHECK_NEAR ((float) grid90.beatSeconds(), 2.0f / 3.0f, 1e-6f);
    CHECK (grid90.totalBeats (20.0) == 30);
    CHECK (grid90.totalBars (20.0) == 8); // 7.5 bars -> 8 drawn

    TempoGrid grid140;
    grid140.bpm = 140.0;
    CHECK_NEAR ((float) grid140.beatSeconds(), 3.0f / 7.0f, 1e-6f);

    // Time signatures: 3/4 has 3 beats per bar, 6/8 has 3
    // quarter-note beats per bar, 5/8 has 2.5.
    TempoGrid grid34;
    grid34.bpm = 120.0;
    grid34.timeSignatureNumerator = 3;
    grid34.timeSignatureDenominator = 4;
    CHECK_NEAR ((float) grid34.beatsPerBar(), 3.0f, 1e-9f);
    CHECK_NEAR ((float) grid34.barSeconds(), 1.5f, 1e-9f);
    CHECK (grid34.barIndexForBeat (6.0) == 2);
    CHECK_NEAR ((float) grid34.barStartBeat (2), 6.0f, 1e-9f);

    TempoGrid grid68;
    grid68.bpm = 120.0;
    grid68.timeSignatureNumerator = 6;
    grid68.timeSignatureDenominator = 8;
    CHECK_NEAR ((float) grid68.beatsPerBar(), 3.0f, 1e-9f);

    TempoGrid grid58;
    grid58.bpm = 120.0;
    grid58.timeSignatureNumerator = 5;
    grid58.timeSignatureDenominator = 8;
    CHECK_NEAR ((float) grid58.beatsPerBar(), 2.5f, 1e-9f);
    CHECK_NEAR ((float) grid58.barStartBeat (2), 5.0f, 1e-9f);

    // Fallback when the host provides no tempo: 120 BPM / 4-4.
    TempoGrid fallback;
    CHECK_NEAR ((float) fallback.bpm, 120.0f, 1e-9f);
    CHECK_NEAR ((float) fallback.beatSeconds(), 0.5f, 1e-9f);
    CHECK (fallback.ppqPosition < 0.0); // unknown: never phases the grid

    // Absurd host values are clamped, never trusted blindly.
    TempoGrid bad;
    bad.bpm = 0.0;
    CHECK_NEAR ((float) bad.clampedBpm(), 120.0f, 1e-9f);
    bad.bpm = 1000.0;
    CHECK_NEAR ((float) bad.clampedBpm(), 300.0f, 1e-9f);
    bad.bpm = -5.0;
    CHECK_NEAR ((float) bad.clampedBpm(), 120.0f, 1e-9f);
    bad.bpm = 10.0; // below the 20 BPM floor
    CHECK_NEAR ((float) bad.clampedBpm(), 20.0f, 1e-9f);
    bad.bpm = std::nan ("");
    CHECK_NEAR ((float) bad.clampedBpm(), 120.0f, 1e-9f);
    bad.bpm = HUGE_VAL;
    CHECK_NEAR ((float) bad.clampedBpm(), 120.0f, 1e-9f);

    // Zero / negative durations have no beats or bars.
    CHECK (grid120.totalBeats (0.0) == 0);
    CHECK (grid120.totalBeats (-1.0) == 0);
    CHECK (grid120.totalBars (0.0) == 0);
    CHECK (grid120.totalBars (-1.0) == 0);

    // Time-signature fields are clamped to 1..16, never trusted.
    TempoGrid oddMeter;
    oddMeter.bpm = 120.0;
    oddMeter.timeSignatureNumerator = 0;
    CHECK_NEAR ((float) oddMeter.beatsPerBar(), 1.0f, 1e-9f);
    oddMeter.timeSignatureNumerator = 99;
    CHECK_NEAR ((float) oddMeter.beatsPerBar(), 16.0f, 1e-9f);
    oddMeter.timeSignatureNumerator = 4;
    oddMeter.timeSignatureDenominator = 0;
    CHECK_NEAR ((float) oddMeter.beatsPerBar(), 16.0f, 1e-9f);

    // Snap clamps to the clip end instead of returning a time past it.
    CHECK_NEAR ((float) grid120.snapSeconds (25.0, 20.0), 20.0f, 1e-6f);
    CHECK_NEAR ((float) grid120.snapSecondsIfClose (25.0, 20.0), 20.0f, 1e-6f);

    // Snap targets follow the real BPM: 1.1 s snaps to 1.0 s at
    // 120 BPM but to 4/3 s (2 beats) at 90 BPM.
    CHECK_NEAR ((float) grid120.snapSeconds (1.1, 20.0), 1.0f, 1e-6f);
    CHECK_NEAR ((float) grid90.snapSeconds (1.1, 20.0), 4.0f / 3.0f, 1e-6f);
    CHECK_NEAR ((float) grid120.snapSecondsIfClose (0.12, 20.0), 0.0f, 1e-6f);
    CHECK_NEAR ((float) grid120.snapSecondsIfClose (0.24, 20.0), 0.24f, 1e-6f);

    // A BPM change moves snap targets but never the notes: times are
    // stored in seconds, so the same melody is untouched.
    MelodyNoteSequence melody;
    melody.durationSeconds = 20.0;
    melody.notes = { { 1.0, 1.5, 60, 60.0f, 0.8f, 0.95f } };
    CHECK (MelodyEditor::moveNote (melody, 0,
                                   grid120.snapSeconds (1.1, 20.0), 60, 20.0));
    CHECK_NEAR ((float) melody.notes[0].startSeconds, 1.0f, 1e-6f);
    CHECK (MelodyEditor::moveNote (melody, 0,
                                   grid90.snapSeconds (1.1, 20.0), 60, 20.0));
    CHECK_NEAR ((float) melody.notes[0].startSeconds, 4.0f / 3.0f, 1e-6f);
    CHECK (melody.notes[0].midiNote == 60); // pitch untouched by a time move
}

// ---- Tool-strip placement (bar-number strip) ----

void testGridMappingMatchesNotePositions()
{
    // BLOCKER regression: the grid's beat lines and the note blocks
    // must share the same x for the same time. The old beat width
    // (gridW / totalBeats) stretched beats to fill the width, so a
    // note sat off its grid line whenever the clip was not a whole
    // number of beats (10.3 s @ 120 BPM: 12.9 px off; 3.1 s @ 60 BPM:
    // 152 px off). The beat width is now seconds-based
    // (gridW * beatSeconds / duration), so beatToX(secondsToBeat(t))
    // and playheadXForSeconds(t) agree for every t.
    auto checkMapping = [] (double duration, double bpm)
    {
        PitchEditorComponent editor;
        editor.setSize (1400, 560);
        auto buffer = makeSine (44100.0, duration, 440.0f, 0.5f);
        auto clip = AudioClip::fromBuffer (buffer, 44100.0, "test", true);
        CHECK (clip != nullptr);
        if (clip == nullptr)
            return;
        editor.setClip (clip);
        TempoGrid grid;
        grid.bpm = bpm;
        editor.setTempoGrid (grid);

        const float gridW = editor.playheadXForSeconds (duration)
                          - editor.playheadXForSeconds (0.0);
        CHECK (gridW > 0.0f);
        const float beatW = editor.beatToXValue (1.0) - editor.beatToXValue (0.0);
        CHECK_NEAR (beatW,
                    gridW * static_cast<float> (grid.beatSeconds() / duration),
                    1e-3f);
        for (const double t : { 0.0, 1.0, 2.5, 5.0, duration - 0.1 })
        {
            if (t < 0.0 || t > duration)
                continue;
            CHECK (std::abs (editor.beatToXValue (grid.secondsToBeat (t))
                             - editor.playheadXForSeconds (t)) <= 1.0f);
        }
    };
    checkMapping (10.3, 120.0);
    checkMapping (3.1, 60.0);
}

void testToolStripPlacement()
{
    // Wide gaps: the strip goes in the first gap that fits — between bar 1's
    // number (ends x=120) and bar 2's number (starts x=302).
    {
        const std::vector<float> bars { 94.0f, 300.0f, 506.0f, 712.0f, 918.0f };
        const int left = toolStripLeftForBarStarts (bars, 96.0f, 1200.0f, 130.0f);
        CHECK (left == 122); // gap [120, 302] fits, left-aligned + 2
    }
    // No early gap fits, but the trailing gap does.
    {
        const std::vector<float> bars { 94.0f, 244.0f, 394.0f, 544.0f,
                                        694.0f, 844.0f, 994.0f };
        const int left = toolStripLeftForBarStarts (bars, 96.0f, 1200.0f, 130.0f);
        CHECK (left == 1022); // after the last number (ends 1020) + 2
    }
    // No gap fits at all: right-aligned, nudged left past numbers,
    // clamped to the zone.
    {
        const std::vector<float> bars { 94.0f, 200.0f };
        const int left = toolStripLeftForBarStarts (bars, 96.0f, 300.0f, 130.0f);
        CHECK (left == 96);
    }
    // Strip wider than the whole zone: no placement avoids a number,
    // so it clamps to the zone left (best effort — see the doc
    // comment in TempoGrid.h).
    {
        const std::vector<float> bars { 94.0f };
        const int left = toolStripLeftForBarStarts (bars, 96.0f, 200.0f, 200.0f);
        CHECK (left == 96);
    }
    // No bars: the strip goes at the left of the zone.
    {
        const int left = toolStripLeftForBarStarts ({}, 96.0f, 1200.0f, 130.0f);
        CHECK (left == 98);
    }
}

// ---- Project saving (P1-6) ----

void setProcessorParam (MelotheonAudioProcessor& processor, const char* id,
                        float value)
{
    if (auto* param = processor.apvts.getParameter (id); param != nullptr)
        param->setValueNotifyingHost (value);
}

// Record a stereo sine take (left full, right half) so the saved
// clip has known, non-trivial samples to compare after a reload.
void recordSineTake (MelotheonAudioProcessor& processor, double sampleRate,
                     double durationSeconds, float peak)
{
    const int blockSize = 512;
    CHECK (processor.startRecording());
    juce::AudioBuffer<float> input (2, blockSize);
    juce::MidiBuffer midi;
    double phase = 0.0;
    const double phaseStep = 2.0 * 3.141592653589793 * 440.0 / sampleRate;
    const int blocks = static_cast<int> (
        std::ceil (durationSeconds * sampleRate / blockSize));
    for (int b = 0; b < blocks; ++b)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            const float s = peak * static_cast<float> (std::sin (phase));
            phase += phaseStep;
            input.getWritePointer (0)[i] = s;
            input.getWritePointer (1)[i] = 0.5f * s;
        }
        processor.processBlock (input, midi);
    }
    processor.stopRecording();
    CHECK (processor.hasActiveClip());
}

// Synthetic analysis for a clip of `durationSeconds`: MIDI 60 for
// the first half, MIDI 64 for the second.
PitchAnalysisResult makeTwoToneAnalysis (double durationSeconds)
{
    PitchAnalysisResult analysis;
    analysis.durationSeconds = durationSeconds;
    analysis.hopSeconds = 0.01;
    analysis.analysisSampleRate = 24000.0;
    const int frames = static_cast<int> (std::llround (durationSeconds / 0.01));
    for (int i = 0; i < frames; ++i)
    {
        PitchFrame frame;
        frame.timeSeconds = i * 0.01;
        frame.voiced = true;
        const bool secondHalf = i * 0.01 >= durationSeconds * 0.5;
        frame.midiPitch = secondHalf ? 64.0f : 60.0f;
        frame.frequencyHz = secondHalf ? 329.63f : 261.63f;
        frame.confidence = 0.9f;
        frame.rmsDb = -20.0f;
        analysis.frames.push_back (frame);
    }
    analysis.voicedFrameCount = frames;
    analysis.minMidi = 60.0f;
    analysis.maxMidi = 64.0f;
    return analysis;
}

void testProjectStateFlacRoundTrip()
{
    // Stereo sine at 48 kHz, peak 0.8: FLAC 24-bit round-trips to
    // within one 24-bit step.
    {
        juce::AudioBuffer<float> source (2, 48000);
        for (int i = 0; i < 48000; ++i)
        {
            const float s = 0.8f * static_cast<float> (
                std::sin (2.0 * 3.141592653589793 * 440.0 * i / 48000.0));
            source.getWritePointer (0)[i] = s;
            source.getWritePointer (1)[i] = 0.5f * s;
        }
        auto clip = AudioClip::fromBuffer (source, 48000.0, "sine", false);
        CHECK (clip != nullptr);
        juce::MemoryBlock flac;
        double peakScale = 0.0;
        CHECK (ProjectState::encodeClipAudio (*clip, flac, peakScale));
        CHECK_NEAR ((float) peakScale, 1.0f, 1e-9f);
        CHECK (flac.getSize() > 0);
        CHECK (flac.getSize() < static_cast<size_t> (48000) * 2 * 4); // compressed
        auto decoded = ProjectState::decodeClipAudio (
            flac.getData(), flac.getSize(), peakScale, "sine", false);
        CHECK (decoded != nullptr);
        CHECK (decoded->sourceSampleRate == 48000.0);
        CHECK (decoded->totalSamples == 48000);
        CHECK (decoded->previewBuffer.getNumChannels() == 2);
        float maxError = 0.0f;
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 48000; ++i)
                maxError = std::max (maxError, std::abs (
                    decoded->previewBuffer.getReadPointer (ch)[i]
                    - source.getReadPointer (ch)[i]));
        CHECK (maxError < 1e-6f);
    }

    // Hot signal (peak 2.0): the peak scale keeps it from clipping.
    {
        juce::AudioBuffer<float> source (1, 44100);
        for (int i = 0; i < 44100; ++i)
            source.getWritePointer (0)[i] = 2.0f * static_cast<float> (
                std::sin (2.0 * 3.141592653589793 * 220.0 * i / 44100.0));
        auto clip = AudioClip::fromBuffer (source, 44100.0, "hot", true);
        CHECK (clip != nullptr);
        juce::MemoryBlock flac;
        double peakScale = 0.0;
        CHECK (ProjectState::encodeClipAudio (*clip, flac, peakScale));
        CHECK (peakScale > 1.99 && peakScale < 2.01);
        auto decoded = ProjectState::decodeClipAudio (
            flac.getData(), flac.getSize(), peakScale, "hot", true);
        CHECK (decoded != nullptr);
        CHECK (decoded->importedFromFile);
        float maxError = 0.0f;
        float maxPeak = 0.0f;
        for (int i = 0; i < 44100; ++i)
        {
            const float v = decoded->previewBuffer.getReadPointer (0)[i];
            maxError = std::max (maxError, std::abs (v - source.getReadPointer (0)[i]));
            maxPeak = std::max (maxPeak, std::abs (v));
        }
        CHECK (maxError < 1e-6f);
        CHECK (maxPeak > 1.99f && maxPeak < 2.01f); // not clipped to 1.0
    }

    // Corrupt FLAC bytes decode to nothing, never crash.
    {
        std::vector<std::uint8_t> garbage (512, 0xAA);
        CHECK (ProjectState::decodeClipAudio (garbage.data(), garbage.size(),
                                             1.0, "bad", false) == nullptr);
        CHECK (ProjectState::decodeClipAudio (nullptr, 0, 1.0, "bad", false) == nullptr);
        CHECK (ProjectState::decodeClipAudio (garbage.data(), garbage.size(),
                                             0.5, "bad", false) == nullptr);
    }
}

void testProjectStateRoundTrip()
{
    MelotheonAudioProcessor source;
    source.prepareToPlay (44100.0, 512);
    recordSineTake (source, 44100.0, 1.0, 0.8f);
    const auto sourceClip = source.getActiveClip();
    CHECK (sourceClip != nullptr);
    const double duration = sourceClip->durationSeconds;

    // Analysis + an edited melody (first note moved up 2 semitones)
    // + a non-default MIDI channel.
    CHECK (source.commitPitchCurveEdit (makeTwoToneAnalysis (duration)));
    CHECK (source.hasMelodyNotes());
    auto edited = *source.getMelodyNotes();
    CHECK (! edited.notes.empty());
    edited.notes.front().midiNote += 2;
    edited.notes.front().sourceMidiPitch += 2.0f;
    CHECK (source.commitMelodyEdit (edited));
    CHECK (source.isPitchAnalysisEdited());
    setProcessorParam (source, "midiChannel", 2.0f / 15.0f); // channel 3
    CHECK (source.getMidiChannel() == 3);
    CHECK (source.getStatus() == MelotheonAudioProcessor::Status::AnalysisReady);

    juce::MemoryBlock blob;
    source.getStateInformation (blob);
    CHECK (blob.getSize() > 0);
    CHECK (blob.getSize() < ProjectState::kMaxStateBytes);
    // A second snapshot encodes deterministically (same size). This
    // proves the encoding is stable, not that the FLAC cache was
    // reused — reuse is unobservable through the public API.
    juce::MemoryBlock blob2;
    source.getStateInformation (blob2);
    CHECK (blob2.getSize() == blob.getSize());

    MelotheonAudioProcessor restored;
    restored.prepareToPlay (44100.0, 512);
    restored.setStateInformation (blob.getData(), static_cast<int> (blob.getSize()));

    CHECK (restored.hasActiveClip());
    CHECK (restored.getMidiChannel() == 3);
    CHECK (restored.isPitchAnalysisEdited());
    CHECK (restored.getStatus() == MelotheonAudioProcessor::Status::AnalysisReady);
    const auto restoredClip = restored.getActiveClip();
    CHECK (restoredClip != nullptr);
    CHECK_NEAR ((float) restoredClip->durationSeconds, (float) duration, 1e-6f);
    CHECK (restoredClip->sourceSampleRate == sourceClip->sourceSampleRate);
    CHECK (restoredClip->totalSamples == sourceClip->totalSamples);
    CHECK (restoredClip->previewBuffer.getNumChannels()
           == sourceClip->previewBuffer.getNumChannels());
    float maxSampleError = 0.0f;
    for (int ch = 0; ch < sourceClip->previewBuffer.getNumChannels(); ++ch)
        for (int i = 0; i < sourceClip->totalSamples; ++i)
            maxSampleError = std::max (maxSampleError, std::abs (
                restoredClip->previewBuffer.getReadPointer (ch)[i]
                - sourceClip->previewBuffer.getReadPointer (ch)[i]));
    CHECK (maxSampleError < 1e-6f);

    const auto restoredAnalysis = restored.getPitchAnalysis();
    CHECK (restoredAnalysis != nullptr);
    CHECK (restoredAnalysis->frames.size() == source.getPitchAnalysis()->frames.size());
    CHECK (restoredAnalysis->hasPitch());
    CHECK_NEAR ((float) restoredAnalysis->durationSeconds,
                (float) source.getPitchAnalysis()->durationSeconds, 1e-9f);
    CHECK_NEAR ((float) restoredAnalysis->hopSeconds,
                (float) source.getPitchAnalysis()->hopSeconds, 1e-9f);
    CHECK_NEAR (restoredAnalysis->minMidi, source.getPitchAnalysis()->minMidi, 1e-6f);
    CHECK_NEAR (restoredAnalysis->maxMidi, source.getPitchAnalysis()->maxMidi, 1e-6f);
    for (size_t i = 0; i < restoredAnalysis->frames.size(); ++i)
    {
        const auto& a = restoredAnalysis->frames[i];
        const auto& b = source.getPitchAnalysis()->frames[i];
        CHECK (a.voiced == b.voiced);
        CHECK_NEAR ((float) a.timeSeconds, (float) b.timeSeconds, 1e-9f);
        CHECK_NEAR (a.midiPitch, b.midiPitch, 1e-6f);
        CHECK_NEAR (a.confidence, b.confidence, 1e-6f);
        CHECK_NEAR (a.frequencyHz, b.frequencyHz, 1e-4f);
        CHECK_NEAR (a.rmsDb, b.rmsDb, 1e-6f);
    }

    const auto restoredMelody = restored.getMelodyNotes();
    const auto sourceMelody = source.getMelodyNotes();
    CHECK (restoredMelody != nullptr && sourceMelody != nullptr);
    CHECK (restoredMelody->notes.size() == sourceMelody->notes.size());
    for (size_t i = 0; i < restoredMelody->notes.size(); ++i)
    {
        const auto& a = restoredMelody->notes[i];
        const auto& b = sourceMelody->notes[i];
        CHECK_NEAR ((float) a.startSeconds, (float) b.startSeconds, 1e-9f);
        CHECK_NEAR ((float) a.endSeconds, (float) b.endSeconds, 1e-9f);
        CHECK (a.midiNote == b.midiNote);
        CHECK_NEAR (a.sourceMidiPitch, b.sourceMidiPitch, 1e-6f);
        CHECK_NEAR (a.velocity, b.velocity, 1e-6f);
        CHECK_NEAR (a.confidence, b.confidence, 1e-6f);
    }
    // The edited first note survived the round-trip (60 -> 62).
    CHECK (restoredMelody->notes.front().midiNote
           == sourceMelody->notes.front().midiNote);
    CHECK (restoredMelody->notes.front().midiNote == 62);

    // The restored clip can actually preview: the decode is not
    // just metadata, it renders audio.
    {
        CHECK (restored.startPreview());
        juce::AudioBuffer<float> output (2, 512);
        juce::MidiBuffer midi;
        float maxOutput = 0.0f;
        for (int block = 0; block < 4; ++block)
        {
            output.clear();
            midi.clear();
            restored.processBlock (output, midi);
            for (int ch = 0; ch < 2; ++ch)
                maxOutput = std::max (maxOutput, output.getMagnitude (ch, 0, 512));
        }
        CHECK (maxOutput > 0.0f);
        restored.stopPreview();
    }

    source.releaseResources();
    restored.releaseResources();

    // Overlapping notes (sorted by start, second starts before the
    // first ends) still load: the reader only requires sorted starts,
    // so one overlapping note never drops the whole project section.
    {
        auto melody = std::make_shared<MelodyNoteSequence>();
        melody->durationSeconds = 2.0;
        MelodyNote first;
        first.startSeconds = 0.0; first.endSeconds = 1.0; first.midiNote = 60;
        first.sourceMidiPitch = 60.0f; first.velocity = 0.8f; first.confidence = 0.9f;
        MelodyNote second;
        second.startSeconds = 0.5; second.endSeconds = 1.5; second.midiNote = 64;
        second.sourceMidiPitch = 64.0f; second.velocity = 0.8f; second.confidence = 0.9f;
        melody->notes.push_back (first);
        melody->notes.push_back (second);
        ProjectState::Data data;
        data.melody = melody;
        juce::ValueTree params ("MelotheonState");
        auto wrapper = ProjectState::serializeState (params, data, nullptr, -1);
        juce::MemoryBlock blob;
        {
            juce::MemoryOutputStream stream (blob, false);
            wrapper.writeToStream (stream);
        }
        auto parsed = ProjectState::deserializeState (
            blob.getData(), static_cast<int> (blob.getSize()));
        CHECK (parsed.hasProject);
        CHECK (parsed.project.melody != nullptr);
        CHECK (parsed.project.melody->notes.size() == 2);
        CHECK (parsed.project.melody->notes[1].midiNote == 64);
    }
}

void testProjectStateEncodeGuards()
{
    // PS-4: a clip shorter than one FLAC block (~4096 samples)
    // must still encode. The old code checked the output size
    // before destroying the writer (whose destructor writes the
    // final partial block), so a ~50 ms clip encoded to 0 bytes
    // and was silently dropped.
    {
        auto buffer = makeSine (44100.0, 0.05, 440.0f, 0.5f);
        auto clip = AudioClip::fromBuffer (buffer, 44100.0, "short", true);
        CHECK (clip != nullptr);
        juce::MemoryBlock flac;
        double peakScale = 1.0;
        CHECK (ProjectState::encodeClipAudio (*clip, flac, peakScale));
        CHECK (flac.getSize() > 0);
    }
    // PS-2: a NaN sample is invisible to the peak scan and would
    // reach the integer conversion as garbage — the encode fails
    // outright instead.
    {
        auto buffer = makeSine (44100.0, 0.5, 440.0f, 0.5f);
        buffer.getWritePointer (0)[100]
            = std::numeric_limits<float>::quiet_NaN();
        auto clip = AudioClip::fromBuffer (buffer, 44100.0, "nan", true);
        CHECK (clip != nullptr);
        juce::MemoryBlock flac;
        double peakScale = 1.0;
        CHECK (! ProjectState::encodeClipAudio (*clip, flac, peakScale));
    }
    // PS-7/PS-8: more than 2 channels, or a rate outside the
    // 8000-192000 Hz range the reader accepts, fails the encode
    // instead of failing silently inside the writer.
    {
        juce::AudioBuffer<float> threeChannel (3, 4410);
        threeChannel.clear();
        auto clip = AudioClip::fromBuffer (threeChannel, 44100.0, "3ch", true);
        CHECK (clip != nullptr);
        juce::MemoryBlock flac;
        double peakScale = 1.0;
        CHECK (! ProjectState::encodeClipAudio (*clip, flac, peakScale));
    }
    {
        auto buffer = makeSine (4000.0, 0.5, 440.0f, 0.5f);
        auto clip = AudioClip::fromBuffer (buffer, 4000.0, "low-rate", true);
        CHECK (clip != nullptr);
        juce::MemoryBlock flac;
        double peakScale = 1.0;
        CHECK (! ProjectState::encodeClipAudio (*clip, flac, peakScale));
    }
}

void testProjectStateOldCorruptAndSkipped()
{
    // Old format: APVTS state only (everything saved before this
    // change). Loads exactly as before — parameters only, no clip.
    {
        MelotheonAudioProcessor source;
        source.prepareToPlay (44100.0, 512);
        setProcessorParam (source, "midiChannel", 4.0f / 15.0f); // channel 5
        juce::MemoryBlock oldBlob;
        {
            // copyState() flushes the just-set parameter into the tree;
            // apvts.state can still hold the default until the APVTS
            // timer runs.
            auto oldState = source.apvts.copyState();
            juce::MemoryOutputStream stream (oldBlob, false);
            oldState.writeToStream (stream);
        }
        MelotheonAudioProcessor fresh;
        fresh.prepareToPlay (44100.0, 512);
        fresh.setStateInformation (oldBlob.getData(),
                                  static_cast<int> (oldBlob.getSize()));
        CHECK (fresh.getMidiChannel() == 5);
        CHECK (! fresh.hasActiveClip());
        CHECK (fresh.getPitchAnalysis() == nullptr);
        source.releaseResources();
        fresh.releaseResources();
    }

    // Unknown wrapper version: parameters are trusted, the project
    // section is ignored, no crash.
    {
        MelotheonAudioProcessor source;
        source.prepareToPlay (44100.0, 512);
        recordSineTake (source, 44100.0, 0.5, 0.8f);
        CHECK (source.commitPitchCurveEdit (
            makeTwoToneAnalysis (source.getActiveClip()->durationSeconds)));
        setProcessorParam (source, "midiChannel", 2.0f / 15.0f); // channel 3
        juce::MemoryBlock blob;
        source.getStateInformation (blob);
        auto wrapper = juce::ValueTree::readFromData (
            blob.getData(), blob.getSize());
        CHECK (wrapper.isValid());
        wrapper.setProperty ("version", 99, nullptr);
        juce::MemoryBlock mutated;
        {
            juce::MemoryOutputStream stream (mutated, false);
            wrapper.writeToStream (stream);
        }
        MelotheonAudioProcessor fresh;
        fresh.prepareToPlay (44100.0, 512);
        fresh.setStateInformation (mutated.getData(),
                                  static_cast<int> (mutated.getSize()));
        CHECK (fresh.getMidiChannel() == 3);
        CHECK (! fresh.hasActiveClip());
        source.releaseResources();
        fresh.releaseResources();
    }

    // Truncated and garbage blobs: no crash, nothing applied.
    {
        MelotheonAudioProcessor source;
        source.prepareToPlay (44100.0, 512);
        recordSineTake (source, 44100.0, 0.5, 0.8f);
        // Set a non-default parameter first: the default channel is
        // already 1, so checking for 1 below would pass vacuously
        // even if the blob were applied.
        setProcessorParam (source, "midiChannel", 3.0f / 15.0f); // channel 4
        juce::MemoryBlock blob;
        source.getStateInformation (blob);
        auto checkTruncated = [&] (std::size_t cut, int expectedChannel)
        {
            MelotheonAudioProcessor fresh;
            fresh.prepareToPlay (44100.0, 512);
            fresh.setStateInformation (blob.getData(), static_cast<int> (cut));
            CHECK (fresh.getMidiChannel() == expectedChannel);
            CHECK (! fresh.hasActiveClip());
            fresh.releaseResources();
        };
        checkTruncated (10, 1);
        checkTruncated (100, 1);
        // Mid-blob cut: the params sit at the front of the blob, so
        // they survive (channel 4 applies) and only the project
        // section is dropped.
        checkTruncated (blob.getSize() / 2, 4);
        std::vector<std::uint8_t> garbage (256, 0xAA);
        MelotheonAudioProcessor fresh;
        fresh.prepareToPlay (44100.0, 512);
        fresh.setStateInformation (garbage.data(), static_cast<int> (garbage.size()));
        CHECK (fresh.getMidiChannel() == 1);
        CHECK (! fresh.hasActiveClip());
        fresh.releaseResources();
        source.releaseResources();
    }

    // Audio-skip path (forced with a tiny cap): analysis and melody
    // survive, the audio is flagged as skipped, the clip is null.
    {
        MelotheonAudioProcessor source;
        source.prepareToPlay (44100.0, 512);
        recordSineTake (source, 44100.0, 0.5, 0.8f);
        CHECK (source.commitPitchCurveEdit (
            makeTwoToneAnalysis (source.getActiveClip()->durationSeconds)));
        ProjectState::Data data;
        data.clip = source.getActiveClip();
        data.analysis = source.getPitchAnalysis();
        data.melody = source.getMelodyNotes();
        ProjectState::EncodedAudioCache cache;
        auto wrapper = ProjectState::serializeState (
            source.apvts.state, data, &cache, source.getClipVersion(), 1024);
        juce::MemoryBlock blob;
        {
            juce::MemoryOutputStream stream (blob, false);
            wrapper.writeToStream (stream);
        }
        auto parsed = ProjectState::deserializeState (
            blob.getData(), static_cast<int> (blob.getSize()));
        CHECK (parsed.hasParams);
        CHECK (parsed.hasProject);
        CHECK (parsed.project.audioSkippedTooLarge);
        CHECK (parsed.project.clip == nullptr);
        CHECK (parsed.project.analysis != nullptr);
        CHECK (parsed.project.analysis->hasPitch());
        CHECK (parsed.project.melody != nullptr);
        CHECK (! parsed.project.melody->empty());
        source.releaseResources();
    }

    // Same audio-skip blob driven through a processor restore (not
    // just deserializeState): the analysis and melody land in the
    // processor, the clip stays null, and the skip is surfaced in
    // the status hint the editor shows.
    {
        MelotheonAudioProcessor source;
        source.prepareToPlay (44100.0, 512);
        recordSineTake (source, 44100.0, 0.5, 0.8f);
        CHECK (source.commitPitchCurveEdit (
            makeTwoToneAnalysis (source.getActiveClip()->durationSeconds)));
        ProjectState::Data data;
        data.clip = source.getActiveClip();
        data.analysis = source.getPitchAnalysis();
        data.melody = source.getMelodyNotes();
        ProjectState::EncodedAudioCache cache;
        auto wrapper = ProjectState::serializeState (
            source.apvts.state, data, &cache, source.getClipVersion(), 1024);
        juce::MemoryBlock blob;
        {
            juce::MemoryOutputStream stream (blob, false);
            wrapper.writeToStream (stream);
        }
        MelotheonAudioProcessor fresh;
        fresh.prepareToPlay (44100.0, 512);
        fresh.setStateInformation (blob.getData(), static_cast<int> (blob.getSize()));
        CHECK (! fresh.hasActiveClip());
        CHECK (fresh.getPitchAnalysis() != nullptr);
        CHECK (fresh.getPitchAnalysis()->hasPitch());
        CHECK (fresh.hasMelodyNotes());
        CHECK (fresh.getStatusHint() == "AUDIO TOO LARGE - NOT SAVED");
        source.releaseResources();
        fresh.releaseResources();
    }
}

} // namespace

// ---- Undo / redo (P1-7) ----

MelodyNoteSequence makeUndoTestMelody (int baseMidi)
{
    MelodyNoteSequence melody;
    melody.durationSeconds = 1.5;
    melody.lowestMidi = baseMidi;
    melody.highestMidi = baseMidi + 4;
    melody.notes = {
        { 0.25, 0.75, baseMidi, static_cast<float> (baseMidi), 0.80f, 0.95f },
        { 1.00, 1.40, baseMidi + 4, static_cast<float> (baseMidi + 4), 0.70f, 0.90f },
    };
    return melody;
}

void testUndoRedoMelodyEdits()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // Empty stacks are a safe no-op.
    CHECK (! processor.canUndoEdit());
    CHECK (! processor.canRedoEdit());
    CHECK (! processor.undoEdit());
    CHECK (! processor.redoEdit());
    CHECK (! processor.hasMelodyNotes());

    // First commit: nothing to restore, so no history step.
    auto m1 = makeUndoTestMelody (60);
    CHECK (processor.commitMelodyEdit (m1));
    CHECK (! processor.canUndoEdit());
    CHECK (processor.getUndoStackSize() == 0);

    // MOVE the first note up: undo restores, redo reapplies.
    auto m2 = m1;
    CHECK (MelodyEditor::moveNote (m2, 0, 0.25, 62, 1.5));
    CHECK (processor.commitMelodyEdit (m2));
    CHECK (processor.canUndoEdit());
    CHECK (processor.getUndoStackSize() == 1);
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 62);
    CHECK (processor.undoEdit());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 60);
    CHECK (processor.getStatusHint() == "UNDO");
    CHECK (processor.canRedoEdit());
    CHECK (processor.redoEdit());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 62);
    CHECK (processor.getStatusHint() == "REDO");

    // CUT the second note: undo removes the split.
    auto m3 = m2;
    CHECK (MelodyEditor::splitNoteAt (m3, 1, 1.20));
    CHECK (processor.commitMelodyEdit (m3));
    CHECK (processor.getMelodyNotes()->notes.size() == 3);
    CHECK (processor.undoEdit());
    CHECK (processor.getMelodyNotes()->notes.size() == 2);
    CHECK (processor.redoEdit());
    CHECK (processor.getMelodyNotes()->notes.size() == 3);

    // STRETCH the first note's right edge: undo restores the length.
    auto m4 = m3;
    CHECK (MelodyEditor::stretchNoteEdge (m4, 0, false, 0.90, 1.5));
    CHECK (processor.commitMelodyEdit (m4));
    CHECK_NEAR (processor.getMelodyNotes()->notes[0].endSeconds, 0.90, 0.001);
    CHECK (processor.undoEdit());
    CHECK_NEAR (processor.getMelodyNotes()->notes[0].endSeconds, 0.75, 0.001);

    processor.releaseResources();
}

void testUndoRedoCurveEdit()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // Base analysis: MIDI 60 first half, 64 second half (1.5 s).
    auto analysis = makeTwoToneAnalysis (1.5);
    CHECK (processor.commitPitchCurveEdit (analysis));
    CHECK (processor.hasMelodyNotes());
    CHECK (! processor.canUndoEdit()); // first commit: nothing to restore

    // Edit the curve at 1.20 s toward MIDI 66: frames move and the
    // melody re-segments.
    auto edited = analysis;
    NoteSegmenter::applyCurveEdit (edited, 1.20, 66.0f, 0.30);
    CHECK (processor.commitPitchCurveEdit (edited));
    CHECK (processor.canUndoEdit());
    const auto* editedFrame = processor.getPitchAnalysis()->findNearestVoicedFrame (1.20);
    CHECK (editedFrame != nullptr);
    if (editedFrame != nullptr)
        CHECK_NEAR (editedFrame->midiPitch, 66.0f, 0.5f);

    // Undo: frames AND melody return to the two-tone base.
    CHECK (processor.undoEdit());
    const auto* restoredFrame = processor.getPitchAnalysis()->findNearestVoicedFrame (1.20);
    CHECK (restoredFrame != nullptr);
    if (restoredFrame != nullptr)
        CHECK_NEAR (restoredFrame->midiPitch, 64.0f, 0.5f);
    bool found64 = false;
    for (const auto& note : processor.getMelodyNotes()->notes)
        if (note.startSeconds <= 1.20 && note.endSeconds >= 1.20)
            found64 = (note.midiNote == 64);
    CHECK (found64);

    // Redo: the edit comes back.
    CHECK (processor.redoEdit());
    const auto* redoneFrame = processor.getPitchAnalysis()->findNearestVoicedFrame (1.20);
    CHECK (redoneFrame != nullptr);
    if (redoneFrame != nullptr)
        CHECK_NEAR (redoneFrame->midiPitch, 66.0f, 0.5f);

    processor.releaseResources();
}

void testUndoRedoClearedByNewEdit()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    auto m1 = makeUndoTestMelody (60);
    CHECK (processor.commitMelodyEdit (m1));
    auto m2 = m1;
    CHECK (MelodyEditor::moveNote (m2, 0, 0.25, 62, 1.5));
    CHECK (processor.commitMelodyEdit (m2));
    CHECK (processor.undoEdit());
    CHECK (processor.canRedoEdit());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 60);

    // A new edit clears the redo stack: the undone move is gone.
    auto m3 = m1;
    CHECK (MelodyEditor::moveNote (m3, 1, 1.00, 67, 1.5));
    CHECK (processor.commitMelodyEdit (m3));
    CHECK (! processor.canRedoEdit());
    CHECK (! processor.redoEdit());
    CHECK (processor.getMelodyNotes()->notes[1].midiNote == 67);
    // ...and the new edit is itself undoable.
    CHECK (processor.undoEdit());
    CHECK (processor.getMelodyNotes()->notes[1].midiNote == 64);

    processor.releaseResources();
}

void testUndoRedoLimit50()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    // 55 distinct commits: C_i has first note pitch 59 + i.
    for (int i = 1; i <= 55; ++i)
    {
        auto melody = makeUndoTestMelody (59 + i);
        CHECK (processor.commitMelodyEdit (melody));
    }
    CHECK (processor.getUndoStackSize() == MelotheonAudioProcessor::maxEditHistorySteps);
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 114); // C55

    // 50 undos reach the oldest kept state (C5, pitch 64), then the
    // stack is empty. (54 pre-edit snapshots were pushed; the 4
    // oldest were dropped by the 50-step cap.)
    for (int i = 0; i < 50; ++i)
        CHECK (processor.undoEdit());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 64);
    CHECK (! processor.canUndoEdit());
    CHECK (! processor.undoEdit());
    CHECK (processor.getRedoStackSize() == 50);

    // 50 redos return to C55.
    for (int i = 0; i < 50; ++i)
        CHECK (processor.redoEdit());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 114);
    CHECK (! processor.canRedoEdit());

    processor.releaseResources();
}

void testUndoAfterReanalyze()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    recordSineTake (processor, 44100.0, 0.5, 0.5f);
    auto m1 = makeUndoTestMelody (60);
    CHECK (processor.commitMelodyEdit (m1));

    // RE-ANALYZE pushes the pre-analyze state, then clears the
    // melody. Cancel the worker right away; undo must still restore
    // the edit (the cancel generation-gates the completion callback).
    CHECK (processor.startPitchAnalysis());
    CHECK (processor.canUndoEdit());
    processor.cancelPitchAnalysis();
    CHECK (processor.undoEdit());
    CHECK (processor.hasMelodyNotes());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 60);
    CHECK (processor.getMelodyNotes()->notes[1].midiNote == 64);

    // Wait for the cancelled worker to settle, then re-assert: a
    // late completion callback clobbering the restored melody after
    // the first assertions would otherwise go unnoticed. The
    // completion is delivered via MessageManager::callAsync, and
    // this console harness has no message pump (runDispatchLoopUntil
    // needs JUCE_MODAL_LOOPS_PERMITTED, off for a console app), so
    // the queued callback cannot be delivered here — the sleep lets
    // the worker finish or abort, and the re-assert proves the
    // restored state is stable once it has. The gate's message-
    // thread half is exercised for real by the host (FL pumps the
    // queue); see the report note.
    std::this_thread::sleep_for (std::chrono::milliseconds (500));
    CHECK (processor.hasMelodyNotes());
    CHECK (processor.getMelodyNotes()->notes[0].midiNote == 60);
    CHECK (processor.getMelodyNotes()->notes[1].midiNote == 64);

    processor.releaseResources();
}

void testUndoHistoryClearedOnNewClip()
{
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);

    recordSineTake (processor, 44100.0, 0.5, 0.5f);
    auto m1 = makeUndoTestMelody (60);
    CHECK (processor.commitMelodyEdit (m1));
    auto m2 = m1;
    CHECK (MelodyEditor::moveNote (m2, 0, 0.25, 62, 1.5));
    CHECK (processor.commitMelodyEdit (m2));
    CHECK (processor.canUndoEdit());

    // A new clip (second take) clears the history: the snapshots
    // belonged to the old clip.
    recordSineTake (processor, 44100.0, 0.5, 0.5f);
    CHECK (! processor.canUndoEdit());
    CHECK (! processor.canRedoEdit());
    CHECK (! processor.undoEdit());

    processor.releaseResources();
}

// ---- MIDI UX (P2-8..12) ----

void testMidiRoutingHint()
{
    // NOTE: this test must run BEFORE testSendToChannelAutoEnablesMidiOut
    // (see main()): the auto-enable listener defers to the message
    // thread, and this test relies on no MessageManager existing yet,
    // so turning SEND on leaves the master off. Guard that ordering
    // assumption: if a MessageManager already exists, the SEND-on
    // steps below no longer prove what they claim.
    CHECK (juce::MessageManager::getInstanceWithoutCreating() == nullptr);
    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);
    auto setParam = [&] (const char* id, float value)
    {
        if (auto* param = processor.apvts.getParameter (id); param != nullptr)
            param->setValueNotifyingHost (value);
    };

    // Switches off: no hint.
    CHECK (processor.getMidiRoutingHint().isEmpty());

    // SEND on but master off: no hint.
    setParam ("sendToChannel", 1.0f);
    CHECK (processor.getMidiRoutingHint().isEmpty());

    // Both on, idle: the routing instruction.
    setParam ("midiOut", 1.0f);
    CHECK (processor.getMidiRoutingHint() == "ROUTE MIDI PORT, THEN PRESS PLAY");

    // Playing: no hint.
    recordSineTake (processor, 44100.0, 0.5, 0.5f);
    CHECK (processor.startPreview());
    CHECK (processor.getMidiRoutingHint().isEmpty());

    // Stopped again: the hint returns.
    processor.stopPreview();
    CHECK (processor.getMidiRoutingHint() == "ROUTE MIDI PORT, THEN PRESS PLAY");

    // A busy status hides it (it must never hide progress).
    processor.setStatus (MelotheonAudioProcessor::Status::Analyzing);
    CHECK (processor.getMidiRoutingHint().isEmpty());
    processor.setStatus (MelotheonAudioProcessor::Status::Error);
    CHECK (processor.getMidiRoutingHint().isEmpty());

    processor.releaseResources();
}

void testMidiOutStatusTransitions()
{
    MidiOutHarness harness (44100.0, 512);
    harness.recordTake (0.6);

    MelodyNoteSequence melody;
    melody.durationSeconds = 0.6;
    melody.lowestMidi = 60;
    melody.highestMidi = 62;
    melody.notes = {
        { 0.10, 0.30, 60, 60.0f, 0.80f, 0.95f },
        { 0.35, 0.55, 62, 62.0f, 0.75f, 0.90f },
    };
    CHECK (harness.processor.commitMelodyEdit (melody));

    using Status = MelotheonAudioProcessor::MidiOutStatus;
    // Both switches off: Off.
    harness.runBlock();
    CHECK (harness.processor.getMidiOutStatus() == Status::Off);

    // Both on, not playing: Waiting.
    harness.enableMidiOut (1);
    harness.runBlock();
    CHECK (harness.processor.getMidiOutStatus() == Status::Waiting);

    // Playing: Streaming.
    CHECK (harness.processor.startPreview());
    harness.runBlock();
    CHECK (harness.processor.getMidiOutStatus() == Status::Streaming);
    harness.runBlocks (3);
    CHECK (harness.processor.getMidiOutStatus() == Status::Streaming);

    // Natural end: the clip ends and the preview stops by itself —
    // back to Waiting (both switches still on, nothing playing).
    // Run until the status flips (bounded: the clip is 0.6 s, so
    // the flip lands a few blocks past the 52-block mark — the
    // engine only notices the end on the block after the last
    // sample renders).
    int blocksToEnd = 0;
    while (harness.processor.getMidiOutStatus() != Status::Waiting
           && blocksToEnd < 60)
    {
        harness.runBlock();
        ++blocksToEnd;
    }
    CHECK (harness.processor.getMidiOutStatus() == Status::Waiting);

    // Paused: back to Waiting. (stopPreview resets the position to
    // 0 first: start() resumes from the current position, so a
    // restart after the natural end needs the reset.)
    harness.processor.stopPreview();
    CHECK (harness.processor.startPreview());
    harness.runBlock();
    CHECK (harness.processor.getMidiOutStatus() == Status::Streaming);
    harness.processor.pausePreview();
    harness.runBlock();
    CHECK (harness.processor.getMidiOutStatus() == Status::Waiting);

    // Master off: Off.
    harness.setParam ("midiOut", 0.0f);
    harness.runBlock();
    CHECK (harness.processor.getMidiOutStatus() == Status::Off);

    harness.processor.releaseResources();
}

void testCursorDecision()
{
    using PC = PitchEditorComponent;
    using GC = PC::GridCursor;
    using Tool = PC::EditTool;

    // Keyboard wins over everything.
    CHECK (PC::cursorForPosition (Tool::Move, true, true, true, 0, true)
           == GC::PointingHand);
    // Playhead beats note and curve.
    CHECK (PC::cursorForPosition (Tool::Move, false, true, true, 0, true)
           == GC::LeftRightResize);
    // Note block: Move body -> dragging hand (even over the curve).
    CHECK (PC::cursorForPosition (Tool::Move, false, false, true, 0, true)
           == GC::DraggingHand);
    // Note block: Move over an edge still drags the whole note
    // (MOVE has no edge resize — only STRETCH does).
    CHECK (PC::cursorForPosition (Tool::Move, false, false, true, -1, false)
           == GC::DraggingHand);
    CHECK (PC::cursorForPosition (Tool::Move, false, false, true, 1, false)
           == GC::DraggingHand);
    // Note block: Cut -> crosshair, any edge.
    CHECK (PC::cursorForPosition (Tool::Cut, false, false, true, -1, false)
           == GC::Crosshair);
    CHECK (PC::cursorForPosition (Tool::Cut, false, false, true, 0, true)
           == GC::Crosshair);
    // Note block: Stretch edge -> resize, body -> dragging hand.
    CHECK (PC::cursorForPosition (Tool::Stretch, false, false, true, -1, false)
           == GC::LeftRightResize);
    CHECK (PC::cursorForPosition (Tool::Stretch, false, false, true, 1, false)
           == GC::LeftRightResize);
    CHECK (PC::cursorForPosition (Tool::Stretch, false, false, true, 0, false)
           == GC::DraggingHand);
    // Curve: MOVE offers the curve-edit cursor...
    CHECK (PC::cursorForPosition (Tool::Move, false, false, false, 0, true)
           == GC::UpDownResize);
    // ...CUT and STRETCH do not (the click does nothing there).
    CHECK (PC::cursorForPosition (Tool::Cut, false, false, false, 0, true)
           == GC::Normal);
    CHECK (PC::cursorForPosition (Tool::Stretch, false, false, false, 0, true)
           == GC::Normal);
    // Nothing under the mouse: normal cursor in every tool.
    CHECK (PC::cursorForPosition (Tool::Move, false, false, false, 0, false)
           == GC::Normal);
    CHECK (PC::cursorForPosition (Tool::Cut, false, false, false, 0, false)
           == GC::Normal);
    CHECK (PC::cursorForPosition (Tool::Stretch, false, false, false, 0, false)
           == GC::Normal);
}

void testSourcePitchVsEditedMidiColours()
{
    // The curve (SOURCE PITCH) and the note blocks (EDITED MIDI)
    // must read as two different things.
    CHECK (PitchEditorComponent::sourcePitchColour
           != PitchEditorComponent::editedMidiColour);
    CHECK (PitchEditorComponent::sourcePitchColour == melotheon::Tokens::meterGreen);
    CHECK (PitchEditorComponent::editedMidiColour == melotheon::Tokens::activeOrange);
    CHECK (juce::String (PitchEditorComponent::sourcePitchLegendText) == "SOURCE PITCH");
    CHECK (juce::String (PitchEditorComponent::editedMidiLegendText) == "EDITED MIDI");

    // The MIDI panel routing guide (P2-9): headline + port steps +
    // the direct-to-Piano-Roll paths. Exact strings: a paint-site
    // swap must fail, not just an empty string.
    CHECK (juce::String (MelotheonAudioProcessorEditor::midiRoutingGuideHeadline)
           == "ROUTE MIDI PORT, THEN PRESS PLAY");
    CHECK (juce::String (MelotheonAudioProcessorEditor::midiRoutingGuidePorts1)
           == "FL: Melotheon MIDI Out port =");
    CHECK (juce::String (MelotheonAudioProcessorEditor::midiRoutingGuidePorts2)
           == "instrument MIDI In port (same no.)");
    CHECK (juce::String (MelotheonAudioProcessorEditor::midiRoutingGuideDirect1)
           == "DRAG / EXPORT MIDI: notes");
    CHECK (juce::String (MelotheonAudioProcessorEditor::midiRoutingGuideDirect2)
           == "straight to the Piano Roll.");
}

void testSendToChannelAutoEnablesMidiOut()
{
    // The listener defers to the message thread: create one for
    // this test (a console test has none by default). Registered
    // LAST in main(): once the MessageManager exists, later
    // sendToChannel changes auto-enable the master.
    juce::MessageManager::getInstance();

    MelotheonAudioProcessor processor;
    processor.prepareToPlay (44100.0, 512);
    auto setParam = [&] (const char* id, float value)
    {
        if (auto* param = processor.apvts.getParameter (id); param != nullptr)
            param->setValueNotifyingHost (value);
    };

    // SEND on auto-enables the MIDI OUT master.
    CHECK (! processor.midiOutEnabled());
    setParam ("sendToChannel", 1.0f);
    CHECK (processor.midiOutEnabled());
    CHECK (processor.sendToChannelEnabled());

    // SEND off leaves MIDI OUT unchanged.
    setParam ("sendToChannel", 0.0f);
    CHECK (processor.midiOutEnabled());

    // Turning the master off manually still stops everything.
    setParam ("midiOut", 0.0f);
    CHECK (! processor.midiOutEnabled());

    // SEND on again re-enables the master.
    setParam ("sendToChannel", 1.0f);
    CHECK (processor.midiOutEnabled());

    processor.releaseResources();
}

int main()
{
    testStereoToMono();
    testStereoPhaseCancellationProtection();
    testMonoCopy();
    testAudioClipMetadata();
    testCaptureStateTransitions();
    testCaptureMaxDurationAndLimit();
    testPreviewStateTransitions();
    testPreviewResampling();
    testPreviewSeek();
    testFileLoaderRejection();
    testFileLoaderValidWav();
    testPitchAnalysisAtSourceRates();
    testLowAndHighPitchAnalysis();
    testPitchAnalysisSilenceAndCancellation();
    testPitchCurveFollowsTwoTones();
    testNoteSegmentationRejectsVibratoAndBridgesGap();
    testMidiExportRoundTrip();
    testMidiDragFilePreparation();
    testSampleInstrumentFollowsMelody();
    testProcessorPassThrough();
    testProcessorRecordAndPreview();
    testProcessorSeek();
    testCurveEditAdjustsFramesAndResegments();
    testAuditionSineFallback();
    testAuditionSamplePlayback();
    testCurveEditCommitAndDragFile();
    testMelodyEditorMoveSplitStretch();
    testMelodyEditCommitUsesEditedMelody();
    testMelodyEditPitchShiftMovesFrames();
    testLiveMidiOutScheduling();
    testLiveMidiOutResyncProcessesWholeBlock();
    testLiveMidiOutSeekResyncClearsOldNote();
    testLiveMidiOutSnapshotPublicationStress();
    testLiveMidiOutPlaybackEndsMidBlock();
    testLiveMidiOutSchedulingAcrossRatesAndBlocks();
    testLiveMidiOutMelodyChangeAndToggleHygiene();
    testLiveMidiOutEventBalanceUnderConcurrentEdits();
    testMidiOutCapacityBound();
    testFindFreeMidiOutSlot();
    testLiveMidiOutDenseBlockNoEventLost();
    testLiveMidiOutPublishNeverBlocksUnderLoad();
    testTempoGridHostBpm();
    testToolStripPlacement();
    testProjectStateFlacRoundTrip();
    testProjectStateRoundTrip();
    testProjectStateEncodeGuards();
    testProjectStateOldCorruptAndSkipped();
    testUndoRedoMelodyEdits();
    testUndoRedoCurveEdit();
    testUndoRedoClearedByNewEdit();
    testUndoRedoLimit50();
    testUndoAfterReanalyze();
    testUndoHistoryClearedOnNewClip();
    testMidiRoutingHint();
    testMidiOutStatusTransitions();
    testCursorDecision();
    testSourcePitchVsEditedMidiColours();
    // LAST: creates the MessageManager (see testMidiRoutingHint note).
    testSendToChannelAutoEnablesMidiOut();
    // After the P2-8 test: constructing a PitchEditorComponent
    // creates a MessageManager (JUCE component internals), which
    // would break testMidiRoutingHint's no-MessageManager premise.
    testGridMappingMatchesNotePositions();
    testProcessorEmptyTake();
    testProcessorTransportExclusion();

    // Clean up the temp files.
    getTestDir().deleteRecursively();

    std::cout << "Melotheon Phase 3C tests: " << (checks - failures) << "/" << checks
              << " checks passed";
    if (failures == 0)
        std::cout << " - ALL GREEN\n";
    else
        std::cout << " - " << failures << " FAILED\n";

    return failures == 0 ? 0 : 1;
}
