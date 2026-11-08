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
#include "Midi/MidiExporter.h"
#include "PluginProcessor.h"

#include <atomic>
#include <cmath>
#include <iostream>
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

    // Beat snap: 16 beats over 1.6 s -> 0.1 s per beat.
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToDisplayBeat (0.24, 1.6), 0.2f, 1e-6f);
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToDisplayBeat (0.26, 1.6), 0.3f, 1e-6f);
    // Snap-if-close: 0.24 s is 0.4 beats from 0.2 s -> no snap;
    // 0.21 s is 0.1 beats away -> snaps to the beat.
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToDisplayBeatIfClose (0.24, 1.6), 0.24f, 1e-6f);
    CHECK_NEAR ((float) MelodyEditor::snapSecondsToDisplayBeatIfClose (0.21, 1.6), 0.2f, 1e-6f);

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

    bool stoppedOldNote = false;
    bool startedDestinationNote = false;
    bool stoppedDestinationNote = false;
    for (const auto metadata : midi)
    {
        const auto message = metadata.getMessage();
        if (message.isNoteOff() && message.getNoteNumber() == 60
            && metadata.samplePosition == 0)
            stoppedOldNote = true;
        if (message.isNoteOn() && message.getNoteNumber() == 64
            && metadata.samplePosition == 0)
            startedDestinationNote = true;
        if (message.isNoteOff() && message.getNoteNumber() == 64
            && std::abs (metadata.samplePosition - 352) <= 1)
            stoppedDestinationNote = true;
    }
    CHECK (stoppedOldNote);
    CHECK (startedDestinationNote);
    CHECK (stoppedDestinationNote);

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

} // namespace

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
