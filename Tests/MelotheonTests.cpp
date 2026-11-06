// Melotheon Phase 2.1 unit tests.
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
#include "PluginProcessor.h"

#include <cmath>
#include <iostream>

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
    source.getWritePointer (1)[3] = -1.0f;

    juce::AudioBuffer<float> mono;
    AudioClip::buildMonoBuffer (source, mono);

    CHECK (mono.getNumChannels() == 1);
    CHECK (mono.getNumSamples() == 4);
    CHECK_NEAR (mono.getReadPointer (0)[0], 0.4f, 1e-6f);  // 0.5 * (0.6 + 0.2)
    CHECK_NEAR (mono.getReadPointer (0)[1], 0.1f, 1e-6f);  // 0.5 * (-0.2 + 0.4)
    CHECK_NEAR (mono.getReadPointer (0)[2], 0.0f, 1e-6f);
    CHECK_NEAR (mono.getReadPointer (0)[3], 0.0f, 1e-6f);  // 0.5 * (1.0 + -1.0)
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

} // namespace

int main()
{
    testStereoToMono();
    testMonoCopy();
    testAudioClipMetadata();
    testCaptureStateTransitions();
    testCaptureMaxDurationAndLimit();
    testPreviewStateTransitions();
    testPreviewResampling();
    testFileLoaderRejection();
    testFileLoaderValidWav();
    testProcessorPassThrough();
    testProcessorRecordAndPreview();
    testProcessorEmptyTake();
    testProcessorTransportExclusion();

    // Clean up the temp files.
    getTestDir().deleteRecursively();

    std::cout << "Melotheon Phase 2.1 tests: " << (checks - failures) << "/" << checks
              << " checks passed";
    if (failures == 0)
        std::cout << " - ALL GREEN\n";
    else
        std::cout << " - " << failures << " FAILED\n";

    return failures == 0 ? 0 : 1;
}
