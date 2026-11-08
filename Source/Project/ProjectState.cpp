// ProjectState implementation. See the header for the contract.
//
// Binary payload layout (little-endian, all reads bounds-checked):
//   'MLTN' magic, u32 version (= 1), u32 flags
//     bit 0: audio present, bit 1: analysis present,
//     bit 2: melody present, bit 3: audio skipped (too large),
//     bit 4: pitch analysis edited
//   audio (when present):
//     f64 sampleRate, u32 numChannels, u64 numSamples, f64 peakScale,
//     u64 flacBytes, <flac bytes>, u32 nameLen + UTF-8 name,
//     u8 importedFromFile
//   analysis (when present):
//     u32 frameCount, f64 hopSeconds, f64 durationSeconds,
//     f64 analysisSampleRate,
//     per frame: f32 frequencyHz, f32 midiPitch, f32 confidence,
//                f32 rmsDb, u8 voiced
//     (frame times are index * hopSeconds, recomputed on load)
//   melody (when present):
//     u32 noteCount, f64 durationSeconds,
//     per note: f64 start, f64 end, i32 midi, f32 sourceMidiPitch,
//               f32 velocity, f32 confidence
//   'MLTN' end magic; the reader also requires the consumed length
//   to equal the blob length exactly.

#include "ProjectState.h"

#include "../Analysis/MelodyEditor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ProjectState
{
namespace
{
    constexpr std::uint32_t kMagic = 0x4d4c544e; // 'MLTN'
    constexpr std::uint32_t kFlagAudio = 1u << 0;
    constexpr std::uint32_t kFlagAnalysis = 1u << 1;
    constexpr std::uint32_t kFlagMelody = 1u << 2;
    constexpr std::uint32_t kFlagAudioSkipped = 1u << 3;
    constexpr std::uint32_t kFlagEdited = 1u << 4;

    // Hard bounds against corrupt blobs: imports allow up to 300 s
    // (AudioFileLoader kMaxImportSeconds), so no legitimate clip is
    // longer than 300 s at any sane rate, and no analysis has more
    // than 300 s / 5 ms of frames (the analyzer clamps its hop to
    // >= 5 ms).
    constexpr std::int64_t kMaxClipSamples = 300 * 192000;
    constexpr std::uint32_t kMaxFrames = 60000;
    constexpr std::uint32_t kMaxNotes = 100000;

    struct PayloadWriter
    {
        juce::MemoryOutputStream stream;

        void u8 (std::uint8_t value) { stream.writeByte (static_cast<int> (value)); }
        void u32 (std::uint32_t value) { stream.writeInt (static_cast<int> (value)); }
        void i32 (std::int32_t value) { stream.writeInt (static_cast<int> (value)); }
        void u64 (std::uint64_t value) { stream.writeInt64 (static_cast<std::int64_t> (value)); }
        void f32 (float value) { stream.writeFloat (value); }
        void f64 (double value) { stream.writeDouble (value); }
        void bytes (const void* data, std::size_t size)
        {
            stream.write (data, static_cast<int> (size));
        }
        void str (const juce::String& value)
        {
            const auto utf8 = value.toRawUTF8();
            const std::size_t length = std::strlen (utf8);
            u32 (static_cast<std::uint32_t> (length));
            bytes (utf8, length);
        }
    };

    struct PayloadReader
    {
        const std::uint8_t* data = nullptr;
        std::size_t size = 0;
        std::size_t pos = 0;
        bool failed = false;

        bool u8 (std::uint8_t& value)
        {
            if (pos + 1 > size) { failed = true; return false; }
            value = data[pos++];
            return true;
        }
        bool u32 (std::uint32_t& value)
        {
            if (pos + 4 > size) { failed = true; return false; }
            std::uint32_t result = 0;
            for (int i = 0; i < 4; ++i)
                result |= static_cast<std::uint32_t> (data[pos + static_cast<std::size_t> (i)])
                        << (8 * i);
            pos += 4;
            value = result;
            return true;
        }
        bool i32 (std::int32_t& value)
        {
            std::uint32_t raw = 0;
            if (! u32 (raw))
                return false;
            value = static_cast<std::int32_t> (raw);
            return true;
        }
        bool u64 (std::uint64_t& value)
        {
            if (pos + 8 > size) { failed = true; return false; }
            std::uint64_t result = 0;
            for (int i = 0; i < 8; ++i)
                result |= static_cast<std::uint64_t> (data[pos + static_cast<std::size_t> (i)])
                        << (8 * i);
            pos += 8;
            value = result;
            return true;
        }
        bool f32 (float& value)
        {
            std::uint32_t raw = 0;
            if (! u32 (raw))
                return false;
            std::memcpy (&value, &raw, sizeof (value));
            return true;
        }
        bool f64 (double& value)
        {
            std::uint64_t raw = 0;
            if (! u64 (raw))
                return false;
            std::memcpy (&value, &raw, sizeof (value));
            return true;
        }
        bool bytes (void* destination, std::size_t count)
        {
            if (pos + count > size) { failed = true; return false; }
            std::memcpy (destination, data + pos, count);
            pos += count;
            return true;
        }
        bool str (juce::String& value)
        {
            std::uint32_t length = 0;
            if (! u32 (length))
                return false;
            if (length > 4096 || pos + length > size) { failed = true; return false; }
            value = juce::String::fromUTF8 (
                reinterpret_cast<const char*> (data + pos),
                static_cast<int> (length));
            pos += length;
            return true;
        }
        bool atEnd() const { return pos == size; }
    };

    float medianOf (std::vector<float> values)
    {
        if (values.empty())
            return 0.0f;
        std::sort (values.begin(), values.end());
        const std::size_t mid = values.size() / 2;
        if (values.size() % 2 == 1)
            return values[mid];
        return 0.5f * (values[mid - 1] + values[mid]);
    }
}

void recomputeAnalysisStats (PitchAnalysisResult& analysis)
{
    int voiced = 0;
    float low = 0.0f;
    float high = 0.0f;
    std::vector<float> confidences;
    std::vector<float> frequencies;
    for (const auto& frame : analysis.frames)
    {
        if (! frame.voiced)
            continue;
        ++voiced;
        if (voiced == 1 || frame.midiPitch < low)
            low = frame.midiPitch;
        if (voiced == 1 || frame.midiPitch > high)
            high = frame.midiPitch;
        confidences.push_back (frame.confidence);
        frequencies.push_back (frame.frequencyHz);
    }
    analysis.voicedFrameCount = voiced;
    analysis.minMidi = low;
    analysis.maxMidi = high;
    analysis.medianConfidence = medianOf (std::move (confidences));
    analysis.medianFrequencyHz = medianOf (std::move (frequencies));
}

bool encodeClipAudio (const AudioClip& clip, juce::MemoryBlock& outFlac,
                      double& outPeakScale)
{
    outFlac.reset();
    outPeakScale = 1.0;
    if (clip.isEmpty() || clip.previewBuffer.getNumChannels() <= 0
        || clip.sourceSampleRate <= 0.0)
        return false;

    const int numChannels = clip.previewBuffer.getNumChannels();
    const int numSamples = clip.previewBuffer.getNumSamples();
    // The loader caps clips at 2 channels and the FLAC reader
    // accepts 8000-192000 Hz; encode and decode must agree, so the
    // same bounds are enforced here instead of failing silently
    // inside the writer.
    if (numChannels > 2
        || clip.sourceSampleRate < 8000.0 || clip.sourceSampleRate > 192000.0)
        return false;

    // Peak over all channels: signals above 0 dBFS must not clip in
    // the 24-bit integer encoding, so the buffer is scaled by
    // 1/peak and the peak is stored for the decode. A non-finite
    // sample (NaN/inf) is invisible to the peak scan (std::max keeps
    // the old peak) and would reach the integer conversion as
    // garbage, so it fails the encode outright.
    double peak = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float* data = clip.previewBuffer.getReadPointer (ch);
        for (int i = 0; i < numSamples; ++i)
        {
            if (! std::isfinite (data[i]))
                return false;
            peak = std::max (peak, static_cast<double> (std::abs (data[i])));
        }
    }
    if (! std::isfinite (peak))
        return false;
    outPeakScale = peak > 1.0 ? peak : 1.0;

    juce::AudioBuffer<float> scaled (numChannels, numSamples);
    if (outPeakScale > 1.0)
    {
        const float gain = static_cast<float> (1.0 / outPeakScale);
        for (int ch = 0; ch < numChannels; ++ch)
            juce::FloatVectorOperations::multiply (
                scaled.getWritePointer (ch),
                clip.previewBuffer.getReadPointer (ch), gain, numSamples);
    }
    else
    {
        for (int ch = 0; ch < numChannels; ++ch)
            scaled.copyFrom (ch, 0, clip.previewBuffer, ch, 0, numSamples);
    }

    // The FLAC writer owns the stream it is given (and deletes it);
    // on a nullptr return it does NOT, so delete it here.
    juce::MemoryOutputStream* stream = new juce::MemoryOutputStream (outFlac, false);
    juce::FlacAudioFormat format;
    std::unique_ptr<juce::AudioFormatWriter> writer (format.createWriterFor (
        stream, clip.sourceSampleRate,
        static_cast<unsigned int> (numChannels), 24, {}, 5));
    if (writer == nullptr)
    {
        delete stream;
        outFlac.reset();
        return false;
    }
    if (! writer->writeFromAudioSampleBuffer (scaled, 0, numSamples))
    {
        // Destroy the writer first: it owns (and deletes) the stream,
        // whose destructor flushes into outFlac — reset the block
        // only after the stream is gone.
        writer.reset();
        outFlac.reset();
        return false;
    }
    // Destroy the writer first: its destructor runs
    // FLAC__stream_encoder_finish() and writes the final partial
    // block, and the stream it owns flushes into outFlac only when
    // it is destroyed. Checking the size before that (or calling
    // flush(), which the FLAC writer does not implement) reads 0
    // for clips shorter than one FLAC block (~4096 samples).
    writer.reset();
    return outFlac.getSize() > 0;
}

std::shared_ptr<AudioClip> decodeClipAudio (const void* flacData,
                                            std::size_t flacSize,
                                            double peakScale,
                                            const juce::String& sourceName,
                                            bool importedFromFile)
{
    if (flacData == nullptr || flacSize == 0
        || ! std::isfinite (peakScale) || peakScale < 1.0 || peakScale > 1000.0)
        return nullptr;

    // The reader takes ownership of the stream on success (AudioFormatReader's
    // destructor deletes its input), so the stream must be heap-allocated and
    // handed over. deleteStreamIfOpeningFails=true covers the failure path:
    // on failure the reader is destroyed and deletes the stream too.
    auto* stream = new juce::MemoryInputStream (flacData, flacSize, false);
    juce::FlacAudioFormat format;
    std::unique_ptr<juce::AudioFormatReader> reader (
        format.createReaderFor (stream, true));
    if (reader == nullptr || reader->sampleRate <= 0.0
        || reader->numChannels < 1 || reader->numChannels > 2
        || reader->lengthInSamples <= 0
        || reader->lengthInSamples > kMaxClipSamples)
        return nullptr;

    const int numSamples = static_cast<int> (reader->lengthInSamples);
    juce::AudioBuffer<float> buffer (
        static_cast<int> (reader->numChannels), numSamples);
    if (! reader->read (&buffer, 0, numSamples, 0, true, true))
        return nullptr;

    if (peakScale > 1.0)
    {
        const float gain = static_cast<float> (peakScale);
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            juce::FloatVectorOperations::multiply (
                buffer.getWritePointer (ch), gain, numSamples);
    }

    return AudioClip::fromBuffer (buffer, reader->sampleRate,
                                  sourceName, importedFromFile);
}

juce::ValueTree serializeState (const juce::ValueTree& params,
                                const Data& data,
                                EncodedAudioCache* audioCache,
                                int clipVersion,
                                std::size_t maxStateBytes)
{
    // The audio is encoded once per clip version (the host may
    // snapshot state repeatedly); the rest of the payload is cheap.
    juce::MemoryBlock flac;
    double peakScale = 1.0;
    bool haveAudio = false;
    if (data.clip != nullptr && ! data.clip->isEmpty())
    {
        if (audioCache != nullptr && audioCache->clipVersion == clipVersion
            && audioCache->flac.getSize() > 0)
        {
            flac = audioCache->flac;
            peakScale = audioCache->peakScale;
            haveAudio = true;
        }
        else if (encodeClipAudio (*data.clip, flac, peakScale))
        {
            haveAudio = true;
            if (audioCache != nullptr)
            {
                audioCache->flac = flac;
                audioCache->peakScale = peakScale;
                audioCache->clipVersion = clipVersion;
            }
        }
    }

    const bool haveAnalysis = data.analysis != nullptr
                            && data.analysis->hasPitch();
    const bool haveMelody = data.melody != nullptr && ! data.melody->empty();

    auto buildWrapper = [&] (bool includeAudio)
        -> std::pair<juce::ValueTree, std::size_t>
    {
        PayloadWriter payload;
        std::uint32_t flags = 0;
        if (includeAudio && haveAudio)
            flags |= kFlagAudio;
        if (! includeAudio && haveAudio)
            flags |= kFlagAudioSkipped;
        if (haveAnalysis)
            flags |= kFlagAnalysis;
        if (haveMelody)
            flags |= kFlagMelody;
        if (data.pitchAnalysisEdited)
            flags |= kFlagEdited;

        payload.u32 (kMagic);
        payload.u32 (kStateVersion);
        payload.u32 (flags);

        if (includeAudio && haveAudio)
        {
            payload.f64 (data.clip->sourceSampleRate);
            payload.u32 (static_cast<std::uint32_t> (
                data.clip->previewBuffer.getNumChannels()));
            payload.u64 (static_cast<std::uint64_t> (data.clip->totalSamples));
            payload.f64 (peakScale);
            payload.u64 (static_cast<std::uint64_t> (flac.getSize()));
            payload.bytes (flac.getData(), flac.getSize());
            payload.str (data.clip->sourceName);
            payload.u8 (data.clip->importedFromFile ? 1 : 0);
        }

        if (haveAnalysis)
        {
            const auto& analysis = *data.analysis;
            payload.u32 (static_cast<std::uint32_t> (analysis.frames.size()));
            payload.f64 (analysis.hopSeconds);
            payload.f64 (analysis.durationSeconds);
            payload.f64 (analysis.analysisSampleRate);
            for (const auto& frame : analysis.frames)
            {
                payload.f32 (frame.frequencyHz);
                payload.f32 (frame.midiPitch);
                payload.f32 (frame.confidence);
                payload.f32 (frame.rmsDb);
                payload.u8 (frame.voiced ? 1 : 0);
            }
        }

        if (haveMelody)
        {
            const auto& melody = *data.melody;
            payload.u32 (static_cast<std::uint32_t> (melody.notes.size()));
            payload.f64 (melody.durationSeconds);
            for (const auto& note : melody.notes)
            {
                payload.f64 (note.startSeconds);
                payload.f64 (note.endSeconds);
                payload.i32 (note.midiNote);
                payload.f32 (note.sourceMidiPitch);
                payload.f32 (note.velocity);
                payload.f32 (note.confidence);
            }
        }

        payload.u32 (kMagic);

        juce::MemoryBlock blob;
        {
            juce::MemoryOutputStream blobStream (blob, false);
            blobStream.write (payload.stream.getData(),
                              static_cast<int> (payload.stream.getDataSize()));
        }

        juce::ValueTree wrapper ("MelotheonProjectState");
        wrapper.setProperty ("version", kStateVersion, nullptr);
        juce::ValueTree paramsNode ("Params");
        paramsNode.addChild (params.createCopy(), -1, nullptr);
        wrapper.addChild (paramsNode, -1, nullptr);
        juce::ValueTree projectNode ("ProjectData");
        projectNode.setProperty ("blob", juce::var (blob), nullptr);
        wrapper.addChild (projectNode, -1, nullptr);

        juce::MemoryOutputStream measure;
        wrapper.writeToStream (measure);
        return { wrapper, measure.getDataSize() };
    };

    auto built = buildWrapper (true);

    // Size guard: the whole wrapper must stay under the cap. The
    // params tree is small; the audio dominates. When it does not
    // fit, drop the audio and keep everything else — never silently
    // drop the analysis or the melody.
    if (built.second > maxStateBytes && haveAudio)
        built = buildWrapper (false);

    return built.first;
}

Deserialized deserializeState (const void* data, int sizeInBytes)
{
    Deserialized result;
    if (data == nullptr || sizeInBytes <= 0)
        return result;

    auto state = juce::ValueTree::readFromData (data, static_cast<size_t> (sizeInBytes));
    if (! state.isValid())
        return result;

    // Old format (everything saved before this change): the whole
    // blob IS the APVTS state (type "MelotheonState"). Load
    // parameters only, exactly as before — the project section
    // simply does not exist.
    if (state.getType() != juce::Identifier ("MelotheonProjectState"))
    {
        result.params = state;
        result.hasParams = true;
        return result;
    }

    const int version = static_cast<int> (state.getProperty ("version", 0));
    const auto paramsNode = state.getChildWithName ("Params");
    if (paramsNode.isValid() && paramsNode.getNumChildren() > 0)
    {
        result.params = paramsNode.getChild (0);
        result.hasParams = true;
    }
    // Unknown wrapper version: parameters are all we trust.
    if (version != kStateVersion)
        return result;

    const auto projectNode = state.getChildWithName ("ProjectData");
    if (! projectNode.isValid())
        return result;
    const juce::var blobVar = projectNode.getProperty ("blob");
    const auto* blob = blobVar.getBinaryData();
    if (blob == nullptr || blob->getSize() == 0)
        return result;

    PayloadReader reader;
    reader.data = static_cast<const std::uint8_t*> (blob->getData());
    reader.size = blob->getSize();

    std::uint32_t magic = 0;
    std::uint32_t payloadVersion = 0;
    std::uint32_t flags = 0;
    if (! reader.u32 (magic) || magic != kMagic
        || ! reader.u32 (payloadVersion) || payloadVersion != kStateVersion
        || ! reader.u32 (flags))
        return result;

    Data project;
    project.audioSkippedTooLarge = (flags & kFlagAudioSkipped) != 0;
    project.pitchAnalysisEdited = (flags & kFlagEdited) != 0;

    if ((flags & kFlagAudio) != 0)
    {
        double sampleRate = 0.0;
        std::uint32_t numChannels = 0;
        std::uint64_t numSamples = 0;
        double peakScale = 1.0;
        std::uint64_t flacBytes = 0;
        juce::String sourceName;
        std::uint8_t imported = 0;
        if (! reader.f64 (sampleRate) || ! reader.u32 (numChannels)
            || ! reader.u64 (numSamples) || ! reader.f64 (peakScale)
            || ! reader.u64 (flacBytes)
            || ! std::isfinite (sampleRate) || sampleRate < 8000.0 || sampleRate > 192000.0
            || ! std::isfinite (peakScale)
            || numChannels < 1 || numChannels > 2
            || numSamples == 0 || numSamples > static_cast<std::uint64_t> (kMaxClipSamples)
            || flacBytes == 0 || flacBytes > reader.size - reader.pos)
            return result;
        const auto* flacData = reader.data + reader.pos;
        reader.pos += static_cast<std::size_t> (flacBytes);
        if (! reader.str (sourceName) || ! reader.u8 (imported))
            return result;
        project.clip = decodeClipAudio (flacData, static_cast<std::size_t> (flacBytes),
                                        peakScale, sourceName, imported != 0);
        if (project.clip == nullptr)
            return result;
    }

    if ((flags & kFlagAnalysis) != 0)
    {
        std::uint32_t frameCount = 0;
        double hopSeconds = 0.0;
        double durationSeconds = 0.0;
        double analysisSampleRate = 0.0;
        if (! reader.u32 (frameCount) || frameCount > kMaxFrames
            || ! reader.f64 (hopSeconds) || ! std::isfinite (hopSeconds)
            || hopSeconds <= 0.0 || hopSeconds > 1.0
            || ! reader.f64 (durationSeconds) || ! std::isfinite (durationSeconds)
            || durationSeconds < 0.0 || durationSeconds > 3600.0
            || ! reader.f64 (analysisSampleRate)
            || ! std::isfinite (analysisSampleRate) || analysisSampleRate <= 0.0)
            return result;
        auto analysis = std::make_shared<PitchAnalysisResult>();
        analysis->hopSeconds = hopSeconds;
        analysis->durationSeconds = durationSeconds;
        analysis->analysisSampleRate = analysisSampleRate;
        analysis->frames.reserve (frameCount);
        for (std::uint32_t i = 0; i < frameCount; ++i)
        {
            PitchFrame frame;
            std::uint8_t voiced = 0;
            if (! reader.f32 (frame.frequencyHz) || ! reader.f32 (frame.midiPitch)
                || ! reader.f32 (frame.confidence) || ! reader.f32 (frame.rmsDb)
                || ! reader.u8 (voiced)
                || ! std::isfinite (frame.frequencyHz)
                || ! std::isfinite (frame.midiPitch)
                || ! std::isfinite (frame.confidence)
                || ! std::isfinite (frame.rmsDb))
                return result;
            frame.timeSeconds = static_cast<double> (i) * hopSeconds;
            frame.voiced = voiced != 0;
            analysis->frames.push_back (frame);
        }
        recomputeAnalysisStats (*analysis);
        project.analysis = std::move (analysis);
    }

    if ((flags & kFlagMelody) != 0)
    {
        std::uint32_t noteCount = 0;
        double durationSeconds = 0.0;
        if (! reader.u32 (noteCount) || noteCount > kMaxNotes
            || ! reader.f64 (durationSeconds) || ! std::isfinite (durationSeconds)
            || durationSeconds < 0.0 || durationSeconds > 3600.0)
            return result;
        auto melody = std::make_shared<MelodyNoteSequence>();
        melody->durationSeconds = durationSeconds;
        melody->notes.reserve (noteCount);
        // The scheduler relies on notes being sorted by start time;
        // overlapping notes are tolerated (the scheduler's note-off
        // hygiene handles them) — only a note starting before the
        // previous note's start breaks the ordering invariant, and
        // only that fails the whole section.
        double lastStart = 0.0;
        for (std::uint32_t i = 0; i < noteCount; ++i)
        {
            MelodyNote note;
            std::int32_t midi = 60;
            if (! reader.f64 (note.startSeconds) || ! reader.f64 (note.endSeconds)
                || ! reader.i32 (midi) || ! reader.f32 (note.sourceMidiPitch)
                || ! reader.f32 (note.velocity) || ! reader.f32 (note.confidence)
                || ! std::isfinite (note.startSeconds)
                || ! std::isfinite (note.endSeconds)
                || ! std::isfinite (note.sourceMidiPitch)
                || ! std::isfinite (note.velocity)
                || ! std::isfinite (note.confidence)
                || midi < 0 || midi > 127
                || note.startSeconds < 0.0 || note.endSeconds < note.startSeconds
                || note.startSeconds < lastStart - 1e-9)
                return result;
            note.midiNote = midi;
            lastStart = note.startSeconds;
            melody->notes.push_back (note);
        }
        MelodyEditor::refreshSequenceMetadata (*melody);
        project.melody = std::move (melody);
    }

    std::uint32_t endMagic = 0;
    if (! reader.u32 (endMagic) || endMagic != kMagic || ! reader.atEnd())
        return result;

    result.project = std::move (project);
    result.hasProject = true;
    return result;
}
}
