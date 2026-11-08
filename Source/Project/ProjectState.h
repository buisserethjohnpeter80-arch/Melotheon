// ProjectState: FL project saving for Melotheon.
//
// getStateInformation used to save ONLY the APVTS parameters, so a
// project reload lost the audio, the pitch analysis, the curve edits
// and the edited notes. This module serializes the whole project:
//
//   MelotheonProjectState (version 1)
//     |- Params      : the APVTS state tree, unchanged
//     |- ProjectData : binary payload (see the .cpp for the layout)
//          - audio: the clip embedded as FLAC 24-bit, with a peak
//            scale so signals above 0 dBFS never clip; the original
//            file path is stored as metadata only and never depended
//            on, so a reload works even when the file is gone
//          - pitch analysis: every frame (the current frames, so
//            curve edits survive), hop, duration, sample rate
//          - melody notes: the full list (start/end/MIDI/velocity/
//            confidence) — this IS the edited state
//          - edit flags (curve/melody edited)
//
// Backward compatibility: a state without the wrapper (all states
// saved before this change) loads exactly as before — parameters
// only, READY / EMPTY SOURCE, no crash. An unknown wrapper version
// loads its parameters and ignores the project section. Truncated
// or corrupted input never crashes: the binary reader is fully
// bounds-checked and any failure falls back to parameters-only.
//
// Size guard: if the embedded audio would push the state past
// kMaxStateBytes (32 MB), the audio is omitted and the audio-skipped
// flag is set instead — analysis, melody and parameters are always
// saved. The editor shows the skip on the next load (see the
// processor restore path).

#pragma once

#include <JuceHeader.h>

#include <memory>

#include "../Audio/AudioClip.h"
#include "../Analysis/MelodyNotes.h"
#include "../Analysis/PitchAnalysis.h"

namespace ProjectState
{
    inline constexpr int kStateVersion = 1;
    inline constexpr std::size_t kMaxStateBytes = 32 * 1024 * 1024;

    struct Data
    {
        std::shared_ptr<const AudioClip> clip;               // nullptr = no audio
        std::shared_ptr<const PitchAnalysisResult> analysis; // nullptr = none
        std::shared_ptr<const MelodyNoteSequence> melody;    // nullptr = none
        bool pitchAnalysisEdited = false;
        bool audioSkippedTooLarge = false; // stored flag, reported on load
    };

    // Cache for the FLAC encoding of one clip, keyed by clip version:
    // hosts may snapshot state repeatedly, and re-encoding a 60 s
    // clip on every call would waste time.
    struct EncodedAudioCache
    {
        juce::MemoryBlock flac;
        double peakScale = 1.0;
        int clipVersion = -1;
    };

    // Build the wrapper ValueTree for getStateInformation. The audio
    // is encoded (or reused from audioCache when clipVersion
    // matches). maxStateBytes caps the whole blob; pass a smaller
    // value only in tests to force the audio-skip path.
    juce::ValueTree serializeState (const juce::ValueTree& params,
                                    const Data& data,
                                    EncodedAudioCache* audioCache,
                                    int clipVersion,
                                    std::size_t maxStateBytes = kMaxStateBytes);

    struct Deserialized
    {
        juce::ValueTree params;   // valid when hasParams
        bool hasParams = false;
        Data project;
        bool hasProject = false;  // true when a v1 project section parsed
    };

    // Parse a state blob. Never crashes on garbage input.
    Deserialized deserializeState (const void* data, int sizeInBytes);

    // Recompute the derived analysis statistics (voiced count, pitch
    // range, medians) from the frames, after a decode.
    void recomputeAnalysisStats (PitchAnalysisResult& analysis);

    // FLAC 24-bit encode/decode of a clip's preview buffer. The
    // buffer is scaled by 1/peak before encoding (peak = max |sample|,
    // 1.0 when the signal stays inside [-1, 1]) and outPeakScale
    // carries that peak, so the decode multiplies back and hot signals
    // never clip. Round-trip error is below 2^-24 * peakScale
    // (2^-24 when the peak is at or below full scale).
    bool encodeClipAudio (const AudioClip& clip, juce::MemoryBlock& outFlac,
                          double& outPeakScale);
    std::shared_ptr<AudioClip> decodeClipAudio (const void* flacData,
                                                std::size_t flacSize,
                                                double peakScale,
                                                const juce::String& sourceName,
                                                bool importedFromFile);
}
