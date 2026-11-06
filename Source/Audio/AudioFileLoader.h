#pragma once

// AudioFileLoader: imports audio files into an AudioClip (Phase 2).
//
// Decoding happens OFF the real-time thread: loadFileAsync() runs the
// decode on a single worker thread and delivers the result to the
// message thread, so the plugin interface never freezes while a file
// loads. loadFileSync() does the same work on the caller's thread
// (used by the worker job and by the unit tests).
//
// Accepted formats: WAV, AIFF and FLAC first. MP3 is offered only
// when the current JUCE configuration already enables it
// (JUCE_USE_MP3AUDIOFORMAT); no codec dependency is added here.
//
// Validation rejects: missing files, empty files, unknown or
// corrupted formats, files with no samples, more than two channels,
// and files longer than kMaxImportSeconds. Failures return a short
// status string (shown in the LCD) instead of any modal alert.

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <atomic>
#include <functional>
#include <memory>

#include "AudioClip.h"

class AudioFileLoader
{
public:
    static constexpr double kMaxImportSeconds = 300.0; // 5 minutes
    static constexpr int kMaxImportChannels = 2;

    struct Result
    {
        std::shared_ptr<const AudioClip> clip; // null on failure
        juce::String errorMessage;             // empty on success

        bool succeeded() const { return clip != nullptr; }
    };

    using Callback = std::function<void (Result)>;

    AudioFileLoader();
    ~AudioFileLoader();

    // Synchronous validation + decode. Safe to call from any thread
    // except the real-time audio thread (it allocates and reads disk).
    Result loadFileSync (const juce::File& file);

    // Asynchronous version: decodes on a worker thread, then invokes
    // the callback exactly once on the message thread. If the loader
    // is destroyed first, the callback is dropped. If a newer load
    // starts while one is in flight, the older result is dropped.
    // aliveFlag must stay valid while the callback is pending: the
    // owner sets it to false in its destructor so a late callback
    // never touches a destroyed owner.
    void loadFileAsync (const juce::File& file, Callback callback,
                        std::shared_ptr<std::atomic<bool>> aliveFlag);

    // Invalidate the result of any load already in flight. Decoding may
    // finish on the worker, but its message-thread callback is dropped.
    // This is used when Record or Play takes ownership of the transport.
    void cancelPendingLoad();

    // File filter string for juce::FileChooser, e.g.
    // "*.wav;*.aif;*.aiff;*.flac" (+ "*.mp3" when supported).
    static juce::String getFileFilter();

private:
    class LoadJob;

    juce::AudioFormatManager formatManager;
    juce::ThreadPool threadPool { 1 };
    std::atomic<int> loadGeneration { 0 };
};
