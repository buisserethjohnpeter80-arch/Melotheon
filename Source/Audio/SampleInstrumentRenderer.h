#pragma once

#include <JuceHeader.h>

#include <functional>
#include <memory>

#include "AudioClip.h"
#include "../Analysis/MelodyNotes.h"

struct SampleRenderResult
{
    std::shared_ptr<const AudioClip> renderedClip;
    int rootMidi = 60;
    float rootFrequencyHz = 0.0f;
    juce::String errorMessage;

    bool succeeded() const { return renderedClip != nullptr; }
};

class SampleInstrumentRenderer
{
public:
    using CancelCallback = std::function<bool()>;
    using ProgressCallback = std::function<void (float)>;

    // Detects the one-shot's root note, then renders every segmented melody
    // note with resampling, a sustain loop and click-free envelopes.
    static SampleRenderResult render (
        const AudioClip& oneShot,
        const MelodyNoteSequence& melody,
        double outputSampleRate,
        CancelCallback shouldCancel = {},
        ProgressCallback reportProgress = {});
};
