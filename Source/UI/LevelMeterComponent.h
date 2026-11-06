#pragma once

// LevelMeterComponent: a vertical segmented input level meter.
//
// Phase 2 feeds it a real level with setLevel(): the processor
// computes smoothed peak dB (-60..0) from the incoming host audio
// and the editor maps it to 0.0 - 1.0. Until the first real level
// arrives the meter runs a slow idle animation so it reads as alive.
//
// Segments: green -60..-12 dB, yellow -12..-3 dB, orange above -3 dB;
// dark inactive segments above the current level.

#include <JuceHeader.h>

class LevelMeterComponent : public juce::Component,
                            private juce::Timer
{
public:
    LevelMeterComponent();
    ~LevelMeterComponent() override;

    // Feed a real level (0.0 - 1.0, mapped from -60..0 dB by the
    // editor). Calling this switches the meter from idle animation
    // to the supplied level.
    void setLevel (float newLevel);

    void paint (juce::Graphics& g) override;

private:
    void timerCallback() override;

    float level = 0.0f;
    bool useExternalLevel = false;
    float animationPhase = 0.0f;
};
