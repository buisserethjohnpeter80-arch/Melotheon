#pragma once

// LevelMeterComponent: a vertical segmented input level meter.
//
// Phase 1 is a VISUAL PLACEHOLDER: the level is a slow idle animation
// driven by this component's own timer. It is NOT a real audio level -
// no audio data reaches this component. A future phase can feed it a
// real peak with setLevel().
//
// Segments: green at low and medium levels, yellow near the top, dark
// inactive segments above the current level.

#include <JuceHeader.h>

class LevelMeterComponent : public juce::Component,
                            private juce::Timer
{
public:
    LevelMeterComponent();
    ~LevelMeterComponent() override;

    // Feed a real level (0.0 - 1.0) from a future phase. Calling this
    // switches the meter from idle animation to the supplied level.
    void setLevel (float newLevel);

    void paint (juce::Graphics& g) override;

private:
    void timerCallback() override;

    float level = 0.0f;
    bool useExternalLevel = false;
    float animationPhase = 0.0f;
};
