// LevelMeterComponent implementation. Segmented vertical meter with a
// slow idle animation (visual placeholder only - see the header).

#include "LevelMeterComponent.h"
#include "MelotheonLookAndFeel.h"
#include <cmath>

LevelMeterComponent::LevelMeterComponent()
{
    startTimer (50);
}

LevelMeterComponent::~LevelMeterComponent()
{
    stopTimer();
}

void LevelMeterComponent::setLevel (float newLevel)
{
    useExternalLevel = true;
    level = juce::jlimit (0.0f, 1.0f, newLevel);
    repaint();
}

void LevelMeterComponent::timerCallback()
{
    if (useExternalLevel)
        return;

    // Slow idle swell: a gentle sine drift so the meter reads as alive
    // without pretending to track real input.
    animationPhase += 0.11f;
    level = 0.42f + 0.22f * std::sin (animationPhase)
          + 0.08f * std::sin (animationPhase * 2.7f);
    level = juce::jlimit (0.05f, 0.85f, level);
    repaint();
}

void LevelMeterComponent::paint (juce::Graphics& g)
{
    using namespace melotheon;
    auto area = getLocalBounds().toFloat();

    g.setColour (Tokens::lcdBackground);
    g.fillRect (area);
    g.setColour (Tokens::border);
    g.drawRect (area, 1.0f);

    const int numSegments = 24;
    const float gap = 1.0f;
    const float innerX = area.getX() + 3.0f;
    const float innerW = area.getWidth() - 6.0f;
    const float innerY = area.getY() + 3.0f;
    const float innerH = area.getHeight() - 6.0f;
    const float segmentH = (innerH - gap * static_cast<float> (numSegments - 1))
                         / static_cast<float> (numSegments);
    const int litSegments = static_cast<int> (std::round (level * numSegments));

    for (int i = 0; i < numSegments; ++i)
    {
        // Segment 0 is at the bottom.
        const float y = innerY + innerH - static_cast<float> (i + 1) * segmentH
                      - static_cast<float> (i) * gap;
        juce::Rectangle<float> seg (innerX, y, innerW, segmentH);
        if (i < litSegments)
        {
            // Top ~25% of the meter is yellow, the rest is green.
            const float fraction = static_cast<float> (i + 1) / static_cast<float> (numSegments);
            g.setColour (fraction > 0.75f ? Tokens::meterYellow : Tokens::meterGreen);
        }
        else
        {
            g.setColour (Tokens::raisedPanel);
        }
        g.fillRect (seg);
    }
}
