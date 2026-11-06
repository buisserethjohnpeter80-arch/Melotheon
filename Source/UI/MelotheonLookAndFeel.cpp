// MelotheonLookAndFeel implementation. See the header for the design
// tokens. All drawing is vector-based; no images or icon fonts.

#include "MelotheonLookAndFeel.h"
#include <cmath>

namespace melotheon
{

juce::Font melotheonFont (float size, bool bold)
{
    juce::Font font (juce::FontOptions (size).withName ("Roboto Condensed"));
    if (bold)
        font = font.boldened();
    return font;
}

} // namespace melotheon

MelotheonLookAndFeel::MelotheonLookAndFeel()
    : juce::LookAndFeel_V4 (juce::LookAndFeel_V4::ColourScheme (
        melotheon::Tokens::primaryText, melotheon::Tokens::panel,
        melotheon::Tokens::raisedPanel, melotheon::Tokens::activeOrange,
        melotheon::Tokens::activeOrange, melotheon::Tokens::secondaryText,
        melotheon::Tokens::primaryText, melotheon::Tokens::primaryText,
        melotheon::Tokens::primaryText))
{
}

void MelotheonLookAndFeel::drawRotarySlider (juce::Graphics& g, int x, int y, int width,
                                             int height, float sliderPos,
                                             float rotaryStartAngle, float rotaryEndAngle,
                                             juce::Slider& slider)
{
    using namespace melotheon;
    const float size = static_cast<float> (juce::jmin (width, height));
    const float centreX = static_cast<float> (x) + static_cast<float> (width) * 0.5f;
    const float centreY = static_cast<float> (y) + static_cast<float> (height) * 0.5f;
    // Body radius leaves room for the tick marks inside the bounds.
    const float radius = size * 0.5f - 7.0f;

    // Tick marks around the sweep.
    const int numTicks = 11;
    g.setColour (Tokens::secondaryText.withAlpha (0.55f));
    for (int i = 0; i < numTicks; ++i)
    {
        const float t = static_cast<float> (i) / static_cast<float> (numTicks - 1);
        const float angle = rotaryStartAngle + t * (rotaryEndAngle - rotaryStartAngle);
        const float x1 = centreX + std::sin (angle) * (radius + 1.0f);
        const float y1 = centreY - std::cos (angle) * (radius + 1.0f);
        const float x2 = centreX + std::sin (angle) * (radius + 5.0f);
        const float y2 = centreY - std::cos (angle) * (radius + 5.0f);
        g.drawLine (x1, y1, x2, y2, 1.0f);
    }

    // Dark circular body with a thin border.
    g.setColour (Tokens::knobBody);
    g.fillEllipse (centreX - radius, centreY - radius, radius * 2.0f, radius * 2.0f);
    g.setColour (Tokens::border);
    g.drawEllipse (centreX - radius, centreY - radius, radius * 2.0f, radius * 2.0f, 1.0f);

    // Thin orange value arc from the start angle to the current value.
    // Path angles use 0 = top-centre, clockwise, which is exactly the
    // rotary-angle convention, so the angles pass through unchanged
    // (same as juce::LookAndFeel_V4::drawRotarySlider).
    const float valueAngle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);
    juce::Path arc;
    arc.addCentredArc (centreX, centreY, radius - 3.0f, radius - 3.0f,
                       0.0f, rotaryStartAngle, valueAngle, true);
    g.setColour (Tokens::activeOrange);
    g.strokePath (arc, juce::PathStrokeType (2.0f));

    // Orange position indicator line from the centre to the edge.
    const float ix = centreX + std::sin (valueAngle) * (radius - 5.0f);
    const float iy = centreY - std::cos (valueAngle) * (radius - 5.0f);
    g.setColour (Tokens::brightOrange);
    g.drawLine (centreX, centreY, ix, iy, 2.5f);
    g.setColour (Tokens::brightOrange);
    g.fillEllipse (centreX - 2.0f, centreY - 2.0f, 4.0f, 4.0f);
    juce::ignoreUnused (slider);
}

void MelotheonLookAndFeel::drawComboBox (juce::Graphics& g, int width, int height,
                                         bool isButtonDown, int buttonX, int buttonY,
                                         int buttonW, int buttonH, juce::ComboBox& box)
{
    using namespace melotheon;
    juce::ignoreUnused (isButtonDown);
    auto area = juce::Rectangle<float> (0.0f, 0.0f, static_cast<float> (width),
                                        static_cast<float> (height));
    g.setColour (Tokens::raisedPanel);
    g.fillRoundedRectangle (area, Tokens::cornerRadius);
    g.setColour (Tokens::border);
    g.drawRoundedRectangle (area, Tokens::cornerRadius, 1.0f);

    // Down arrow on the right, drawn as a small triangle.
    const float arrowX = static_cast<float> (buttonX) + static_cast<float> (buttonW) * 0.5f;
    const float arrowY = static_cast<float> (buttonY) + static_cast<float> (buttonH) * 0.5f;
    juce::Path arrow;
    arrow.startNewSubPath (arrowX - 5.0f, arrowY - 2.0f);
    arrow.lineTo (arrowX + 5.0f, arrowY - 2.0f);
    arrow.lineTo (arrowX, arrowY + 4.0f);
    arrow.closeSubPath();
    g.setColour (box.isEnabled() ? Tokens::primaryText : Tokens::secondaryText);
    g.fillPath (arrow);
}

juce::Font MelotheonLookAndFeel::getComboBoxFont (juce::ComboBox& box)
{
    juce::ignoreUnused (box);
    return melotheon::melotheonFont (melotheon::Tokens::buttonSize, true);
}

juce::Font MelotheonLookAndFeel::getLabelFont (juce::Label& label)
{
    juce::ignoreUnused (label);
    return melotheon::melotheonFont (melotheon::Tokens::labelSize);
}

juce::Font MelotheonLookAndFeel::getTextButtonFont (juce::TextButton& button, int buttonHeight)
{
    juce::ignoreUnused (button, buttonHeight);
    return melotheon::melotheonFont (melotheon::Tokens::buttonSize, true);
}

void MelotheonLookAndFeel::drawButtonBackground (juce::Graphics& g, juce::Button& button,
                                                const juce::Colour& backgroundColour,
                                                bool shouldDrawButtonAsHighlighted,
                                                bool shouldDrawButtonAsDown)
{
    using namespace melotheon;
    auto area = button.getLocalBounds().toFloat();
    juce::Colour fill = backgroundColour;
    if (! button.isEnabled())
        fill = Tokens::raisedPanel.withAlpha (0.6f);
    else if (shouldDrawButtonAsDown)
        fill = fill.brighter (0.12f);
    else if (shouldDrawButtonAsHighlighted)
        fill = fill.brighter (0.07f);

    g.setColour (fill);
    g.fillRoundedRectangle (area, Tokens::cornerRadius);
    g.setColour (Tokens::border);
    g.drawRoundedRectangle (area, Tokens::cornerRadius, 1.0f);

    // Selected (toggle-on) state: thin orange outline.
    if (button.getToggleState())
    {
        g.setColour (Tokens::activeOrange);
        g.drawRoundedRectangle (area.reduced (1.0f), Tokens::cornerRadius, 1.5f);
    }
}

void MelotheonLookAndFeel::drawButtonText (juce::Graphics& g, juce::TextButton& button,
                                          bool shouldDrawButtonAsHighlighted,
                                          bool shouldDrawButtonAsDown)
{
    juce::ignoreUnused (shouldDrawButtonAsHighlighted, shouldDrawButtonAsDown);
    g.setFont (getTextButtonFont (button, button.getHeight()));
    g.setColour (button.findColour (juce::TextButton::textColourOffId));
    g.drawText (button.getButtonText(), button.getLocalBounds(),
                juce::Justification::centred, true);
}

void MelotheonLookAndFeel::drawToggleButton (juce::Graphics& g, juce::ToggleButton& button,
                                            bool shouldDrawButtonAsHighlighted,
                                            bool shouldDrawButtonAsDown)
{
    using namespace melotheon;
    juce::ignoreUnused (shouldDrawButtonAsHighlighted, shouldDrawButtonAsDown);
    auto area = button.getLocalBounds().toFloat();
    const float ledSize = 14.0f;
    const float ledY = area.getCentreY() - ledSize * 0.5f;

    // Orange LED square: filled when on, dark outline when off.
    juce::Rectangle<float> led (area.getX(), ledY, ledSize, ledSize);
    if (button.getToggleState())
    {
        g.setColour (Tokens::activeOrange);
        g.fillRect (led);
        g.setColour (Tokens::brightOrange);
        g.drawRect (led, 1.0f);
    }
    else
    {
        g.setColour (Tokens::raisedPanel);
        g.fillRect (led);
        g.setColour (Tokens::border);
        g.drawRect (led, 1.0f);
    }

    g.setFont (melotheonFont (Tokens::labelSize, true));
    g.setColour (button.isEnabled() ? Tokens::primaryText : Tokens::secondaryText);
    g.drawText (button.getButtonText(),
                juce::Rectangle<float> (area.getX() + ledSize + 8.0f, area.getY(),
                                        area.getWidth() - ledSize - 8.0f, area.getHeight()),
                juce::Justification::centredLeft, true);
}
