// IconButton implementation. See the header for the kind list.
// The button body is a dark raised rectangle; the selected (toggle-on)
// state gets an orange fill (transport buttons) or an orange outline.

#include "IconButton.h"
#include "MelotheonLookAndFeel.h"

IconButton::IconButton (Kind kindToUse)
    : juce::Button ("IconButton"), kind (kindToUse)
{
    setMouseCursor (juce::MouseCursor::PointingHandCursor);
}

void IconButton::paintButton (juce::Graphics& g, bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown)
{
    using namespace melotheon;
    auto area = getLocalBounds().toFloat();
    const bool enabled = isEnabled();
    const bool selected = getToggleState();

    juce::Colour fill = Tokens::raisedPanel;
    if (! enabled)
        fill = Tokens::raisedPanel.withAlpha (0.55f);
    else if (selected)
        fill = Tokens::activeOrange;
    else if (shouldDrawButtonAsDown)
        fill = Tokens::raisedPanel.brighter (0.15f);
    else if (shouldDrawButtonAsHighlighted)
        fill = Tokens::raisedPanel.brighter (0.08f);

    g.setColour (fill);
    g.fillRoundedRectangle (area, Tokens::cornerRadius);
    g.setColour (selected ? Tokens::brightOrange : Tokens::border);
    g.drawRoundedRectangle (area, Tokens::cornerRadius, 1.0f);

    const juce::Colour iconColour = selected ? juce::Colours::black
        : (enabled ? Tokens::primaryText : Tokens::secondaryText);

    if (kind == Kind::PlayText || kind == Kind::DragFile)
    {
        // Icon on the left, text after it.
        const float iconBox = area.getHeight();
        paintIcon (g, juce::Rectangle<float> (area.getX() + 6.0f, area.getY(),
                                              iconBox - 12.0f, iconBox), iconColour);
        g.setFont (melotheonFont (Tokens::buttonSize, true));
        g.setColour (iconColour);
        g.drawText (labelText,
                    juce::Rectangle<float> (area.getX() + iconBox, area.getY(),
                                            area.getWidth() - iconBox - 6.0f, area.getHeight()),
                    juce::Justification::centredLeft, true);
    }
    else
    {
        paintIcon (g, area.reduced (area.getHeight() * 0.22f), iconColour);
    }
}

void IconButton::paintIcon (juce::Graphics& g, juce::Rectangle<float> area,
                            const juce::Colour& colour) const
{
    const float cx = area.getCentreX();
    const float cy = area.getCentreY();

    switch (kind)
    {
        case Kind::Record:
        {
            // Solid circle: bright orange when selected (the button body
            // is orange too, so the circle reads lighter), active orange
            // when idle on the dark body.
            g.setColour (getToggleState() ? melotheon::Tokens::brightOrange
                                          : melotheon::Tokens::activeOrange);
            g.fillEllipse (area.reduced (area.getWidth() * 0.18f));
            break;
        }
        case Kind::Play:
        case Kind::PlayText:
        {
            g.setColour (colour);
            juce::Path triangle;
            triangle.startNewSubPath (area.getX() + area.getWidth() * 0.28f,
                                      area.getY() + area.getHeight() * 0.2f);
            triangle.lineTo (area.getX() + area.getWidth() * 0.28f,
                             area.getY() + area.getHeight() * 0.8f);
            triangle.lineTo (area.getX() + area.getWidth() * 0.78f, cy);
            triangle.closeSubPath();
            g.fillPath (triangle);
            break;
        }
        case Kind::Stop:
        {
            g.setColour (colour);
            g.fillRect (area.reduced (area.getWidth() * 0.24f));
            break;
        }
        case Kind::Folder:
        {
            // Simple folder outline: tab + body.
            const float x = area.getX() + area.getWidth() * 0.12f;
            const float y = area.getY() + area.getHeight() * 0.2f;
            const float w = area.getWidth() * 0.76f;
            const float h = area.getHeight() * 0.6f;
            juce::Path folder;
            folder.startNewSubPath (x, y + h * 0.3f);
            folder.lineTo (x, y);
            folder.lineTo (x + w * 0.4f, y);
            folder.lineTo (x + w * 0.5f, y + h * 0.3f);
            folder.lineTo (x + w, y + h * 0.3f);
            folder.lineTo (x + w, y + h);
            folder.lineTo (x, y + h);
            folder.closeSubPath();
            g.setColour (melotheon::Tokens::brightOrange);
            g.fillPath (folder);
            break;
        }
        case Kind::DragFile:
        {
            // Four-direction move mark: small triangles up/down/left/right.
            g.setColour (colour);
            const float s = area.getWidth() * 0.14f;
            juce::Path up;
            up.startNewSubPath (cx, cy - s * 2.4f);
            up.lineTo (cx - s * 0.6f, cy - s * 1.4f);
            up.lineTo (cx + s * 0.6f, cy - s * 1.4f);
            up.closeSubPath();
            g.fillPath (up);
            juce::Path down;
            down.startNewSubPath (cx, cy + s * 2.4f);
            down.lineTo (cx - s * 0.6f, cy + s * 1.4f);
            down.lineTo (cx + s * 0.6f, cy + s * 1.4f);
            down.closeSubPath();
            g.fillPath (down);
            juce::Path left;
            left.startNewSubPath (cx - s * 2.4f, cy);
            left.lineTo (cx - s * 1.4f, cy - s * 0.6f);
            left.lineTo (cx - s * 1.4f, cy + s * 0.6f);
            left.closeSubPath();
            g.fillPath (left);
            juce::Path right;
            right.startNewSubPath (cx + s * 2.4f, cy);
            right.lineTo (cx + s * 1.4f, cy - s * 0.6f);
            right.lineTo (cx + s * 1.4f, cy + s * 0.6f);
            right.closeSubPath();
            g.fillPath (right);
            break;
        }
    }
}
