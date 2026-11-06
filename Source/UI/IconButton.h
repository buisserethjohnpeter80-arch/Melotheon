#pragma once

// IconButton: a vector-drawn button with normal / hover / pressed /
// selected / disabled states. Icons are drawn with juce::Path or simple
// geometry - no images, no icon fonts, no emoji, no unicode symbols.
//
// It subclasses juce::Button so it has real button behaviour (toggle
// state, clicking, keyboard focus) and can be attached to an
// AudioProcessorValueTreeState bool parameter.
//
// Kinds: Record (orange circle), Play (triangle), Stop (square),
// Folder (folder outline), DragFile (four-direction move mark + text),
// PlayText (triangle + text label, used by PLAY ORIGINAL / PLAY
// INSTRUMENT / DRAG MIDI TO FL).

#include <JuceHeader.h>

class IconButton : public juce::Button
{
public:
    enum class Kind
    {
        Record,
        Play,
        Stop,
        Folder,
        DragFile,
        PlayText
    };

    IconButton() = default;
    explicit IconButton (Kind kindToUse);

    void setKind (Kind kindToUse) { kind = kindToUse; repaint(); }
    void setText (const juce::String& textToUse) { labelText = textToUse; repaint(); }

    void paintButton (juce::Graphics& g, bool shouldDrawButtonAsHighlighted,
                      bool shouldDrawButtonAsDown) override;

private:
    void paintIcon (juce::Graphics& g, juce::Rectangle<float> area,
                    const juce::Colour& colour) const;

    Kind kind = Kind::Play;
    juce::String labelText;
};
