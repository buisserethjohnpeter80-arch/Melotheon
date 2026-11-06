#pragma once

// Melotheon design tokens and LookAndFeel.
//
// ONE central place for every colour, font size and corner radius used
// by the interface. No colour literals are scattered through the other
// UI files: they all read from melotheon::tokens below.
//
// Colours match the reference image: dark FL Studio-style panels,
// orange active states, green status displays.

#include <JuceHeader.h>

namespace melotheon
{

// Brand text. Written with explicit UTF-8 byte escapes so it renders
// as MELOTHEON under any MSVC source codepage. Every other UI string
// in the plugin is plain ASCII.
inline const char* brandTextUtf8 = "MELOTH\xC3\x89ON";

// Base design size in pixels. PluginEditor::resized() scales every
// layout rectangle from these coordinates.
inline constexpr int designWidth = 1400;
inline constexpr int designHeight = 560;

// Layout regions in base coordinates, measured from the reference
// image (the image wins over the written spec numbers where they
// differ).
inline constexpr int headerX = 10;
inline constexpr int headerY = 10;
inline constexpr int headerW = 1380;
inline constexpr int headerH = 78;
inline constexpr int leftX = 10;
inline constexpr int leftY = 98;
inline constexpr int leftW = 245;
inline constexpr int leftH = 338;
inline constexpr int centerX = 258;
inline constexpr int centerY = 98;
inline constexpr int centerW = 810;
inline constexpr int centerH = 338;
inline constexpr int rightX = 1078;
inline constexpr int rightY = 98;
inline constexpr int rightW = 312;
inline constexpr int rightH = 338;
inline constexpr int bottomX = 10;
inline constexpr int bottomY = 446;
inline constexpr int bottomW = 1380;
inline constexpr int bottomH = 104;

struct Tokens
{
    static inline const juce::Colour background { 0xff1f252c };
    static inline const juce::Colour panel { 0xff2b333d };
    static inline const juce::Colour raisedPanel { 0xff333d47 };
    static inline const juce::Colour pitchBackground { 0xff151b21 };
    static inline const juce::Colour border { 0xff49545f };
    static inline const juce::Colour gridLine { 0xff2b343d };
    static inline const juce::Colour primaryText { 0xffe9edf1 };
    static inline const juce::Colour secondaryText { 0xffaab3bc };
    static inline const juce::Colour activeOrange { 0xffff8a1f };
    static inline const juce::Colour brightOrange { 0xffffad36 };
    static inline const juce::Colour statusGreen { 0xff9cff72 };
    static inline const juce::Colour meterGreen { 0xff56e36c };
    static inline const juce::Colour meterYellow { 0xffffdc4d };
    static inline const juce::Colour lcdBackground { 0xff14181d };
    static inline const juce::Colour knobBody { 0xff22292f };
    static inline const juce::Colour waveformGrey { 0xff6b7680 };

    // Font sizes at the 1400 x 560 base size.
    static constexpr float brandSize = 28.0f;
    static constexpr float lcdSize = 18.0f;
    static constexpr float buttonSize = 15.0f;
    static constexpr float labelSize = 13.0f;
    static constexpr float smallLabelSize = 12.0f;

    static constexpr float cornerRadius = 3.0f;
    static constexpr float panelCornerRadius = 4.0f;
};

// Compact professional sans-serif. "Roboto Condensed" when the system
// has it, otherwise JUCE falls back to the default sans-serif face.
juce::Font melotheonFont (float size, bool bold = false);

} // namespace melotheon

// MelotheonLookAndFeel: dark FL Studio-style drawing for rotary knobs,
// combo boxes, buttons and LED toggles. All drawing is vector-based
// (juce::Path / geometry); no images, no icon fonts, no emoji.
class MelotheonLookAndFeel : public juce::LookAndFeel_V4
{
public:
    MelotheonLookAndFeel();

    void drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                           float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                           juce::Slider& slider) override;

    void drawComboBox (juce::Graphics& g, int width, int height, bool isButtonDown,
                       int buttonX, int buttonY, int buttonW, int buttonH,
                       juce::ComboBox& box) override;

    void drawButtonBackground (juce::Graphics& g, juce::Button& button,
                               const juce::Colour& backgroundColour,
                               bool shouldDrawButtonAsHighlighted,
                               bool shouldDrawButtonAsDown) override;

    void drawButtonText (juce::Graphics& g, juce::TextButton& button,
                         bool shouldDrawButtonAsHighlighted,
                         bool shouldDrawButtonAsDown) override;

    void drawToggleButton (juce::Graphics& g, juce::ToggleButton& button,
                           bool shouldDrawButtonAsHighlighted,
                           bool shouldDrawButtonAsDown) override;

    juce::Font getComboBoxFont (juce::ComboBox& box) override;
    juce::Font getLabelFont (juce::Label& label) override;
    juce::Font getTextButtonFont (juce::TextButton& button, int buttonHeight) override;
};
