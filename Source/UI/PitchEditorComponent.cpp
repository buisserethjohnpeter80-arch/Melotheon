// PitchEditorComponent implementation. All drawing is proportional to
// the component bounds, so the editor stays aligned when resized.
// Static placeholder data only - see the header.

#include "PitchEditorComponent.h"
#include "MelotheonLookAndFeel.h"
#include <cmath>

const PitchEditorComponent::Note PitchEditorComponent::placeholderNotes[] = {
    { 0.0, 1.0, 48 }, // C3
    { 1.0, 1.0, 50 }, // D3
    { 2.0, 1.0, 52 }, // E3
    { 3.0, 1.0, 55 }, // G3
    { 4.0, 2.0, 57 }, // A3
    { 6.0, 1.0, 59 }, // B3
    { 7.0, 1.0, 60 }, // C4
    { 8.0, 2.0, 62 }, // D4
    { 10.0, 1.0, 64 }, // E4
    { 11.0, 1.0, 67 }, // G4
    { 12.0, 2.0, 69 }, // A4 (melody peak)
    { 14.0, 2.0, 57 }, // A3
};

bool PitchEditorComponent::isBlackKey (int midiNote) const
{
    const int pc = midiNote % 12;
    return pc == 1 || pc == 3 || pc == 6 || pc == 8 || pc == 10;
}

juce::String PitchEditorComponent::noteName (int midiNote) const
{
    static const char* names[] = { "C", "C#", "D", "D#", "E", "F",
                                   "F#", "G", "G#", "A", "A#", "B" };
    return juce::String (names[midiNote % 12])
         + juce::String ((midiNote / 12) - 1);
}

void PitchEditorComponent::paint (juce::Graphics& g)
{
    using namespace melotheon;
    const float width = static_cast<float> (getWidth());
    const float height = static_cast<float> (getHeight());

    g.setColour (Tokens::pitchBackground);
    g.fillRect (0.0f, 0.0f, width, height);

    const float gridX = keyboardWidth;
    const float gridY = barNumberHeight;
    const float gridW = width - keyboardWidth;
    const float gridH = height - barNumberHeight;
    const int numRows = highestMidi - lowestMidi + 1; // 25 semitones
    const float rowH = gridH / static_cast<float> (numRows);
    const float beatW = gridW / static_cast<float> (totalBeats);

    auto midiToY = [&] (int midiNote) -> float
    {
        // Higher notes are higher on screen.
        return gridY + static_cast<float> (highestMidi - midiNote) * rowH;
    };
    auto beatToX = [&] (double beat) -> float
    {
        return gridX + static_cast<float> (beat) * beatW;
    };

    // ---- Bar numbers 1-4 across the top. ----
    g.setFont (melotheonFont (12.0f, true));
    g.setColour (Tokens::secondaryText);
    for (int bar = 0; bar < 4; ++bar)
    {
        g.drawText (juce::String (bar + 1),
                    juce::Rectangle<float> (beatToX (bar * beatsPerBar) + 4.0f, 0.0f,
                                            beatW, barNumberHeight),
                    juce::Justification::centredLeft, true);
    }

    // ---- Horizontal semitone lines; stronger lines on C (octaves). ----
    for (int midi = lowestMidi; midi <= highestMidi + 1; ++midi)
    {
        const float y = midiToY (midi);
        const bool isOctave = (midi % 12 == 0);
        g.setColour (isOctave ? Tokens::border.withAlpha (0.8f) : Tokens::gridLine);
        g.drawLine (gridX, y, width, y, isOctave ? 1.0f : 0.5f);
    }

    // ---- Vertical beat lines; stronger lines on bar starts. ----
    for (int beat = 0; beat <= totalBeats; ++beat)
    {
        const float x = beatToX (beat);
        const bool isBar = (beat % beatsPerBar == 0);
        g.setColour (isBar ? Tokens::border.withAlpha (0.8f) : Tokens::gridLine);
        g.drawLine (x, gridY, x, height, isBar ? 1.0f : 0.5f);
    }

    // ---- Dense grey waveform behind the notes. ----
    // Deterministic filled shape (layered sines + a slow envelope
    // varying 0.4..1.0), mirrored around the grid centre so it fills
    // much of the editor height like the reference image.
    // Placeholder only.
    {
        const float centreY = gridY + gridH * 0.5f;
        const float maxAmp = gridH * 0.30f;
        const int steps = 420;
        auto waveOffset = [&] (float t) -> float
        {
            const float env = 0.4f + 0.6f * (0.5f + 0.5f * std::sin (t * 6.28318f * 1.5f + 0.9f));
            return maxAmp * env
                * (0.55f * std::sin (t * 6.28318f * 9.0f)
                   + 0.30f * std::sin (t * 6.28318f * 23.0f + 1.7f)
                   + 0.15f * std::sin (t * 6.28318f * 41.0f + 0.4f));
        };
        juce::Path top;
        for (int i = 0; i <= steps; ++i)
        {
            const float t = static_cast<float> (i) / static_cast<float> (steps);
            const float x = gridX + t * gridW;
            const float y = centreY - waveOffset (t);
            if (i == 0)
                top.startNewSubPath (x, y);
            else
                top.lineTo (x, y);
        }
        // Filled mirrored shape: top edge left-to-right, bottom edge
        // right-to-left.
        juce::Path fill = top;
        for (int i = steps; i >= 0; --i)
        {
            const float t = static_cast<float> (i) / static_cast<float> (steps);
            fill.lineTo (gridX + t * gridW, centreY + waveOffset (t));
        }
        fill.closeSubPath();
        g.setColour (Tokens::waveformGrey.withAlpha (0.35f));
        g.fillPath (fill);
        // 1 px signal line over the fill.
        g.setColour (Tokens::waveformGrey.withAlpha (0.5f));
        g.strokePath (top, juce::PathStrokeType (1.0f));
    }

    // ---- Orange monophonic note blocks. ----
    for (int i = 0; i < numPlaceholderNotes; ++i)
    {
        const Note& note = placeholderNotes[i];
        const float x = beatToX (note.startBeat) + 1.0f;
        const float w = static_cast<float> (note.lengthBeats) * beatW - 2.0f;
        const float y = midiToY (note.midiNote) + 1.0f;
        const float h = rowH - 2.0f;
        juce::Rectangle<float> block (x, y, w, h);

        g.setColour (Tokens::activeOrange);
        g.fillRoundedRectangle (block, 2.0f);
        g.setColour (Tokens::brightOrange);
        g.drawRoundedRectangle (block, 2.0f, 1.0f);
    }

    // ---- ONE continuous stepped pitch curve over all notes. ----
    // The line climbs from note to note (portamento-style ramps at
    // the boundaries) and falls back down, like the reference image:
    // within each note it eases in from the previous note's pitch and
    // eases out toward the next note's pitch, with only a small
    // vibrato in the middle. Every sample is clamped to its own
    // note block, so the curve never leaves the blocks.
    {
        juce::Path curve;
        bool firstPoint = true;
        for (int i = 0; i < numPlaceholderNotes; ++i)
        {
            const Note& note = placeholderNotes[i];
            const float x = beatToX (note.startBeat) + 1.0f;
            const float w = static_cast<float> (note.lengthBeats) * beatW - 2.0f;
            const float y = midiToY (note.midiNote) + 1.0f;
            const float h = rowH - 2.0f;
            const float centreY = y + h * 0.5f;
            const float prevCentreY = (i == 0)
                ? centreY
                : midiToY (placeholderNotes[i - 1].midiNote) + rowH * 0.5f;
            const float nextCentreY = (i == numPlaceholderNotes - 1)
                ? centreY
                : midiToY (placeholderNotes[i + 1].midiNote) + rowH * 0.5f;
            const float startY = juce::jlimit (y + 1.0f, y + h - 1.0f, prevCentreY);
            const float endY = juce::jlimit (y + 1.0f, y + h - 1.0f, nextCentreY);
            const int steps = 48;
            for (int s = 0; s <= steps; ++s)
            {
                const float t = static_cast<float> (s) / static_cast<float> (steps);
                float py = centreY;
                if (t < 0.15f)
                    py = startY + (centreY - startY) * (t / 0.15f);
                else if (t > 0.85f)
                    py = centreY + (endY - centreY) * ((t - 0.85f) / 0.15f);
                py += std::sin (t * 6.28318f * 2.0f) * h * 0.15f;
                py = juce::jlimit (y + 1.0f, y + h - 1.0f, py);
                const float px = x + t * w;
                if (firstPoint)
                {
                    curve.startNewSubPath (px, py);
                    firstPoint = false;
                }
                else
                {
                    curve.lineTo (px, py);
                }
            }
        }
        g.setColour (Tokens::brightOrange);
        g.strokePath (curve, juce::PathStrokeType (1.5f));
    }

    // ---- Piano keyboard on the left. ----
    {
        // White-key background first.
        g.setColour (juce::Colour (0xffe8e8e8));
        g.fillRect (0.0f, gridY, keyboardWidth, gridH);
        // White keys: one rectangle per white note.
        for (int midi = lowestMidi; midi <= highestMidi; ++midi)
        {
            if (isBlackKey (midi))
                continue;
            const float y = midiToY (midi);
            g.setColour (juce::Colour (0xfff2f2f2));
            g.fillRect (0.0f, y, keyboardWidth, rowH);
            g.setColour (juce::Colour (0xff9a9a9a));
            g.drawLine (0.0f, y + rowH, keyboardWidth, y + rowH, 0.5f);
        }
        // Black keys drawn over the white keys.
        for (int midi = lowestMidi; midi <= highestMidi; ++midi)
        {
            if (! isBlackKey (midi))
                continue;
            const float y = midiToY (midi);
            g.setColour (juce::Colour (0xff1a1a1a));
            g.fillRect (0.0f, y, keyboardWidth * 0.62f, rowH);
        }
        // C3 / C4 / C5 labels on the white C keys.
        g.setFont (melotheonFont (11.0f, true));
        g.setColour (juce::Colour (0xff333333));
        for (int midi : { 48, 60, 72 })
        {
            const float y = midiToY (midi);
            g.drawText (noteName (midi),
                        juce::Rectangle<float> (keyboardWidth - 24.0f, y, 22.0f, rowH),
                        juce::Justification::centredLeft, true);
        }
        // Right edge of the keyboard.
        g.setColour (Tokens::border);
        g.drawLine (keyboardWidth, gridY, keyboardWidth, height, 1.0f);
    }

    // ---- Green vertical playhead (bar 2, beat 5.5) with a marker. ----
    {
        const float playheadX = beatToX (5.5);
        g.setColour (Tokens::statusGreen);
        g.drawLine (playheadX, gridY, playheadX, height, 1.5f);
        juce::Path marker;
        marker.startNewSubPath (playheadX - 6.0f, 0.0f);
        marker.lineTo (playheadX + 6.0f, 0.0f);
        marker.lineTo (playheadX, 9.0f);
        marker.closeSubPath();
        g.fillPath (marker);
    }

    // ---- Outer border. ----
    g.setColour (Tokens::border);
    g.drawRect (0.0f, 0.0f, width, height, 1.0f);
}
