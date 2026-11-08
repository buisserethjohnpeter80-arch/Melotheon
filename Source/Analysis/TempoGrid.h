// TempoGrid: the piano-roll beat grid, derived from the HOST tempo.
//
// The grid used to divide every clip into 16 equal parts, so a 20 s
// clip showed a "beat" of 1.25 s while FL at 120 BPM has a 0.5 s beat.
// Now one beat is 60 / BPM seconds and bars come from the host time
// signature, so the drawn grid, the bar numbers and the MOVE/CUT/
// STRETCH snap targets all agree with FL's tempo.
//
// Bar phase: bar 1 starts at the clip start. The clip's position in
// the host project is unknowable — an imported file has none, and a
// live take may start anywhere (transport stopped, seeks) — so the
// host PPQ position is read and cached (see the processor) but never
// used to phase the grid. Inventing a project position would put bar
// lines in the wrong place; clip-start phasing is always honest.
//
// Notes themselves are stored in seconds (see MelodyNotes.h), so a
// BPM change only moves grid lines and snap targets, never the notes.
//
// Pure value type — no JUCE, no threads — so the grid math and the
// tool-strip placement are unit-tested without a plugin window.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

struct TempoGrid
{
    // Host BPM. 120 is ONLY the fallback when the host provides no
    // tempo (standalone, pluginval, transport stopped with no
    // playhead). Clamped to a sane range so a bad host value cannot
    // produce an absurd grid.
    double bpm = 120.0;
    // Time signature numerator / denominator, e.g. 3 and 4 for 3/4.
    int timeSignatureNumerator = 4;
    int timeSignatureDenominator = 4;
    // Last known host PPQ position in quarter notes, or -1 when the
    // host never provided one. Cached for diagnostics and future
    // project-position features; it does NOT phase the grid (see
    // above).
    double ppqPosition = -1.0;

    static constexpr double kFallbackBpm = 120.0;
    static constexpr double kMinimumBpm = 20.0;
    static constexpr double kMaximumBpm = 300.0;

    double clampedBpm() const
    {
        if (! std::isfinite (bpm) || bpm <= 0.0)
            return kFallbackBpm;
        return std::clamp (bpm, kMinimumBpm, kMaximumBpm);
    }

    // Seconds per quarter-note beat.
    double beatSeconds() const { return 60.0 / clampedBpm(); }

    // Quarter-note beats per bar. 4/4 -> 4, 3/4 -> 3, 6/8 -> 3
    // (six eighth notes), 2/2 -> 4. May be fractional for odd meters
    // such as 5/8 (2.5); bar starts are still exact positions.
    double beatsPerBar() const
    {
        const int numerator = std::clamp (timeSignatureNumerator, 1, 16);
        const int denominator = std::clamp (timeSignatureDenominator, 1, 16);
        return static_cast<double> (numerator) * 4.0
             / static_cast<double> (denominator);
    }

    double barSeconds() const { return beatSeconds() * beatsPerBar(); }

    // Whole beats covering [0, durationSeconds]. The last beat may be
    // partial; its line falls past the right edge (the beat width is
    // seconds-based, so a beat is always beatSeconds() of the clip)
    // and is not drawn.
    int totalBeats (double durationSeconds) const
    {
        if (durationSeconds <= 0.0 || beatSeconds() <= 0.0)
            return 0;
        return static_cast<int> (std::ceil (durationSeconds / beatSeconds()));
    }

    // Whole bars covering [0, durationSeconds]; at least 1 when the
    // clip has any length, so bar 1 is always drawn.
    int totalBars (double durationSeconds) const
    {
        if (durationSeconds <= 0.0 || barSeconds() <= 0.0)
            return 0;
        return std::max (1, static_cast<int> (
            std::ceil (durationSeconds / barSeconds())));
    }

    double beatToSeconds (double beat) const { return beat * beatSeconds(); }
    double secondsToBeat (double seconds) const { return seconds / beatSeconds(); }

    // Beat position (possibly fractional) of the start of bar
    // `barIndex` (0-based: bar 1 starts at beat 0).
    double barStartBeat (int barIndex) const
    {
        return static_cast<double> (barIndex) * beatsPerBar();
    }

    // 0-based bar index containing `beat`.
    int barIndexForBeat (double beat) const
    {
        if (beatsPerBar() <= 0.0)
            return 0;
        return std::max (0, static_cast<int> (std::floor (beat / beatsPerBar())));
    }

    // Snap a time to the nearest beat, clamped to the clip.
    double snapSeconds (double seconds, double durationSeconds) const
    {
        if (durationSeconds <= 0.0)
            return seconds;
        const double beat = std::round (secondsToBeat (seconds));
        return std::clamp (beatToSeconds (beat), 0.0, durationSeconds);
    }

    // Snap only when the time is already within toleranceBeats of a
    // beat (used by CUT, so an off-grid click keeps its exact time).
    double snapSecondsIfClose (double seconds, double durationSeconds,
                               double toleranceBeats = 0.25) const
    {
        if (durationSeconds <= 0.0)
            return seconds;
        const double beat = secondsToBeat (seconds);
        const double nearest = std::round (beat);
        if (std::abs (beat - nearest) <= toleranceBeats)
            return std::clamp (beatToSeconds (nearest), 0.0, durationSeconds);
        return seconds;
    }

    bool operator== (const TempoGrid& other) const
    {
        return std::abs (clampedBpm() - other.clampedBpm()) < 0.001
            && timeSignatureNumerator == other.timeSignatureNumerator
            && timeSignatureDenominator == other.timeSignatureDenominator;
    }

    bool operator!= (const TempoGrid& other) const { return ! (*this == other); }
};

// Tool-strip placement for the bar-number strip: given the x
// positions of the bar numbers (bar starts), find the left edge for a
// strip of `stripWidth` pixels inside [zoneLeft, zoneRight] that does
// not cover any bar number. Bar numbers occupy [barX + 2,
// barX + numberWidth]. The first free gap wide enough wins; when no
// gap fits, the strip is right-aligned to the zone and nudged left
// past any number it would cover, then clamped to the zone. That
// clamp is best-effort: when the zone is too narrow to fit the strip
// without covering a number, the strip lands at zoneLeft and may
// cover one — there is no placement that avoids it. Pure function
// so the placement is unit-tested without a component.
inline int toolStripLeftForBarStarts (const std::vector<float>& barStarts,
                                      float zoneLeft, float zoneRight,
                                      float stripWidth,
                                      float numberWidth = 26.0f)
{
    const float need = stripWidth + 4.0f;

    // Free intervals between the occupied bar-number intervals.
    float freeStart = zoneLeft;
    for (const float barX : barStarts)
    {
        const float occupiedStart = barX + 2.0f;
        const float occupiedEnd = barX + numberWidth;
        if (occupiedStart - freeStart >= need)
            return static_cast<int> (std::floor (freeStart + 2.0f));
        freeStart = std::max (freeStart, occupiedEnd);
    }
    if (zoneRight - freeStart >= need)
        return static_cast<int> (std::floor (freeStart + 2.0f));

    // No gap fits: right-align, then nudge left past any number the
    // strip would cover.
    float left = zoneRight - stripWidth;
    for (const float barX : barStarts)
    {
        const float occupiedStart = barX + 2.0f;
        const float occupiedEnd = barX + numberWidth;
        if (left < occupiedEnd && left + stripWidth > occupiedStart)
            left = occupiedStart - stripWidth - 2.0f;
    }
    return static_cast<int> (std::floor (std::max (left, zoneLeft)));
}
