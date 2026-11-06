# Melotheon

Melotheon is a professional Windows VST3 plugin for FL Studio: capture a
melody (voice, guitar, piano) and turn it into MIDI on a channel
instrument.

## Phase 1 — visual interface (current)

This phase builds the **complete visual interface only**:

- Header with the custom Melotheon "M" monogram, MELOTHÉON brand,
  Record / Play / Stop transport buttons, input selector, READY status
  LCD and detected-note LCD.
- Left capture panel: IMPORT AUDIO, MONO toggle, NOISE GATE knob and a
  vertical level meter.
- Center piano-roll editor: dark grid, piano keyboard (C3–C5), bar
  numbers 1–4, static placeholder notes with pitch curves, a grey
  waveform and a green playhead — all drawn programmatically.
- Right instrument panel: instrument selector, MIDI OUT toggle, MIDI
  channel selector, DRAG MIDI TO FL, PLAY ORIGINAL, PLAY INSTRUMENT.
- Bottom correction strip: CENTER, VARIATION, TRANSITION, SENSITIVITY
  and QUANTIZE knobs, KEY display, ANALYZE and SEND TO CHANNEL.

Every control is a real JUCE component. Knobs are draggable, dropdowns
open with placeholder choices, and Record / Play / Stop change their
visual selected state only.

## What is NOT implemented yet

- Pitch detection, audio analysis, MIDI generation.
- Instrument hosting, audio recording.
- Choir generation, voice multiplication.
- AI / ONNX models.
- Real Send to Channel / Drag MIDI behaviour.

These are marked `// TODO: PHASE_2_AUDIO_LOGIC` in the code. The
processor passes incoming audio through unchanged.

## Build (Windows, GitHub Actions)

Push to `main` (or run the workflow manually): the **Windows** workflow
builds the VST3 on a `windows-latest` runner and uploads
`melotheon-vst3` as an artifact. No local Visual Studio install needed.

## Build (local, Visual Studio 2022)

```bat
cmake -S . -B build -A x64
cmake --build build --config Release
```

The `.vst3` bundle lands under
`build/Melotheon_artefacts/Release/VST3/`. Copy it to the VST3 folder
(usually `C:\Program Files\Common Files\VST3\`) and rescan plugins in
FL Studio.

## Tech

- C++20, JUCE 8.0.8 (pinned via FetchContent), CMake, VST3 only.
- Editor: 1400 × 560 default (1000 × 400 min, 1800 × 720 max),
  aspect-ratio-preserving resize, HiDPI-safe.
- Design tokens (colours, fonts, sizes) live in
  `Source/UI/MelotheonLookAndFeel.h` — one central place.
