# Melotheon

Melotheon is a professional Windows VST3 plugin for FL Studio: capture a
melody (voice, guitar, piano) and turn it into MIDI on a channel
instrument.

## Phase 1 — visual interface (complete)

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

## Phase 2.1 — stable audio foundation (complete)

Phase 2 connects the audio foundation to the Phase 1 interface, with
no change to the layout, colours, sizes or brand text:

- **Live capture:** audio arriving from the FL Studio Mixer Track
  through `processBlock()` is recorded into a preallocated 60 s
  buffer (Record / Stop). No allocation happens in `processBlock()`.
- **Import Audio:** WAV, AIFF and FLAC (plus MP3 when the JUCE
  configuration enables it) are decoded on a worker thread via
  `juce::FileChooser::launchAsync()`; the UI never freezes.
- **Preview:** Play / Stop / Play Original play the active clip
  through the Mixer Track, resampled to the host rate so speed and
  pitch stay correct. The green playhead follows playback.
- **Meter:** the vertical meter shows the real smoothed input peak
  (-60..0 dB; green -60..-12, yellow -12..-3, orange above -3).
- **Waveform:** the real clip waveform replaces the placeholder; the
  placeholder notes are hidden and the editor shows
  "AUDIO READY · CLICK ANALYZE".
- **Status LCD:** READY / LOADING / RECORDING / AUDIO READY /
  PLAYING / STOPPED / NO INPUT / ERROR / LIMIT REACHED / EMPTY SOURCE.
- **Input source:** LIVE INPUT (record host audio) or IMPORTED AUDIO
  (play the loaded file / recorded clip).

The active clip is an immutable `AudioClip` (original channels kept
for preview, separate cancellation-safe mono copy for analysis). Ownership stays
on the message thread. The audio callback reads a stable non-owning
pointer, so `processBlock()` never performs shared-pointer reference
counting or destruction.

Phase 2.1 also hardens the working foundation:

- Record and Play are mutually exclusive. Play finalizes a valid take;
  Record stops and resets preview.
- Stop waits for the current capture copy before publishing the clip.
- An empty take reports NO INPUT and never replaces the active clip.
- Stop/Pause always wins over an in-flight preview render.
- Closing the editor while the file chooser is open is safe.
- A pending import cannot replace a clip after Record or Play takes over.

## Phase 3A — monophonic pitch map (current)

ANALYZE now performs real offline pitch tracking without changing the
approved interface layout:

- Voice, whistle, solo guitar and other one-note-at-a-time sources.
- 55–2500 Hz working range, 24 kHz internal analysis rate and one
  pitch/confidence frame every 10 ms.
- Two-resolution YIN tracking: a longer low-frequency window plus a
  short high-frequency window, harmonic selection and light glitch
  stabilization.
- Analysis runs on a cancellable worker thread. FL Studio's audio
  callback and the plugin UI remain responsive.
- The central editor overlays the real pitch curve on the waveform.
  Its orange opacity reflects confidence and its keyboard range adapts
  to the detected register.
- The header shows the current note, Hertz and confidence while the
  original clip plays. ANALYZING progress and NO STABLE PITCH are
  explicit states.
- Strong anti-phase stereo inputs cannot erase the analysis signal:
  Melotheon selects the safer source channel while preserving the
  original stereo audio for preview.

Phase 3A intentionally produces a continuous pitch map, not editable
notes or MIDI. That separation lets the detector be validated before
note boundaries and MIDI timing are added.

## What is NOT implemented yet

- Note segmentation, editable note blocks and MIDI generation/export.
- Instrument hosting, choir generation, voice multiplication.
- AI / ONNX models.
- Real Send to Channel / Drag MIDI behaviour.
- SEND TO CHANNEL, DRAG MIDI TO FL and PLAY INSTRUMENT are still
  non-functional.

These are marked `// TODO: PHASE_3_AUDIO_LOGIC` in the code.

## Build (Windows, GitHub Actions)

Push to `main` (or run the workflow manually): the **Windows** workflow
builds the VST3, executes the CTest suite, validates the real VST3 bundle
with pluginval strictness level 5, then uploads `melotheon-vst3` as an
artifact. No local Visual Studio install is needed.

## Build (local, Visual Studio 2022)

```bat
cmake -S . -B build -A x64
cmake --build build --config Release
```

The `.vst3` bundle lands under
`build/Melotheon_artefacts/Release/VST3/`. Copy it to the VST3 folder
(usually `C:\Program Files\Common Files\VST3\`) and rescan plugins in
FL Studio.

Run the Phase 3A unit tests after any build:

```bat
ctest --test-dir build -C Release --output-on-failure
```

## Tech

- C++20, JUCE 8.0.8 (pinned via FetchContent), CMake, VST3 only.
- Editor: 1400 × 560 default (1000 × 400 min, 1800 × 720 max),
  aspect-ratio-preserving resize, HiDPI-safe.
- Design tokens (colours, fonts, sizes) live in
  `Source/UI/MelotheonLookAndFeel.h` — one central place.
