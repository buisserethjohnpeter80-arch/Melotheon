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

## Phase 3C — melody notes, MIDI and sample instrument (current)

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

The continuous curve is now converted into stable, non-overlapping
notes. Five-frame median smoothing, pitch hysteresis, short-gap repair
and minimum-note filtering keep normal vibrato from creating unwanted
MIDI notes. The same note sequence powers these features:

- **EXPORT MIDI:** saves a standard `.mid` file using the selected MIDI
  channel and the current FL Studio tempo when available.
- **MIDI chip:** one chip opens a small panel with two choices.
  **DRAG MIDI TO FL** starts a real OS file drag of a temp `.mid` so
  the melody can be dropped into FL Studio's Playlist or Channel
  Rack. **EXPORT MIDI** runs the save-file flow above. A plain click
  on the drag row shows "PRESS AND DRAG THIS ROW INTO FL STUDIO" in the status
  LCD instead of doing nothing, and the same instruction is printed
  as a caption under the rows so it is visible before any click;
  while a drag is in flight the row stays highlighted orange. The
  temp file is owned by the editor and deleted only after the drag
  ends, so the panel cannot dismiss under a drag in flight.
- **Draggable playhead:** grab the green playhead line in the piano
  roll and drag it left or right to seek; playback continues from
  the new position, or Play starts from there.
- **Piano-key audition:** click a piano key on the left of the piano
  roll and it really plays: the loaded one-shot pitched from its
  detected root, or a built-in sine tone when no sound is loaded, so
  a key always makes sound. Release the mouse (or slide off the key)
  to stop it; sliding across keys while held re-triggers them. The
  note name and Hertz show in the note LCD while the key is held.
- **Manual pitch-curve correction:** the orange vocal pitch curve is
  not a static picture. Grab it with the mouse and drag vertically
  to correct the pitch at that time (semitone steps; the note name
  shows in the note LCD while dragging). On release the frames near
  the grab point are adjusted with a smooth falloff and the notes
  are re-segmented, so the note blocks follow the corrected curve.
  The status LCD notes "CURVE EDITED - RE-ANALYZE TO RESET": a later
  RE-ANALYZE overwrites manual edits. The grey waveform is the
  recorded audio and is never editable.
- **Load Sound:** the folder button in INSTRUMENT loads a WAV, AIFF,
  FLAC or supported MP3 one-shot. Melotheon detects its root pitch,
  renders the sung timing with that sound off the audio thread and lets
  PLAY INSTRUMENT preview it. Unpitched sounds safely default to C4.
- **Professional piano navigation:** real note blocks, a wider keyboard,
  FIT, `-` / `+` vertical zoom, a scrollbar and mouse-wheel scrolling.

## What is NOT implemented yet

- Direct note-block dragging (notes follow the corrected pitch curve,
  but blocks themselves cannot be moved or resized with the mouse).
- Third-party instrument hosting, choir generation, voice multiplication.
- AI / ONNX models.
- Real SEND TO CHANNEL and live MIDI output routing.

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

Run the Phase 3C unit tests after any build:

```bat
ctest --test-dir build -C Release --output-on-failure
```

## Tech

- C++20, JUCE 8.0.8 (pinned via FetchContent), CMake, VST3 only.
- Editor: 1400 × 560 default (1000 × 400 min, 1800 × 720 max),
  aspect-ratio-preserving resize, HiDPI-safe.
- Design tokens (colours, fonts, sizes) live in
  `Source/UI/MelotheonLookAndFeel.h` — one central place.
