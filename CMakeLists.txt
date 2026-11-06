# Melotheon VST3 plugin - CMake build.
#
# Pinned dependency: JUCE 8.0.8 (FetchContent, tag - never an unversioned
# branch). This project is built by GitHub Actions on a Windows runner, so
# the user does NOT need Visual Studio installed locally to get a .vst3.
# A developer who does have VS 2022 can still build locally with the same
# commands (see README.md).
#
# Output: a Windows 64-bit VST3 named exactly "Melotheon".
#   cmake -S . -B build -A x64
#   cmake --build build --config Release
# The .vst3 bundle lands under build/Melotheon_artefacts/Release/VST3/.
#
# Phase 2 scope: audio foundation (capture, import, preview, meter,
# status, real waveform) on top of the Phase 1 interface. Still no
# pitch detection, no MIDI generation, no instrument hosting
# (see README.md). The processor passes audio through unchanged while
# preview is inactive.

cmake_minimum_required(VERSION 3.22)

project(Melotheon VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# JUCE 8.0.8, pinned by tag.
FetchContent_Declare(
  JUCE
  GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
  GIT_TAG 8.0.8
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(Melotheon
    COMPANY_NAME "Potorik Track"
    BUNDLE_ID "com.potoriktrack.melotheon"
    PLUGIN_MANUFACTURER_CODE Ptrk
    PLUGIN_CODE Mlth
    FORMATS VST3
    PRODUCT_NAME "Melotheon"
    DESCRIPTION "Melotheon audio-to-MIDI melody capture for FL Studio"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES "Fx"
)

# REQUIRED (do not remove): juce_add_plugin does NOT create JuceHeader.h
# by itself in JUCE 8.0.8 - this call is what generates it (via the
# juceaide tool at build time). Without it every plugin source fails
# with C1083 "Cannot open include file: 'JuceHeader.h'".
juce_generate_juce_header(Melotheon)

target_sources(Melotheon
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/Audio/AudioCaptureEngine.cpp
        Source/Audio/AudioFileLoader.cpp
        Source/Audio/AudioPreviewEngine.cpp
        Source/UI/MelotheonLookAndFeel.cpp
        Source/UI/PitchEditorComponent.cpp
        Source/UI/LevelMeterComponent.cpp
        Source/UI/IconButton.cpp
)

# /utf-8 on MSVC so the UTF-8 brand string "MELOTHEON" (with E-acute)
# in the source is read as UTF-8. All other UI strings are plain ASCII.
target_compile_options(Melotheon
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

target_compile_definitions(Melotheon
    PRIVATE
        # No embedded web browser in the plugin UI.
        JUCE_WEB_BROWSER=0
        # Use the OS-native HTTP stack on Windows (WinHTTP) instead of
        # requiring libcurl.
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        # No JUCE splash screen in the plugin window.
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(Melotheon
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_formats
        juce::juce_audio_basics
        juce::juce_gui_basics
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# ---- Phase 2.1 unit tests (console app, no GUI, no host) ----
# Covers the audio foundation: stereo-to-mono conversion, mono copy,
# capture limits / state transitions, preview state transitions and
# resampling, file-loader rejection + valid WAV round-trip, AudioClip
# metadata, and processor pass-through. The test target is built by
# the normal build (including Windows CI) so a broken test breaks
# the build. Windows CI also executes it and validates the final VST3
# bundle with pluginval. Run locally with
# `ctest --test-dir build -C Release --output-on-failure`.
include(CTest)

# juce_add_console_app (not plain add_executable): only targets
# created with a juce_add_* function get the generated-sources
# directory that juce_generate_juce_header() needs below.
juce_add_console_app(MelotheonTests)

target_sources(MelotheonTests
    PRIVATE
        Tests/MelotheonTests.cpp
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/Audio/AudioCaptureEngine.cpp
        Source/Audio/AudioFileLoader.cpp
        Source/Audio/AudioPreviewEngine.cpp
        Source/UI/MelotheonLookAndFeel.cpp
        Source/UI/PitchEditorComponent.cpp
        Source/UI/LevelMeterComponent.cpp
        Source/UI/IconButton.cpp
)

# The test sources include <JuceHeader.h> (as the plugin does), so
# the test target gets its own generated header.
juce_generate_juce_header(MelotheonTests)

target_include_directories(MelotheonTests
    PRIVATE
        Source
)

target_compile_options(MelotheonTests
    PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

target_compile_definitions(MelotheonTests
    PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(MelotheonTests
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_audio_formats
        juce::juce_audio_basics
        juce::juce_gui_basics
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

add_test(NAME MelotheonTests COMMAND MelotheonTests)
