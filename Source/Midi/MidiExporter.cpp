#include "MidiExporter.h"

#include <algorithm>
#include <cmath>

juce::MidiFile MidiExporter::createMidiFile (
    const MelodyNoteSequence& sequence,
    const MidiExportSettings& requestedSettings)
{
    MidiExportSettings settings = requestedSettings;
    settings.bpm = std::clamp (settings.bpm, 20.0, 400.0);
    settings.channel = std::clamp (settings.channel, 1, 16);
    settings.ticksPerQuarterNote = std::clamp (settings.ticksPerQuarterNote, 96, 9600);

    const double ticksPerSecond = settings.bpm
                                * static_cast<double> (settings.ticksPerQuarterNote) / 60.0;

    juce::MidiMessageSequence track;
    // Standard MIDI meta-event type 3 is the sequence/track name.
    auto trackName = juce::MidiMessage::textMetaEvent (3, "Melotheon Melody");
    trackName.setTimeStamp (0.0);
    track.addEvent (trackName);

    auto tempo = juce::MidiMessage::tempoMetaEvent (
        static_cast<int> (std::llround (60000000.0 / settings.bpm)));
    tempo.setTimeStamp (0.0);
    track.addEvent (tempo);

    auto timeSignature = juce::MidiMessage::timeSignatureMetaEvent (4, 4);
    timeSignature.setTimeStamp (0.0);
    track.addEvent (timeSignature);

    double lastTick = 0.0;
    for (const MelodyNote& note : sequence.notes)
    {
        if (note.endSeconds <= note.startSeconds)
            continue;

        const double startTick = std::max (0.0, note.startSeconds * ticksPerSecond);
        const double endTick = std::max (startTick + 1.0,
                                         note.endSeconds * ticksPerSecond);
        const auto velocity = static_cast<juce::uint8> (std::clamp (
            static_cast<int> (std::lround (note.velocity * 126.0f)), 1, 127));

        auto noteOn = juce::MidiMessage::noteOn (
            settings.channel, std::clamp (note.midiNote, 0, 127), velocity);
        noteOn.setTimeStamp (startTick);
        track.addEvent (noteOn);

        auto noteOff = juce::MidiMessage::noteOff (
            settings.channel, std::clamp (note.midiNote, 0, 127));
        noteOff.setTimeStamp (endTick);
        track.addEvent (noteOff);
        lastTick = std::max (lastTick, endTick);
    }

    auto endOfTrack = juce::MidiMessage::endOfTrack();
    endOfTrack.setTimeStamp (lastTick + 1.0);
    track.addEvent (endOfTrack);
    track.sort();
    track.updateMatchedPairs();

    juce::MidiFile midiFile;
    midiFile.setTicksPerQuarterNote (settings.ticksPerQuarterNote);
    midiFile.addTrack (track);
    return midiFile;
}

bool MidiExporter::writeToFile (const MelodyNoteSequence& sequence,
                                const juce::File& target,
                                const MidiExportSettings& settings,
                                juce::String& errorMessage)
{
    errorMessage.clear();
    if (sequence.empty())
    {
        errorMessage = "ANALYZE FIRST";
        return false;
    }
    if (target == juce::File{})
    {
        errorMessage = "INVALID MIDI PATH";
        return false;
    }

    const juce::File parent = target.getParentDirectory();
    if (! parent.isDirectory())
    {
        errorMessage = "INVALID MIDI PATH";
        return false;
    }

    // FileChooser already requested overwrite confirmation. Delete the old
    // file first so FileOutputStream cannot append to an existing SMF.
    if (target.existsAsFile() && ! target.deleteFile())
    {
        errorMessage = "MIDI FILE IN USE";
        return false;
    }

    juce::FileOutputStream output (target);
    if (! output.openedOk())
    {
        errorMessage = "MIDI WRITE ERROR";
        return false;
    }

    auto midiFile = createMidiFile (sequence, settings);
    if (! midiFile.writeTo (output, 1))
    {
        errorMessage = "MIDI WRITE ERROR";
        return false;
    }
    output.flush();
    return output.getStatus().wasOk();
}
