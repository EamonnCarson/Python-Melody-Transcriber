import numpy as np
import scipy.signal as sig
import scipy.io.wavfile
import matplotlib.pyplot as plt

class Note:
    str_template = 'Play note {midi} at volume {amplitude} at time {note_start_time} for duration {duration}.'

    def __init__(self,
                 midi=None,
                 amplitude=None,
                 note_start_time=None,
                 duration=None,
                 midi_vibrato=None,
                 amplitude_vibrato=None):
        self.midi = midi
        self.amplitude = amplitude
        self.note_start_time = note_start_time
        self.duration = duration
        self.midi_vibrato = midi_vibrato
        self.amplitude_vibrato = amplitude_vibrato

    def __str__(self):
        return self.str_template.format(self.midi,
                                        self.amplitude,
                                        self.note_start_time,
                                        self.duration)

def visualize_transcription(transcription, signal, sampling_rate):
    """
    Given a transcription produces a graph of relevant attributes
    :param transcription: a list of Note objects
    :param signal: the original signal
    :return: N/A
    """
    delta = 0.001
    note_edges = []
    midi = []
    amplitude = []
    for note in transcription:
        # before note start
        note_edges.append(note.note_start_time - delta)
        midi.append(0)
        amplitude.append(0)
        # note start
        note_edges.append(note.note_start_time)
        midi.append(note.midi)
        amplitude.append(note.amplitude)
        # note end
        note_edges.append(note.note_start_time + note.duration)
        midi.append(note.midi)
        amplitude.append(note.amplitude)
        # after note end
        note_edges.append(note.note_start_time + note.duration + delta)
        midi.append(0)
        amplitude.append(0)
    plt.subplot(211)
    plt.plot(midi, note_edges, 'r-')
    plt.plot(amplitude, note_edges, 'b--')
    plt.subplot(212)
    f, t, Sxx = sig.spectrogram(signal, sampling_rate)
    plt.pcolormesh(t, f, Sxx)
    plt.show()

def get_mono_wav(filename):
    """
    Opens the wav file stored at filename and return a
    numpy array that represents the signal values (single channel
    :param filename: the name of the file to be read from
    :return: a numpy array
    """
    try:
        sampling_rate, wav_data = scipy.io.wavfile.read(filename);
        wav_data = np.array(wav_data)
    except:
        print('File "{}" could not be opened.'.format(filename))
        exit(1)
    # combine channels into mono
    if len(wav_data.shape) > 1:
        mono_wav_data = np.sum(wav_data, -1).ravel()
    else:
        mono_wav_data = wav_data
    return (sampling_rate, mono_wav_data)

def clean_data(time, frequency, min_freq=0, max_freq=4000):
    """
    Removes None values and values outside of the specified range
    :param time:
    :param frequency:
    :param min_freq:
    :param max_freq:
    :return:
    """
    # filter out nan values
    valid  = ~np.isnan(frequency)
    time = time[valid]
    frequency = frequency[valid]
    # filter out values that we deem invalid
    above_min = (frequency >= min_freq)
    below_max = (frequency <= max_freq)
    time = time[above_min & below_max]
    frequency = frequency[above_min & below_max]
    return (time, frequency)

def frequency_to_midi(frequency):
    """
    Converts frequency array into midi array
    :param frequency:
    :return:
    """
    midi_array = 69 + 12 * np.log2(frequency / 440.)
    return midi_array

def detect_notes(time, midi, min_time_gap, min_midi_gap, minimum_note_length):
    """

    :param time:
    :param midi:
    :param min_time_gap: the minimum gap between the same note to register as a new note
    :param min_midi_gap: the minimum pitch difference (in midi) to register a new note
    :return:
    """
    # detect time gaps
    time_forward_difference = time[1:] - time[:-1]
    time_gap_exists = (time_forward_difference >= min_time_gap)
    time_gap_ends   = (~time_gap_exists[1:] &  time_gap_exists[:-1]) # true then false
    time_gap_starts = ( time_gap_exists[1:] & ~time_gap_exists[:-1]) # false then true
    # detect pitch gaps
    midi_forward_difference = midi[1:] - midi[:-1]
    midi_gap_exists = (midi_forward_difference >= min_midi_gap)
    # calculate ranges
    index_offset = 0 # TODO should this be 0, 1, or 2?
    note_starts = time_gap_starts | midi_gap_exists
    note_ends   = time_gap_ends   | midi_gap_exists
    inside_interval = False
    current_start = None
    intervals = []
    for i in range(0, len(note_starts)):
        if inside_interval and note_ends[i]:
            if i - current_start >= minimum_note_length:
                intervals.append((current_start + index_offset, i + index_offset))
                current_start = None
                inside_interval = False
            else: # invalid note
                current_start = None
                inside_interval = False
        if not inside_interval and note_starts[i]:
            current_start = i
            inside_interval = True
    return intervals

def detect_notes_by_bins(time, midi, minimum_note_duration=0.1):
    # round midi into integers
    discrete_midi = (np.around(midi) + 0.1).astype(int) # +0.1 to ensure proper truncation
    # collapse contiguous values
    notes = []
    anchor = 0
    for i in range(1, len(time)):
        if (discrete_midi[anchor] != discrete_midi[i]):
            # the note has ended
            note_duration = time[i-1] - time[anchor]
            note_start = time[anchor]
            note_midi = discrete_midi[i]
            if (note_duration >= minimum_note_duration):
                note = (note_midi, note_start, note_duration)
                notes.append(note)
            anchor = i
    return notes

def get_note_representation(time, frequency, note_bounds):
    notes = []
    for start, end in note_bounds:
        time_window = time[start:end]
        freq_window = frequency[start:end]
        clean_time, clean_freq = clean_data(time_window, freq_window)
        duration = clean_time[-1] - clean_time[0]
        note_start = clean_time[0]
        pitch_hz = np.mean(clean_freq) #TODO: should I be averaging the frequency or the MIDI note?
        midi = 69 + 12 * np.log2(pitch_hz / 440.)
        note = Note(midi, 1, note_start, duration, 0, 0)
        notes.append(note)
    for note in notes:
        print(note)
