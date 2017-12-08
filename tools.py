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
    max_time = signal.size / sampling_rate
    delta = 0.001
    note_edges = []
    midi = []
    amplitude = []
    # prepend origin
    note_edges.append(0)
    midi.append(0)
    amplitude.append(0)
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
    # append end
    note_edges.append(max_time)
    midi.append(0)
    amplitude.append(0)

    # plot
    plt.subplot(211)
    plt.plot(note_edges, midi, 'r-')
    plt.plot(note_edges, amplitude, 'b--')
    plt.subplot(212)
    plt.plot(np.linspace(0, max_time, signal.size), signal, 'g-')
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

