import numpy as np
import scipy.signal as sig
import mpm
import tools
import matplotlib.pyplot as plt
import soundfile

# GENERAL UTILITIES

def get_hilbert_envelope(signal):
    """
    Given a real signal (represented as an array of displacements) this
    function returns the amplitude envelope for that signal.
    :param signal: signal you want the envelope for (np.array)
    :return: amplitude envelope (np.array)
    """
    analytic_signal = sig.hilbert(signal)
    envelope = np.abs(analytic_signal)
    return np.abs(analytic_signal)

def get_processed_envelope(signal, sampling_rate, compression_alpha=2, window_size=512,
                           min_frequency=1, max_frequency=500, window_type='hamming'):

    # following http://sam-koblenski.blogspot.com/2015/10/everyday-dsp-for-programmers-signal.html
    # our signal is sound, so its already centered at 0, so exponential windowed moving average of absolute signal
    envelope = moving_average(np.abs(signal), window_size)

    # perform bandpass filter to smooth
    normalized_min_frequency = min_frequency / sampling_rate
    normalized_max_frequency = max_frequency / sampling_rate
    bandpass = sig.firwin(2* window_size, [normalized_min_frequency, normalized_max_frequency],
                          pass_zero=False, window=window_type)
    envelope = sig.filtfilt(bandpass, 1, envelope)

    # compress envelope and change the range to [0, 1]
    # envelope = compress(envelope, compression_alpha)
    envelope = local_compress(envelope, 0.1, compression_alpha, window_size * 4)
    plt.plot(envelope)
    plt.show()
    return envelope

def get_note_amplitude_statistics(envelope, notes):
    """
    given an amplitude envelope and a list of notes' (start, end) return
    the mean and standard deviation of the amplitude in each range
    :param envelope: the amplitude envelope of a signal
    :param notes: the (start, end) tuple pairs describing notes
    :return: a list of means and a list of standard deviations
    """
    means = []
    stddevs = []
    for note in notes:
        note_window = envelope[note[0] : note[1]]
        means.append(np.mean(note_window))
        stddevs.append(np.std(note_window))
    return means, stddevs

def get_pitch(signal, sampling_rate, window_size=2048, window_increment=512, k=0.9):
    assert(window_size % window_increment == 0)
    frequencies = []
    for offset in range(0, window_size, window_increment):
        borders = np.arange(offset, signal.size, window_size)
        windows = np.split(signal, borders)
        for window in windows[1:]: # skip the first window because it's non-standard.
            frequencies.append(mpm.mpm(window, sampling_rate, k))
    frequencies = np.array(frequencies)
    return np.nanmean(frequencies)

def get_transcription(signal, sampling_rate, note_intervals, k=0.9):
    """
    Given a signal and list of note intervals (notes) outputs a list of Note objects
    :param signal:
    :param sampling_rate:
    :param note_intervals:
    :param k:
    :return: a list of note objects representing a transcription
    """
    envelope = get_hilbert_envelope(signal)
    transcription = []
    for note_interval in note_intervals:
        # get timing
        note_start_time = note_interval[0] / sampling_rate
        note_duration = (note_interval[1] - note_interval[0]) / sampling_rate
        # get amplitude statistics
        note_amplitude_window = envelope[note_interval[0] : note_interval[1]]
        amplitude = np.mean(note_amplitude_window)
        amplitude_vibrato = np.std(note_amplitude_window)
        # get frequency statistics
        note_signal_window = signal[note_interval[0] : note_interval[1]]
        frequency = get_pitch(note_signal_window, sampling_rate, k=k)
        midi = tools.frequency_to_midi(frequency)
        # record note
        note = tools.Note(midi, amplitude, note_start_time, note_duration, 0, amplitude_vibrato)
        transcription.append(note)
    return transcription

def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def compress(envelope, threshold, alpha):
    """
    compresses (decreases the dynamic range) of the envelope

    :param envelope: envelope of a signal
    :param alpha: higher values decrease the dynamic range, 1 does nothing
    :return:
    """
    # first normalize the values to the range [0, 1]
    maximum = np.max(envelope)
    if np.max(envelope) > threshold:
        # the compression
        envelope = envelope / max
        return 1 - (1 - envelope) ** alpha
    else:
        return np.zeros(envelope.shape)

def local_compress(envelope, threshold, alpha, window_size):
    """
    Like compress but it compresses by the local maxima within a neighborhood
    around a point (controlled by window_size)
    :param envelope:
    :param threshold: signal is scaled only if local maximum exceeds maximum * threshold
    :param alpha:
    :param window_size:
    :return:
    """
    # TODO: I'd like to use a sliding maximum here, but am too lazy to code it
    # first normalize the values to the range [0, 1]
    compressed_windows = []
    absolute_threshold = np.max(envelope) * threshold
    for window in np.array_split(envelope, window_size):
        local_max = np.max(window)
        if local_max >= absolute_threshold:
            compressed_windows.append( 1 - ( 1 - window / local_max) ** alpha )
        else:
            compressed_windows.append( np.zeros( window.shape ))
    return np.concatenate(compressed_windows)

def ewma(data, window):
    """
    taken from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    performs the Exponential Weighted Moving Average of a signal
    :param data: signal
    :param window: window size
    :return:
    """
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

# PITCH BASED METHODS

def moving_window_pitch_transcriber(signal, sampling_rate, window_size = 2048, window_increment = 512, k = 0.9):
    freq_array = []
    for start in range(0, signal.size, window_increment):
        # do pitch detection
        window = signal[start : start + window_size]
        freq_array.append(mpm.mpm(window, sampling_rate, k))
    # calculate times
    time = np.arange(0, signal.size / sampling_rate, window_increment / sampling_rate) + window_size / sampling_rate
    frequency = np.array(freq_array, dtype='float')
    midi = tools.frequency_to_midi(frequency)
    #plt.plot(time, midi, 'ro')
    #plt.show()
    return time, midi

# THRESHOLD METHODS

def threshold_transcriber(signal, sampling_rate, k=0.9):
    # Constants (for testing)
    THRESHOLD = 0.5
    SUSTAIN = 0.1 * sampling_rate # minimum note length is 0.1 seconds

    # get a list of notes
    note_intervals = []
    envelope = get_processed_envelope(signal, sampling_rate)
    above_threshold = envelope >= THRESHOLD
    streak_boundaries = np.zeros(envelope.shape, dtype=bool)
    streak_boundaries[1:] |= ~above_threshold[:-1] &  above_threshold[1:] # false then true
    streak_boundaries[1:] |=  above_threshold[:-1] & ~above_threshold[1:] # true then false
    streak_boundaries[0] = np.True_
    split_indices = np.where(streak_boundaries)[0]
    streaks = np.split(above_threshold, split_indices)
    streaks.pop()
    for start_index, streak in zip(split_indices, streaks[1:]): # first streak is always [] so [1:]
        if streak.size > SUSTAIN and streak[0] != 0:
            # there is a note
            end_index = start_index + streak.size
            note_intervals.append( (start_index, end_index) )

    # return transcription
    return get_transcription(signal, sampling_rate, note_intervals, k)

# TODO: totally does not work
def windowed_threshold_transcriber(signal, sampling_rate, window_size = 2048, window_increment = 512, k = 0.9):
    # Constants
    MINIMUM_THRESHOLD = 0.5
    START_THRESHOLD = 0.5
    START_SUSTAIN = window_size * 3 / 4
    END_THRESHOLD = 0.4
    END_SUSTAIN = window_size / 2
    # do note detection
    envelope = get_processed_envelope(signal, sampling_rate)
    notes = threshold_note_detection(envelope,
                                     MINIMUM_THRESHOLD, START_THRESHOLD, START_SUSTAIN,
                                     END_THRESHOLD, END_SUSTAIN,
                                     window_size, window_increment)
    amp_means, amp_stddevs = get_note_amplitude_statistics(envelope, notes)
    # do pitch detection for each note
    frequencies = []
    for note in notes:
        note_window = signal[note[0] : note[1]]
        frequency = mpm.mpm(signal, sampling_rate, k)
        frequencies.append(frequency)
    # create list of note objects
    transcription = []
    for note, amp_mean, amp_stddev, frequency in zip(notes, amp_means, amp_stddevs, frequencies):
        midi = tools.frequency_to_midi(frequency)
        note_start_time = note[0] / sampling_rate
        note_end_time = note[1] / sampling_rate
        note_object = tools.Note(midi,
                                 amp_mean,
                                 note_start_time,
                                 note_end_time - note_start_time,
                                 None,
                                 amp_stddev)
        transcription.append(note_object)
    return transcription

def threshold_note_detection(envelope,
                             minimum_threshold, start_threshold, start_sustain,
                             end_threshold, end_sustain,
                             window_size = 2048, window_increment = 512):
    """
    Given an amplitude envelope of a signal, calculate the intervals of notes in the signal.
    :param envelope: amplitude envelope of the signal
    :param minimum_volume: the minimum volume that the start of note threshold allows
    :param start_threshold: the relative threshold w.r.t the max volume of the window to start note
    :param start_sustain: the number of samples above the start-threshold required to start note
    :param end_threshold: the relative threshold w.r.t the current note volume to end the note
    :param end_sustain: the number of samples below the end-threshold required to end note
    :param window_size: the size of the window (in samples)
    :param window_increment: the number of samples the window increments by each iteration
    :return: a list of tuples (start, end) which are the sample indexes of the start and end of each note
    """
    start = None        # iteration variable
    notes = []          # the list of window-index ranges that correspond to notes
    note_on = False     # True iff we are in the middle of a note.
    note_start = None   # the index (in envelope) of the start of the current note
    note_volume = None  # the volume of the current note, none if there is no current note.
    for start in range(0, envelope.size, window_increment):
        window = envelope[start : start + window_size]
        if note_on:
            # we are in the middle of a note, so check if the note ends
            note_has_ended = threshold_note_end(window, note_volume, end_threshold, end_sustain)
            if (note_has_ended):
                # if the note ended, record the note in notes
                note_end = start # the start of the current window
                note = (note_start, note_end)
                notes.append(note)
        else:
            # no current note, so check if a note starts
            note_on, note_volume = threshold_note_start(window, minimum_threshold, start_threshold, start_sustain)
    # cap off the last note if necessary
    if note_on:
        note = (note_start, envelope.size)
        notes.append(note)
    return notes

def threshold_note_start(envelope, minimum_threshold, threshold, sustain):
    """
    Good for when you have a sharp attack and pauses between notes
    (e.g. when singing with only the syllable 'ba')
    :param envelope: amplitude function of the current window
    :param minimum_volume: the minimum (absolute) threshold allowed
    :param threshold: a threshold relative to the maximum amplitude of the envelope
    :param sustain: the number of samples the volume must be above the threshold to register a note
    :return: a boolean of whether there is a note, and the volume of that note start.
    """
    maximum = np.max(envelope)
    # we scale our absolute threshold based on the maximum
    volume = threshold * maximum
    if volume < minimum_threshold:
        volume = minimum_threshold
    above_threshold = (envelope >= volume)
    # find the boundaries of streaks of values above the threshold
    streak_boundaries = np.zeros(envelope.shape, dtype=bool)
    streak_boundaries[1:]  |= ~above_threshold[:-1] &  above_threshold[1:] # false then true
    streak_boundaries[1:]  |=  above_threshold[:-1] & ~above_threshold[1:] # true then false
    streak_boundaries[0] = np.True_
    split_indices = np.where(streak_boundaries)[0]
    streaks = np.split(envelope, split_indices)
    streaks.pop() # remove the last range b/c superfluous
    # if any 'true' streak is longer than our sustain requirement, return that a note has started
    for streak in streaks:
        if streak.size >= sustain and streak[0]:
            return (True, volume)
    return (False, None)

def threshold_note_end(envelope, note_volume, threshold, sustain):
    """
    Returns an end note when there is a sustained run below note_volume * threshold
    :param envelope: amplitude function of the current window
    :param note_volume: threshold volume for the note-on of the current note
    :param threshold: threshold w.r.t the note_volume for note-off report
    :param sustain: number of samples the threshold must be sustained
    :return:
    """
    below_threshold = (envelope <= note_volume * threshold)
    # find the boundaries of streaks of values above the threshold
    streak_boundaries = np.zeros(envelope.shape, dtype=bool)
    streak_boundaries[1:]  |= ~below_threshold[:-1] &  below_threshold[1:] # false then true
    streak_boundaries[1:]  |=  below_threshold[:-1] & ~below_threshold[1:] # true then false
    streak_boundaries[0] = np.True_
    split_indices = np.where(streak_boundaries)[0]
    streaks = np.split(envelope, split_indices)
    streaks.pop() # remove the last range b/c superfluous
    # if any streak is longer than our sustain requirement, return that a note has ended
    for streak in streaks:
        if streak.size >= sustain and streak[0]:
            return True
    return False
