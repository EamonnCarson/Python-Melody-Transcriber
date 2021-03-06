import numpy as np
import note_detection

def running_mpm(signal, sampling_rate, window_size = 2048, window_increment = 512, k = 0.9):
    freq_array = []
    note_active = []
    active_volume = None
    breakpoint = -1;
    envelope = note_detection.get_envelope(signal)
    for start in range(0, signal.size, window_increment):
        # do pitch detection
        window = signal[start : start + window_size]
        freq_array.append(mpm(window, sampling_rate, k))
        # do note detection
        envelope_window = envelope[start : start + window_size]
        if (active_volume != None):
            if (note_detection.threshold_note_end(envelope_window, active_volume, 0.3, 0.05 * sampling_rate)):
                # the note ends
                active_volume = None
                note_active.append(False)
            else:
                # the note continues
                note_active.append(True)
        else:
            note_on, volume = note_detection.threshold_note_start(envelope_window, 0.7, 0.05 * sampling_rate)
            if (note_on):
                # the note starts
                active_volume = volume
                note_active.append(True)
            else:
                # the note continues to be off
                note_active.append(False)
    # calculate times
    time = np.arange(0, signal.size / sampling_rate, window_increment / sampling_rate) + window_size / sampling_rate
    frequency = np.array(freq_array, dtype='float')
    note_active = np.array(note_active, dtype=bool)
    # rudimentary: TODO fix this
    note_starts = np.argwhere((~note_active[:-1] &  note_active[1:])) + 1
    note_ends   = np.argwhere(( note_active[:-1] & ~note_active[1:])) + 1
    if (note_starts.size != note_ends.size):
        assert(False)
    note_bounds = zip(note_starts, note_ends)
    return (time, frequency, note_bounds)

def mpm(signal, sampling_rate, k = 0.9):
    """
    Runs the MPM pitch-finding algorithm developed by Philip McLeod, Geoff Wyvill
    at the university of Ortago.
    :param signal: the window of signal that MPM runs on
    :param sampling_rate: the sampling rate of the signal
    :param k: the sensitivity of the max analysis (see McLeod §5)
    :return: pitch (MIDI), clarity estimate
    """
    nsdf_signal = nsdf(signal)
    intervals = find_positive_intervals(nsdf_signal)
    key_maxima = find_key_maxima(nsdf_signal, intervals)
    frequency = find_pitch(key_maxima, sampling_rate, k)
    return frequency
    # TODO: calculate clarity

### Section: Normalized Square Difference Calculation

def nsdf(signal):
    """
    Calculates the Normalized Square Difference Function of the input signal where,
    if we let $d'$ be the SDF then 
    $$ d'(t) = \sum_{j = 0}^{W - \tau - 1} (x_j - x_{j + \tau})^2 $$
    where $W$ is our window size, $\tau$ is our offset.
    :param signal: the signal within the window we are calculating the SDF in.
    :return: an array of SDF calculations for all offsets $\tau$
    """
    r_prime = acf(signal)
    m_prime = sum_squares(signal)
    n_prime = (2 * r_prime) / m_prime
    # TODO: deal with 0's in m_prime
    return n_prime

def acf(signal):
    """
    calculates the type II auto-correlation function (ACF) $r'$ such that
    $$ r' = \sum_{j = t}^{W - \tau - 1} x_j x_{j + \tau} $$
    where $W$ is the window-size / signal length, $\tau$ is the offset
    :param signal: the signal within the window we are calculating the acf in.
    :return: an array of ACF calculations for all offsets $\tau$.
    """
    window_length = signal.size
    autocorrelation = np.correlate(signal, signal, mode='full')
    autocorrelation = autocorrelation[-window_length:] # cut off part before valid intersection
    return autocorrelation
    # TODO test if the above works
    # passes basic test

def sum_squares(signal):
    """
    calculates the sum of squares $m'$ such that
    $$ m' = \sum_{j = t}^{W - \tau - 1} x_j^2 + x_{j + \tau}^2 $$
    where $W$ is the window-size / signal length, $\tau$ is the offset
    :param signal: the signal within the window we are calculating the sum of squares in.
    :return: an array of sum of squares calculations for all offsets $\tau$.

    The code below calculates based on the fact that
    m'(\tau) = sum_{j = 0}^{W - \tau} (signal[j]^2 + signal[W - j]^2)
    """
    squared_signal = np.square(signal)
    left_cum_squared_sum = np.cumsum(squared_signal)
    right_cum_squared_sum = np.cumsum(np.flipud(squared_signal))
    sum_squares = left_cum_squared_sum + right_cum_squared_sum
    return np.flipud(sum_squares)
    # TODO test if the above works
    # passes basic test

### Section: Peak Finding Routine

def find_positive_intervals(signal):
    """
    Given a signal, find all intervals where the signal is positive.
    :param signal: the signal within the window that we find the positive intervals of
    :return: a list of tuples (a,b) representing the intervals (a,b]
    """
    intervals = []
    # explanation: if signal changes between positive and non-positive, then
    # boundaries is non-zero on the first index after the change.
    # specifically, boundaries[i] = 1 if non-positive -> positive between i-1 and i
    # and boundaries[i] = -1 if positive -> non-positive between i-1 and i.
    positive = (signal > 0).astype(int)
    boundaries = positive[1:] - positive[:-1]
    # plus one to keep aligned with original array
    neg_slope_zeros = (np.argwhere(boundaries == -1) + 1).ravel()
    pos_slope_zeros = (np.argwhere(boundaries == 1) + 1).ravel()
    # special cases
    if len(pos_slope_zeros) <= 1 or len(neg_slope_zeros) <= 1:
        # too little data, lets abort
        return intervals
    if pos_slope_zeros[0] > neg_slope_zeros[0]:
        # the first positive slope was before signal began, so skip first neg slope.
        extra_interval = None
        neg_slope_zeros = neg_slope_zeros[1:]
    if pos_slope_zeros[-1] > neg_slope_zeros[-1]:
        # the last negative slope was after the signal ended, so make an extra interval
        extra_interval = (pos_slope_zeros[-1], signal.size)
        pos_slope_zeros = pos_slope_zeros[:-1]
    # at this point pos and neg slope zeros should be the same size
    assert(neg_slope_zeros.size == pos_slope_zeros.size)
    # zip into intervals
    intervals = list(zip(pos_slope_zeros.ravel(), neg_slope_zeros.ravel()))
    if (extra_interval is not None):
        intervals.append(extra_interval)
    return intervals

def find_key_maxima(signal, positive_intervals):
    """
    Given a signal and set of zero crossings,
    let $L$ be the set of intervals on which the signal is positive
    define $M_{key}$ as $\{ max(I) : I \in L \}$
    we return $M_{key}$
    :param signal: the signal within the window that we find the key maxima of.
    :param zero_crossings: a list of tuples (<zero-position>, <Slope>)
    :return: a list of tuples (index, value) which mark the index and the value of each key maximum.
    """
    key_maxima_indices = [a + np.argmax(signal[a:b]) for a,b in positive_intervals]
    key_maxima = signal[key_maxima_indices]
    return np.vstack((key_maxima, key_maxima_indices))

def find_pitch(key_maxima, sampling_rate, k):
    """
    Given a list of key maximums, return a frequency that is (probably) the pitch
    of the signal.
    :param key_maxima: a list of tuples (index, value) corresponding to key maxima
    :return: a frequency (in Hz)
    """
    MAXIMA_VALUES = 0
    MAXIMA_INDICES = 1
    if key_maxima.size == 0:
        return np.nan
    argmax = np.argmax(key_maxima[MAXIMA_VALUES])
    maximum = key_maxima[MAXIMA_VALUES, argmax]
    maximum_index = key_maxima[MAXIMA_INDICES, argmax]
    candidates = key_maxima[:, argmax + 1:] # all after the maximum
    valid_candidates = np.where(candidates[MAXIMA_VALUES] > maximum * k)
    if valid_candidates[0].size > 0:
        first_valid_index = valid_candidates[0][0]
        matching_index = candidates[MAXIMA_INDICES, first_valid_index]
        delta_index = matching_index - maximum_index
        frequency = sampling_rate / float(delta_index)
        return frequency
    else:
        # if all else fails, return nan
        return np.nan