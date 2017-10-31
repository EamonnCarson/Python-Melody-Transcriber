import scipy.io.wavfile
import numpy as np
import mpm

def get_mono_wav(filename):
    # open the wav file
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

def test_sine_wave(freq):
    # generate a sine signal at specified frequency
    t = np.linspace(0, freq * 2 * np.pi, 14400)
    signal = np.sin(t)
    mpm_freq = mpm.mpm(signal, 14400)
    print('expected: {}\ngot: {}\n'.format(freq, mpm_freq))

def test_acf():
    print('acf test')
    test_signal = np.array([0, 1, 2, 3, 4])
    expected = [30, 20, 11, 4, 0]
    actual = mpm.acf(test_signal)
    print('expected: {}\ngot: {}\n'.format(expected, actual))

def test_sum_squares():
    print('sum squares (m\') test')
    test_signal = np.array([0, 1, 2, 3, 4])
    expected = [60, 44, 34, 26, 16]
    actual = mpm.sum_squares(test_signal)
    print('expected: {}\ngot: {}\n'.format(expected, actual))

if __name__ == '__main__':
    test_sine_wave(440)
    test_sine_wave(200)
    test_acf()
    test_sum_squares()
