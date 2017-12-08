import scipy.io.wavfile
import numpy as np
import mpm
import matplotlib.pyplot as plt
import tools
import note_detection as transcribe

wav_audio = 'audio/{}.wav'

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

def test_piano():
    print('piano test')
    sampling_rate, signal = get_mono_wav(wav_audio.format('A Tone (~440Hz)'))
    expected = 440
    actual = mpm.mpm(signal, sampling_rate)
    print('expected: {}\ngot: {}\n'.format(expected, actual))
    sampling_rate, signal = get_mono_wav(wav_audio.format('Middle C'))
    expected = 261.63
    actual = mpm.mpm(signal, sampling_rate)
    print('expected: {}\ngot: {}\n'.format(expected, actual))

def test_running_mpm():
    sampling_rate, signal = get_mono_wav(wav_audio.format('Simple Melody (singing, CGFGFEDCD)'))
    t, f, n = mpm.running_mpm(signal, sampling_rate, window_size=4096, window_increment=1024, k=0.9)
    tools.get_note_representation(t, f, n)
    t, f = tools.clean_data(t, f)
    m = tools.frequency_to_midi(f)
    print("\nsinging")
    plt.plot(t, m, 'bo')
    plt.show()
    """
    sampling_rate, signal = get_mono_wav(wav_audio.format('Simple Melody (piano, CGFGFEDCD)'))
    t, f = mpm.running_mpm(signal, sampling_rate, window_size=4096, window_increment=1024, k=0.9)
    tools.clean_data(t, f)
    m = tools.frequency_to_midi(f)
    print("\npiano")
    tools.get_note_representation(t, f)
    plt.plot(t, m, 'ro')
    plt.plot(t, n, 'ro')
    plt.show()
    """

def test_note_detection():
    sampling_rate, signal = get_mono_wav(wav_audio.format('Simple Melody (singing, CGFGFEDCD)'))
    transcription = transcribe.threshold_transcriber(signal, sampling_rate)
    tools.visualize_transcription(transcription, signal, sampling_rate)

if __name__ == '__main__':
    test_sine_wave(440)
    test_sine_wave(200)
    test_acf()
    test_sum_squares()
    #test_piano()
    #test_running_mpm()
    test_note_detection()
