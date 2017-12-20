import note_detection as transcribe
import tools
import sys

def wav_transcriber(wav_file_path, output_file_path):
    sampling_rate, signal = tools.get_mono_wav(wav_file_path)
    transcription = transcribe.threshold_transcriber(signal, sampling_rate)
    tools.visualize_transcription(transcription, signal, sampling_rate)
    tools.transcription_to_max_dict(output_file_path, transcription)

if __name__ == '__main__':
    parameters = sys.argv[1:]
    if len(parameters) < 2:
        print('Usage: \"wav_transcriber <wav_file_path> <output_file_path>\"')
    wav_file_path = parameters[0]
    output_file_path = parameters[1]
    wav_transcriber(wav_file_path, output_file_path)
