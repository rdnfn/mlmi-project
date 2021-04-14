import transcribe.environments.piano as envs
from transcribe.constants.general import *


def create_audio_from_midi_list(midi_list, audio_folder = "./tmp/tmp_audio/", min_midi=MIN_MIDI):
    
    env = envs.PianoTranscription(tmp_directory=audio_folder, verbose=2, min_midi=MIN_MIDI)
    env.initialize_sequence()
    
    audio_file_name = "sample"
    
    for note in midi_list:
        env.take_action_without_reward(note)
        audio_file_name += "-{}".format(note + min_midi)

    audio_file_name = audio_file_name[:min(60,len(audio_file_name))] + ".flac"
    env.create_simulated_audio(audio_file_name=audio_file_name)
    
    return audio_folder + audio_file_name