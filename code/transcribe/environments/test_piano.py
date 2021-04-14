import pytest
import torch
import soundfile
import matplotlib.pyplot as plt
import numpy as np

import transcribe.environments.piano as envs

TEST_FOLDER = "./tmp/test_audio_samples/"

# inputs are tuples (midi, actions, expected_rewards)
test_reward_input = [([22,88,22],[22,88,22],[1,1,1]),
                     ([22,88,22],[88,88,88],[-1,0,-1]),
                     ([88,20],   [88,88],   [1,-1])]



# Helper functions
def create_audio_from_midi_list(midi_list, audio_folder = TEST_FOLDER):
    
    env = envs.PianoTranscription(tmp_directory=audio_folder, verbose=2)
    env.initialize_sequence()
    
    audio_file_name = "sample"
    
    for note in midi_list:
        env.take_action_without_reward(note)
        audio_file_name += "-{}".format(note)

    audio_file_name += ".flac"
    env.create_simulated_audio(audio_file_name=audio_file_name)
    
    return audio_folder + audio_file_name
    
def get_rewards_for_actions(action_list, audio_path,verbose=2, show_state_img=False):
    
    if audio_path is not None:
        audio, sr = soundfile.read(audio_path, dtype='int16')
        audio = torch.FloatTensor([audio]).div_(32768.0)
    else:
        audio = None
    
    env = envs.PianoTranscription(verbose=verbose)
    state = env.initialize_sequence(audio)
    rewards = []
    
    if show_state_img:
        state_img = []
    
    for i, midi_note in enumerate(action_list):
        if show_state_img:
            state_img.append(state)
            
            
        reward, next_state, status = env.take_action(torch.tensor([midi_note]))
        state = next_state
        info_string = "Action: {:3}, Reward: {:4}"
        rewards.append(reward.item())
        if verbose > 0:
            print(info_string.format(midi_note, reward.item()))
            
    if show_state_img:   
        plt.imshow(np.transpose(np.vstack(state_img)),aspect='auto')
        
    return rewards



# Tests
class TestPianoTranscription:
    
    @pytest.mark.parametrize("true_midi, actions, expected_rewards", test_reward_input)
    def test_reward_function(self, true_midi, actions, expected_rewards):
        audio_path = create_audio_from_midi_list(true_midi)
        rewards = get_rewards_for_actions(action_list=actions,
                                          audio_path=audio_path)
        
        assert rewards == expected_rewards
        