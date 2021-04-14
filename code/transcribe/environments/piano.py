import torch
import torchaudio
import mir_eval.util
import os.path
import numpy as np
import soundfile
import warnings

import transcribe.utils.pianoteq       as pianoteq
import transcribe.environments.rewards as rewards
import transcribe.utils.utils          as utils
from transcribe.constants.general import *

import external.midi
import external.decoding

warnings.simplefilter('always')


class PianoTranscription:
    """ RL environment for monophonic piano music transcription.
    """
    
    def __init__(self,
                 mdp_type=MDP_TYPE,
                 num_keys=NUM_KEYS,
                 state_size=STATE_SIZE,
                 mel_sample_rate=MEL_SAMPLE_RATE,
                 mel_win_len=MEL_WIN_LENGTH,
                 mel_hop_len=MEL_HOP_LENGTH,
                 min_midi=MIN_MIDI,
                 mel_n_mels=MEL_N_MELS,
                 pianoteq_path=PIANOTEQ_PATH,
                 silent_start_length=SILENT_START_LENGTH,
                 reward_settings=REWARD_SETTINGS,
                 verbose=VERBOSE,
                 fixed_final_state=FIXED_FINAL_STATE,
                 tmp_directory="./tmp/piano_env_tmp/",
                 neg_reward_ends_episode=True):
        
        self.mdp_type = mdp_type
        self.num_actions = num_keys + 1
        self.num_keys = num_keys
        self.state_size = state_size
        self.position=None
        
        self.silent_start_length = silent_start_length
        self.fixed_final_state = fixed_final_state
        self.save_path = tmp_directory
        self.neg_reward_ends_episode = neg_reward_ends_episode
        
        self.mel_sample_rate = mel_sample_rate
        self.mel_hop_len = mel_hop_len
        self.min_midi = min_midi
        
        self.reward_settings = reward_settings
        self.verbose = verbose
        
        self.pianoteq_client = pianoteq.PianoteqClient(pianoteq_path)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=mel_sample_rate,
                                                                    win_length=mel_win_len,
                                                                    n_fft=mel_win_len,
                                                                    hop_length=mel_hop_len,
                                                                    n_mels=mel_n_mels)
        
    
    def initialize_sequence(self, audio=None):
        """Initialize new sequence with audio tensor."""
        
        if not torch.is_tensor(audio):
            warnings.warn("Audio given ('{}') is not a tensor, entering audio creation mode with silent audio (just zeros tensor) initialization.".format(audio))
            audio = torch.zeros(1,16000*60) # 1 min silent audio
        
        with torch.no_grad():
            self.target_mel = self.mel_spectrogram(audio)
        
        state = self.reset()
            
        return state
    
    
    def reset(self):
        self.set_state(0)
        self.reached_final_state = False
        self.addit_attempt_avail = 1
        self.action_list = []
        
        if not self.fixed_final_state:
            self.final_state = self.target_mel.size(2) - 1
        else:
            self.final_state = self.fixed_final_state
        
        self.previous_reward = 0
        
        if self.silent_start_length:
            self.transcription = dict(onsets=torch.zeros(self.silent_start_length, self.num_keys),
                                      frames=torch.zeros(self.silent_start_length, self.num_keys),
                                      velocity=torch.zeros(self.silent_start_length, self.num_keys))
        else:
            self.transcription = dict(onsets=torch.Tensor(),
                                      frames=torch.Tensor(),
                                      velocity=torch.Tensor())
            
        return self.current_state
        
    
    
    def set_state(self, position):
        """Set the self.current_state for the given position.
        """
        
        self.position = position
        self.current_state = self.target_mel[0,:,self.position]
      
    
    def step(self, action, compute_reward=True):
        """ Move one timestep forward.
        """
        if compute_reward:
            reward, euc_reward, disc_reward = self.take_action(action)
        else:
            self.add_action_to_transcription(action)
            reward = None
        
        next_position = self.position + 1
        if next_position <= self.final_state:
            self.set_state(next_position)
            done = False
        else:
            done = True
            self.reached_final_state = True
            self.current_state = None
            
        if compute_reward:
            # different MDP variants
            if self.mdp_type == 'stopping':
                if disc_reward < 0:
                    done = True
            elif self.mdp_type == 'repeating':
                if disc_reward < 0:
                    if self.addit_attempt_avail > 0:
                        self.step_back()
                        self.addit_attempt_avail -= 1
                    else:
                        done = True
                else:
                    self.addit_attempt_avail += 1  
                
        
        return self.current_state, reward, done, {}
    
    def step_back(self):
        self.remove_last_action_in_transcription()
        prev_position = self.position - 1
        self.set_state(prev_position)
        state = self.current_state
        
        return state
        
        
    def take_action(self, action):
        """Take an action in the environment.
        
        Parameters
        ---
        action: int, index of note to be transcribed, 
                range: 0<=action<=num_keys
                if action==num_keys: no note played
        """
        
        if self.verbose > 10:
            print("ENV: Start of taking action: {}, step: {}".format(action.item(), self.position))
            
        self.add_action_to_transcription(action)
        pred_audio = self.create_simulated_audio(warn_if_empty_midi=False)

        ### Creating predicted mel spetrogram
        with torch.no_grad():
            pred_mel = self.mel_spectrogram(pred_audio)
        
        ### Getting reward signal
        # by comparing similarity in neighbourhood around current state
        
        if self.action_list:
            previous_action = self.action_list[self.position-1]
        else:
            previous_action = None
        
        if previous_action == action:
            onset = False
        elif action == (self.num_actions - 1):
            onset = False
            print('ENV: no action taken and no onset')
        else:
            onset = True
        
        target_mel = self.target_mel[0,:,self.position]
        pred_mel = pred_mel[0,:,self.position]
        
        reward, euc_reward, disc_reward = self.get_reward(target_mel, pred_mel, onset)
        
        ### Updating environment state
        self.previous_reward = reward
        
        return reward, euc_reward, disc_reward
    
    
    def take_action_without_reward(self, action):
        next_state, reward, done, _ = self.step(action, compute_reward=False)
        
        return next_state, done
    
    
    def add_action_to_transcription(self, action):
        """Adding action to the internal transcription."""
        if self.position is None:
            raise Exception("Sequence in environment not initialized. Use PianoTranscription.initialize_sequence() to initialize with audio.")
        
        self.action_list.append(action)
    
        onsets = self.transcription['onsets']     # torch.FloatTensor, shape = [frames, bins]
        frames = self.transcription['frames']     # torch.FloatTensor, shape = [frames, bins]
        velocity = self.transcription['velocity'] # torch.FloatTensor, shape = [frames, bins]
        
        num_keys = self.num_keys
        
        ### Adding new action
        
        new_row = torch.zeros(1,88)
        if action < num_keys:
            action = self.min_midi + action
            new_row[:,action] = 1
        # else: no action taken
        
        onsets = torch.cat([onsets, new_row], dim=0)
        frames = torch.cat([frames, new_row], dim=0)
        velocity = torch.cat([velocity, new_row], dim=0)
        
        self.transcription['onsets'] = onsets
        self.transcription['frames'] = frames
        self.transcription['velocity'] = velocity
        
    def remove_last_action_in_transcription(self):
        del self.action_list[-1]
        self.transcription['onsets'] = self.transcription['onsets'][:-1]
        self.transcription['frames'] = self.transcription['frames'][:-1]
        self.transcription['velocity'] = self.transcription['velocity'][:-1]
    
    
    def create_simulated_audio(self, warn_if_empty_midi=True,
                               audio_file_name='transcription.flac',
                               midi_file_name='transcription.midi'):
        """Simulates piano audio from transcription using 
        pianoteq client. Goes from onsets, frames, velocity 
        arrays to MIDI to audio via pianoteq simulation and
        returns the simulated audio.
        """
        
        onsets = self.transcription['onsets']     # torch.FloatTensor, shape = [frames, bins]
        frames = self.transcription['frames']     # torch.FloatTensor, shape = [frames, bins]
        velocity = self.transcription['velocity'] # torch.FloatTensor, shape = [frames, bins]

        ### Creating MIDI
        # Format of values returned below:
        # pitches: np.ndarray with bin_indices
        # intervals: np.ndarray of rows containing (onset_index, offset_index)
        # velocities: np.nd_array of velocity values
        
        pitches, intervals, velocities = external.decoding.extract_notes(onsets, frames, velocity,
                                                                         onset_threshold=0.5, 
                                                                         frame_threshold=0.5)
        
        # Changing interval values from in indices to seconds
        seconds_per_hop = self.mel_hop_len/ self.mel_sample_rate
        intervals = (intervals * seconds_per_hop).reshape(-1, 2)
        
        # Changing pitch values from midi integers to hz
        pitches = np.array([mir_eval.util.midi_to_hz(midi + 21) for midi in pitches])

        os.makedirs(self.save_path, exist_ok=True)
        midi_path = os.path.join(self.save_path, midi_file_name)
        audio_path = os.path.join(self.save_path, audio_file_name)
        
        external.midi.save_midi(midi_path, pitches, intervals, velocities)
        
        num_frames = onsets.size(0)
        len_in_secs = num_frames * seconds_per_hop + 1
        len_audio_file = int(len_in_secs * self.mel_sample_rate)
        
        if not utils.check_midi_empty(midi_path):
            # creating predicted audio via pianoteq
            self.pianoteq_client.create_audio(midi_path=midi_path,
                                              out_path=audio_path,
                                              tmp_midi_path=midi_path)
            
            if audio_file_name[-4:] == ".mp3":
                warnings.warn("Created .mp3 file as simulated audio. Return None as simulated_audio unless flac format.")
                return None
            simulated_audio, sr = soundfile.read(audio_path, dtype='int16')
            simulated_audio = torch.ShortTensor([simulated_audio]).float().div_(32768.0)
            
            # making sure audio is long enough 
            # (e.g. if no key presses at end)
            len_diff = len_audio_file - simulated_audio.size(1)
            if len_diff > 0:
                simulated_audio = torch.cat([simulated_audio,torch.zeros(1,len_diff)],dim=1)
                if self.verbose > 0:
                    print("ENV: Extended audio! Size():", simulated_audio.size())
                
        else:
            # creating empty audio if midi empty
            simulated_audio = torch.zeros(1,len_audio_file)
            if warn_if_empty_midi:
                raise Exception("No notes played. Therefore cannot save any audio file.")
        
        return simulated_audio
    
    
    def get_reward(self, target_mel, pred_mel, onset):
        """Compute and return reward signal by comparing mel spectrograms.
        """
        
        reward_type = self.reward_settings['REWARD_TYPE']
        neg_thresh = self.reward_settings['NEG_THRESH']
        pos_thresh = self.reward_settings['POS_THRESH']
        onset_neg_thresh = self.reward_settings['ONSET_NEG_THRESH']
        onset_pos_thresh = self.reward_settings['ONSET_POS_THRESH']
        neg_value = self.reward_settings['NEG_VALUE']
        pos_value = self.reward_settings['POS_VALUE']
        neutral_value = self.reward_settings['NEUTRAL_VALUE']
        
        euc_reward = rewards.euclidean_reward(target_mel, pred_mel, 
                                              reward_min=None, reward_max=None,
                                              verbose=self.verbose)
        
        if reward_type in ["euclidean", "discrete"]:
            disc_reward = rewards.discrete_reward(euc_reward, neg_thresh, pos_thresh, 
                                                 neg_value, pos_value, neutral_value,
                                                 verbose=self.verbose)
        
        
        if reward_type == "euclidean":
            reward = euc_reward
        elif reward_type == "discrete":
            reward = disc_reward
        elif reward_type == "onset_discrete":
            disc_reward = rewards.onset_discrete_reward(euc_reward, neg_thresh, 
                                                   pos_thresh, onset, neg_value,
                                                   pos_value, neutral_value,
                                                   onset_neg_thresh, onset_pos_thresh,
                                                   verbose=self.verbose)
            reward = disc_reward
            
        else:
            raise ValueError("Unkown reward signal name given as reward_type: '{}'.".format(reward_type))
        
        if self.reward_settings['DIFF_REWARD']:
            reward = reward - self.previous_reward
        
        return reward, euc_reward, disc_reward
    
    def get_action_space_list(self):
        action_list = list(range(self.num_actions))
        return action_list
