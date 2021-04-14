import sacred
ex = sacred.Experiment("monphonic_rl_transcription")
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

import datetime
import torch
import itertools
import gym

import transcribe.algorithms.reinforce as reinforce
import transcribe.environments.piano as environments
import transcribe.models.basic as models
import transcribe.data as data



@ex.config
def config():

    #env_name = 'CartPole-v0'
    env_name = 'MonoTranscription-v0'

    # general params
    rand_seed = 0
    verbose = 2

    # env params
    mdp_type = 'repeating'
    reward_settings = dict(
        REWARD_TYPE = "discrete",
        DIFF_REWARD = False,
        NEG_THRESH = -0.05,
        POS_THRESH = -0.05,
        ONSET_NEG_THRESH = -100,
        ONSET_POS_THRESH = -100,
        NEG_VALUE = -1,
        POS_VALUE = 1,
        NEUTRAL_VALUE = 0,
    )

    # alg params
    num_episodes = 40
    learning_rate = 0.0005
    batch_size=1
    gamma=0.5

    # policy net params
    hidden_size=128
    hidden_layer_num=3


    # data
    data_set = "SingleAudio"
    single_audio_path = "./tmp/entchen/entchen_v1_highoct.flac"

    # experiment data and logging
    logdir = 'runs/mono_transcr-{}'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

    # if called multiple times do not append new observer
    if not ex.observers:
        ex.observers.append(sacred.observers.FileStorageObserver(logdir))


@ex.automain
def run_experiment(rand_seed, verbose, env_name, mdp_type, num_episodes,
                   data_set, single_audio_path, reward_settings,
                   learning_rate, batch_size, gamma, hidden_size, hidden_layer_num):

    if env_name == "MonoTranscription-v0":
        env = environments.PianoTranscription(verbose=verbose, reward_settings=reward_settings,
                                              mdp_type=mdp_type)

        audio_dataset = data.SingleAudioDataset(single_audio_path)
        audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=1, shuffle=True, drop_last=True)
        data_batch = next(itertools.cycle(audio_loader))
        audio = data_batch['audio'][0]

        env.initialize_sequence(audio)

        print(type(audio))

        action_space_size = env.num_actions
        state_space_size = env.state_size
    else:
        env = gym.make(env_name)
        state_space_size = env.env.observation_space.shape[0]
        action_space_size = env.action_space.n

    policy_net = models.PolicyNet(state_space_size, action_space_size,
                                        hidden_size=hidden_size, hidden_layer_num=hidden_layer_num)
    reinforce.reinforce(policy_net, env, num_episodes=num_episodes,
                                   learning_rate=learning_rate,
                                   batch_size=batch_size, gamma=gamma ,verbose=verbose)
