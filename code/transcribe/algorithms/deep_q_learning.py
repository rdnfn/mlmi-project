# Deep Q-learning [1] for music transcription with experience
# replay. Code strongly influenced and partly directly adopted
# from [2] (especially optimize_model() is very close).
#
# [1] Mnih, Volodymyr, et al. "Playing atari with deep
# reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
# [2] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import itertools
from collections import namedtuple
import os.path
import soundfile
import datetime

import transcribe.models.basic as models
import transcribe.utils.utils as utils
import transcribe.environments.piano as environments
import transcribe.data as data
from transcribe.constants.general import *

import external.dataset

import sacred
ex = sacred.Experiment("deep_q_learning")

from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'

@ex.config
def config():

    # general params
    rand_seed = 0
    verbose = 2

    # env params
    mdp_type = 'repeating'

    # q learning params
    num_episodes = 40
    max_iterations = 300
    gamma = 0.5
    learning_rate = 0.001
    memory_capacity = 1000
    batch_size = 16
    target_update_frequency = 30
    modified_exploration = True
    reward_type = "discrete"
    reward_settings = dict(
        REWARD_TYPE = reward_type,
        DIFF_REWARD = False,
        NEG_THRESH = -0.05,
        POS_THRESH = -0.05,
        ONSET_NEG_THRESH = -100,
        ONSET_POS_THRESH = -100,
        NEG_VALUE = -1,
        POS_VALUE = 1,
        NEUTRAL_VALUE = 0,
    )

    # policy params
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay_rate = 100
    no_note_epsilon = 0.0

    # data
    dataset = "SingleAudio"
    single_audio_path = "./tmp/entchen/entchen_v1_highoct.flac"

    # experiment data and logging
    logdir = 'runs/dql-{}'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S'))

    # if called multiple times do not append new observer
    if not ex.observers:

        ex.observers.append(sacred.observers.FileStorageObserver(logdir))
        rundir = logdir + "/1/"
    else:
        rundir = ex.observers[0].dir

    print(rundir)
    model_save_path = None
    model_save_frequency = 1
    eps_log_file_folder = None



@ex.capture
def optimize_model(optimizer, policy_net, target_net, memory, batch_size, gamma, verbose=1):
    if batch_size > len(memory.memory):
        return

    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Create non-final-state mask
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                  dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).view(-1,STATE_SIZE)

    state_batch = torch.cat(batch.state).view(batch_size,-1)
    action_batch = torch.cat(batch.action).view(batch_size,-1)
    reward_batch = torch.cat(batch.reward).view(batch_size)

    # Compute Q(s, a)
    full_state_action_values = policy_net(state_batch)
    state_action_values = full_state_action_values.gather(1, action_batch)


    # Compute max_{a'}(Q(s',a')) or set it equal to 0 for final states
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute r + \gamma * max_{a'}(Q(s',a')) for the given s'
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    if verbose > 10:
        print("OPTIM: state_batch", state_batch)
        print("OPTIM: state_action_values", state_action_values)
        print("OPTIM: full_state_action_values", full_state_action_values)
        print("OPTIM: next_state_values", next_state_values)

    # Compute smooth L1 loss (also known as Huber loss)
    # For details see https://pytorch.org/docs/master/generated/torch.nn.SmoothL1Loss.html?highlight=huber
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Run optimizer on gradients with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


@ex.automain
def train_model(num_episodes,
                max_iterations,
                mdp_type,
                gamma,
                learning_rate,
                batch_size,
                rand_seed,
                memory_capacity,
                single_audio_path,
                dataset,
                epsilon_start,
                epsilon_end,
                epsilon_decay_rate,
                no_note_epsilon,
                target_update_frequency,
                model_save_path,
                model_save_frequency,
                verbose,
                modified_exploration,
                eps_log_file_folder,
                reward_settings,
                maestro_path=MAESTRO_PATH,
                maestro_groups=None,
                sequence_length=SEQUENCE_LENGTH):

    random.seed(rand_seed)

    # Initialisations
    if dataset=="MAESTRO":
        audio_dataset = external.dataset.MAESTRO(path=maestro_path, groups=maestro_groups, sequence_length=sequence_length)
    elif dataset=="SingleAudio":
        audio_dataset = data.SingleAudioDataset(single_audio_path)

    audio_loader = torch.utils.data.DataLoader(audio_dataset, batch_size=1, shuffle=True, drop_last=True)

    env = environments.PianoTranscription(verbose=verbose, reward_settings=reward_settings, mdp_type=mdp_type)

    num_actions = env.num_actions
    state_size = env.state_size

    memory = utils.ReplayMemory(max_capacity=memory_capacity)
    policy_net = models.DeepQNetwork(input_size=state_size, output_size=num_actions)
    target_net = models.DeepQNetwork(input_size=state_size, output_size=num_actions)

    # Setting policy and target networks equal
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    scheduler = utils.Scheduler(epsilon_start=epsilon_start,
                                epsilon_end=epsilon_end,
                                epsilon_decay_rate=epsilon_decay_rate,
                                target_update_frequency=target_update_frequency,
                                model_save_frequency=model_save_frequency)

    # Counts and lists for logging purposes
    total_steps_done = 0
    eps_steps_list = []
    final_pos_list = []

    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)

    loop = tqdm(range(num_episodes), desc="Episodes")
    for episode, batch in zip(loop, itertools.cycle(audio_loader)):

        eps_steps_done = 0

        if verbose > 0:
            print("TRAIN: New episode started.")

        audio_input = batch['audio'][0]
        if verbose > 10:
            print("TRAIN: Audio input:", audio_input)
        state = env.initialize_sequence(audio_input)
        exploration_mode = False

        for time in range(max_iterations):

            step_position = env.position

            if modified_exploration:
                # Modified version of DQL
                if exploration_mode:
                    if current_attempt_num >= len(random_action_list):
                        # If we tried all possible actions but still no luck,
                        # we try going back to the previous state
                        state = env.step_back()
                        step_position = env.position
                        random_action_list = env.get_action_space_list()
                        random.shuffle(random_action_list)
                        current_attempt_num = 0
                        if verbose > 0:
                            print("TRAIN: Stepping back, because stuck.")

                    epsilon = 1
                    action = random_action_list[current_attempt_num]
                    action = torch.LongTensor([[action]])
                    action_type = "exrand"
                    current_attempt_num += 1
                else:
                    # Greedy
                    epsilon = 0
                    action, action_type = utils.select_action(policy_net, state,
                                                              epsilon, no_note_epsilon,
                                                              num_actions)
            else:
                # Vanilla version of DQL, epsilon greedy policy
                epsilon = scheduler.get_epsilon(total_steps_done)
                action, action_type = utils.select_action(policy_net, state,
                                                          epsilon, no_note_epsilon,
                                                          num_actions)

            next_state, reward, done, _ = env.step(action)
            memory.add(state, action, reward, next_state)

            optimize_model(optimizer, policy_net, target_net,
                           memory, batch_size, gamma, verbose)

            if verbose > 0:
                info_string = "TRAIN: Action: {:3}, Type: {:5}, Reward: {:4}, Episode: {:5}, Epsilon: {:.3f}, Total_Steps: {:6}, Ep_step: {:2}"
                print(info_string.format(action.item(),action_type, reward.item(), episode, epsilon, total_steps_done, step_position))

            if scheduler.should_update_targets(total_steps_done):
                target_net.load_state_dict(policy_net.state_dict())
                if verbose > 0:
                    print("TRAIN: Weights updated.")

            total_steps_done += 1
            eps_steps_done += 1


            if modified_exploration:
                # Modified version of DQL
                if done and env.reached_final_state:
                    # We didn't do a mistake, just end of transcription
                    # and therefore episode
                    break
                elif torch.all(torch.eq(state, next_state)) and not exploration_mode:
                    # Environment stopped but be haven't reached final state yet
                    # -> enter exploration mode
                    exploration_mode = True
                    random_action_list = env.get_action_space_list()
                    random.shuffle(random_action_list)
                    current_attempt_num = 0
                if exploration_mode and not torch.all(torch.eq(state, next_state)):
                        # We managed to get it right
                        # -> exit exploration mode and go back to greedy policy
                        exploration_mode = False

            state = next_state

            if done:
                break


        eps_steps_list.append(eps_steps_done)
        final_pos_list.append(env.position)

        ex.log_scalar("episode_step_num", eps_steps_done, episode)
        ex.log_scalar("episode_final_position", env.position, episode)

        if eps_log_file_folder:
            with open(eps_log_file_folder + "dql_log_episode_step_num.txt", "w") as f:
                for s in eps_steps_list:
                    f.write(str(s) +"\n")

            with open(eps_log_file_folder + "dql_log_final_step_pos.txt", "w") as f:
                for s in final_pos_list:
                    f.write(str(s) +"\n")

        if model_save_path:
            if scheduler.should_save_model(episode):
                curr_model_save_path = model_save_path + "model-ep{0:04d}.pt".format(episode)
                torch.save(policy_net.state_dict(), curr_model_save_path)




def transcribe(audio_path, model_path, transcr_folder="./tmp/piano_env_tmp/",
               transcr_filename="transcription.flac", verbose=VERBOSE):
    """Transcribing audio using trained model.
    """

    audio, sr = soundfile.read(audio_path, dtype='int16')
    audio = torch.FloatTensor([audio]).div_(32768.0)

    q_net = models.DeepQNetwork()
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()

    env = environments.PianoTranscription(verbose=verbose, tmp_directory=transcr_folder)
    state = env.initialize_sequence(audio)

    done = False
    step = 0

    while not done:
        with torch.no_grad():
            action = q_net(state).argmax().view(1,1)

        if verbose > 0:
            print("Step: {}, Action: {}".format(step, action))

        # Add actions without computing reward
        # (No simulation necessary after each action)
        state, done = env.take_action_without_reward(action)
        step += 1

    env.create_simulated_audio(audio_file_name=transcr_filename)



