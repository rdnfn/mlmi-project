import gym
import numpy as np

import transcribe.algorithms.reinforce as  reinforce
import transcribe.models.basic as models

env = gym.make('CartPole-v0')

state_space_size = env.env.observation_space.shape[0]
action_space_size = env.action_space.n

policy_net = models.PolicyNet(state_space_size, action_space_size, hidden_size=32, take_log_state=False)
reinforce.reinforce(policy_net, env, num_episodes=4000, learning_rate=0.005, batch_size=8, gamma=0.999, verbose=0)