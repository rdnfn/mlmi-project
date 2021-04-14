import torch
import numpy as np
from tqdm.notebook import tqdm

# import sys
# sys.path.append('./../')
# import environments.transcription
# import utils.rl


def reinforce(policy_net, environment, num_episodes, batch_size=32,
              gamma=0.99, learning_rate=0.01, verbose=1):
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    all_rewards = []
    
    batch_discounted_returns = []
    batch_discount_factors = []
    batch_actions = []
    batch_action_log_probs = []
    
    avg_reward=0
    
    for episode in tqdm(range(num_episodes)):
        states, actions, rewards, action_log_probs = run_episode(policy_net, environment, verbose=verbose)
        discounted_returns = get_discounted_returns(rewards, gamma)
        discount_factors = [gamma**i for i in range(len(states))]
        
        batch_actions.extend(actions)
        batch_action_log_probs.extend(action_log_probs)
        batch_discounted_returns.extend(discounted_returns)
        batch_discount_factors.extend(discount_factors)
        
        if (episode+1)%batch_size==0:
            batch_discounted_returns_tensor = torch.FloatTensor(batch_discounted_returns)
            batch_discount_factors_tensor = torch.FloatTensor(batch_discount_factors)
            batch_action_log_probs_tensor = torch.gather(torch.stack(batch_action_log_probs), 1, 
                                                         torch.LongTensor(batch_actions).view(-1,1)).squeeze()

            # \gamma^t * G\delta * ln(\pi(a_t, s_t, \theta)) for all t
            # Note other people take mean here but not the original algorithm
            loss = -(batch_discount_factors_tensor * batch_discounted_returns_tensor *\
                     batch_action_log_probs_tensor).sum()
            loss = loss/batch_size

            # Compute gradients and update theta
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            batch_discounted_returns = []
            batch_discount_factors = []
            batch_actions = []
            batch_action_log_probs = []
        
        episode_reward=np.sum(rewards)     
        all_rewards.append(episode_reward)
        avg_reward = np.mean(all_rewards[-100:])
        
        if verbose > 0 and (episode+1)%10==0:
            print("Episode {}, current average reward (last 100 eps): {}".format(episode + 1, avg_reward))
            
    return all_rewards
        
        

def run_episode(policy_net, env, verbose=0):
    state = env.reset()
    done = False
    
    states = []
    actions = []
    rewards = []
    action_log_probs_list = []
    
    
    while not done:
        
        try:
            step_position = env.position
        except:
            step_position = None
        
        action, action_probs = policy_net.sample_action(state)
        if action==None:
            print("No action could be sampled. Network likely broken")
            break
        
        next_state, reward, done, _ = env.step(action)
        
        action_log_probs = torch.log(action_probs)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        action_log_probs_list.append(action_log_probs)
        
        state = next_state
        
        if verbose > 0:
            print('TRAIN: action: {}, reward {}, done {}, step {}'.format(action, reward, done, step_position))
        if verbose > 1:
            print('TRAIN: action_probs:', action_probs)
            
    if verbose>1:
        print('TRAIN: finished episode')

    
    return states, actions, rewards, action_log_probs_list   

            
def get_discounted_returns(rewards, gamma):
    discounted_returns = []
    current_return = 0
    
    for step in reversed(range(len(rewards))):
        current_return = rewards[step] + gamma*current_return
        discounted_returns.append(current_return)
    
    return list(reversed(discounted_returns))
        
        
            
            





