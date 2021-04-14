"""Reward functions for RL environment.
"""

import torch


def euclidean_reward(target_mel, pred_mel, reward_min=None, reward_max=None, verbose=0):
    """Negative Euclidean norm of the difference between the two 
    flattened inputs. If reward_min or reward_max values are given, it caps
    the reward at appropriate levels.
    """

    target_mel = torch.flatten(target_mel)
    pred_mel = torch.flatten(pred_mel)
    diff = target_mel-pred_mel
    reward = -torch.norm(diff, p=2).view(1,1)

    if reward_min:
        reward = torch.max(torch.Tensor([reward, reward_min])).view(1,1)
    if reward_max:
        reward = torch.min(torch.Tensor([reward, reward_max])).view(1,1)
        
    if verbose > 1:
        print("REWARDS: Euclidean reward: {:.6f}".format(reward.item()))

    return reward


def discrete_reward(euc_reward, neg_thresh, pos_thresh, 
                    neg_value=-1, pos_value=1, neutral_value=0, verbose=0):
    """Discrete reward.
    
    By default returns -1 for euclidean reward values < neg_thresh,
    +1 for values above pos_thresh and 0 otherwise.
    """
    euc_reward = euc_reward.item()
    
    if euc_reward <= neg_thresh:
        reward = neg_value
    elif euc_reward >= pos_thresh:
        reward = pos_value
    else:
        reward = neutral_value
    
    reward = torch.Tensor([reward]).view(1,1)
    
    return reward


def onset_discrete_reward(euc_reward, neg_thresh, 
                          pos_thresh, onset, neg_value=-1, 
                          pos_value=1, neutral_value=0,
                          onset_neg_thresh=-100, onset_pos_thresh=-100,
                          verbose=0):
    """Novel discrete reward, discrete reward that encourages 
    new (novel) notes.
    
    An onset-sensitive version of discrete reward, is able to have different
    threshold levels for onset and non-onset actions. This can help with
    the higher noise/variability during onset events.
    """
    
    if onset:
        pos_thresh = onset_pos_thresh
        neg_thresh = onset_neg_thresh
        
    reward = discrete_reward(target_mel, pred_mel, neg_thresh, 
                              pos_thresh, neg_value, pos_value, 
                              neutral_value, verbose=verbose)
    
    return reward
        