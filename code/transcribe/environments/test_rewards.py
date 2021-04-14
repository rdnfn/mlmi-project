import pytest
import torch

import transcribe.environments.rewards

zeros_input = torch.zeros(1,100)
ones_input = torch.ones(1,100)

inputs_mel = [(zeros_input, zeros_input),
              (ones_input, ones_input),
              (ones_input, zeros_input),
              (ones_input*99, zeros_input),
              (ones_input*100, zeros_input)]

expected_euclidean = [0,0,-10,-990,-1000]
expected_novel_discrete = [1,1,1,0,-1]

def join_inputs_expected(inputs, expected):
    return list(zip(*list(map(list,zip(*inputs))), expected))


# creates tuple of (input_1, input_2, expected)
euclid_reward_input = join_inputs_expected(inputs_mel, expected_euclidean)
novel_discrete_reward_input = join_inputs_expected(inputs_mel, expected_novel_discrete)


class TestEuclidean:
    
    @pytest.mark.parametrize("true_input, pred_input, expected", euclid_reward_input)
    def test_euclidean_reward(self, true_input, pred_input,expected):
        reward = rewards.euclidean_reward(target_mel=true_input,
                                        pred_mel=pred_input) 
        assert reward == expected
        

        
class TestNovelDiscrete:
    
    @pytest.mark.parametrize("true_input, pred_input, expected", novel_discrete_reward_input)
    def test_euclidean_reward(self, true_input, pred_input,expected):
        reward = rewards.novel_discrete_reward(target_mel=true_input,
                                               pred_mel=pred_input,
                                               novel=True,
                                               neg_thresh=-1000, 
                                               pos_thresh=-100)
        assert reward == expected
    
    
    
    
    
