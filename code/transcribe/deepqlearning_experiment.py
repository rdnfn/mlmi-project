from transcribe.algorithms.deep_q_learning import ex

for _ in range(10):
    r = ex.run(config_updates={'reward_type': 'euclidean'})
    #r = ex.run(config_updates={'reward_type': 'discrete', 'modified_exploration':False})