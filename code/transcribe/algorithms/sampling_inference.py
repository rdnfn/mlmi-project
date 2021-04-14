def sampling_inference(audio, action_distribution, env, max_action_samples=10, proceed_thresh=-0.1):
    
    input_audio, input_len = get_input_representation(audio)
    
    for step in range(input_len):
        
        action_reward_list = []
        next_action = None
        
        for action_num in range(max_action_samples):
            action = sample_action(action_distribution)
            reward = env.evaluate_action(action)
            
            if reward < proceed_thresh:
                next_action = action
                break
            
            action_reward_list.append([action, reward])
            
        if not next_action:
            action = get_max_reward_action(action_reward_list)
        
        env.take_action_without_reward(action)
        
        
    env.get_current_transcription()
        
    return transcription
            
            
            
            
        
        
