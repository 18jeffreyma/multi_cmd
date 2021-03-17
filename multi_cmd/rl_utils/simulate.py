
import torch


if __name__ == '__main__':
    # Import game utilities from marlenv package.
    import gym, envs, sys
    from network import policy, critic

    # Initialize game environment.
    env = gym.make('python_4p-v1')

    p1 = policy()
    p1.load_state_dict(torch.load("model_checkpoints/actor1_9000.pth", map_location=torch.device('cpu')))
    # p1 = policy()
    policies = [p1 for _ in range(4)]
     
    
    obs = env.reset()
    dones = [False for _ in range(len(policies))]

    while (True):
        env.render()
        # Record state...

        # Since env is usually on CPU, send observation to GPU,
        # sample, then collect back to CPU.
        actions = []
        for i in range(len(policies)):
            obs_gpu = torch.tensor([obs[i]], dtype=torch.float32)
            dist = policies[i](obs_gpu)
            print(dist.probs)
            action = dist.sample().numpy()
            # TODO(jjma): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
            if action.ndim == 1 and action.size == 1:
                action = action[0]

            actions.append(action)

        # Advance environment one step forwards.
        obs, rewards, dones, _ = env.step(actions)

        # Break once all players are done.
        if all(dones):
            break
    
    env.close()
    

               
    
    
