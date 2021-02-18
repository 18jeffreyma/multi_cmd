
import time
import numpy as np
import torch
from multi_cmd.optim import cmd_utils, potentials
from utils import critic_update, get_advantage

# TODO(jjma): Casework for this?
DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)

# TODO(jjma): Refine this class structure based on feedback.
class MultiCoPG:
    # TODO(jjma): Horizon, gamma tuning?
    def __init__(
        self,
        env,
        policies,
        critics,
        batch_size=32,
        self_play=False,
        potential=potentials.squared_distance(1),
        critic_lr=1e-3,
        device=torch.device('cpu'),
        dtype=DEFAULT_DTYPE
    ):
        """
        :param env: OpenAI gym instance to train on
        :param policies: List of policies in same order as gym observations.
        :param batch_size: Number of trajectories to collect for each training step.
        :param potential: Bregman potential to use for optimizer.
        :param device: Device to compute on.

        Initialize training wrapper with any gym and policies.
        """
        # Store game environment.
        self.env = env

        # Training parameters.
        self.policies = policies
        self.critics = critics
        self.self_play = self_play
        
        # Sampling parameters.
        self.batch_size = batch_size

        # Device to be used.
        self.device = device
        self.dtype = dtype

        # Optimizers for policies and critics.
        self.policy_optim = cmd_utils.CMD_RL(
            [p.parameters() for p in self.policies], bregman=potential, device=self.device
        )

        # TODO(jjma): Implement self play in a cleaner way.
        if self.self_play:
            assert(len(self.critics) == 1)
            self.critic_optim = [
                torch.optim.Adam(self.critics[0].parameters(), lr=critic_lr)
            ]
        else:
            self.critic_optim = [
                torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics
            ]
            

    def sample(self, verbose=False):
        """
        :param verbose: Print debugging information if requested.

        Sample observations actions and states. Returns trajectory observations
        actions rewards in single list format (which seperates trajectories
        using done mask).
        """
        # We collect trajectories all into one list (using a done mask) for simplicity.
        mat_states_t, mat_actions_t, mat_rewards_t, mat_done_t = [], [], [], []

        # If verbose, show how long sampling takes.
        if verbose:
            batch_start_time = time.time()

        for j in range(self.batch_size):
            # Reset environment for each trajectory in batch.
            obs = env.reset()

            while (True):
                # Record state...
                mat_states_t.append(obs)

                # Since env is usually on CPU, send observation to GPU,
                # sample, then collect back to CPU.
                actions = []
                for i, p in enumerate(self.policies):
                    obs_gpu = torch.tensor([obs[i]], device=self.device, dtype=self.dtype)
                    action = p(obs_gpu).sample().cpu().numpy()
                    actions.append(action)

                # Advance environment one step forwards.
                obs, rewards, dones, _ = env.step(actions)

                # Record actions, rewards, and inverse done mask.
                mat_actions_t.append(actions)
                mat_rewards_t.append(rewards)
                mat_done_t.append(~dones)

                # Break once all players are done.
                if all(dones):
                    break

        # Print sampling time for debugging purposes.
        if verbose:
            print('sample took:', time.time() - batch_start_time)

        # Create data on GPU for later update step.
        mat_states = torch.tensor(mat_states_t, dtype=self.dtype, device=self.device).transpose(0, 1)
        mat_actions = torch.tensor(mat_actions_t, dtype=self.dtype, device=self.device).transpose(0, 1)
        mat_rewards = torch.tensor(mat_rewards_t, dtype=self.dtype, device=self.device).transpose(0, 1)
        mat_done = torch.tensor(mat_done_t, dtype=self.dtype, device=self.device).transpose(0, 1)

        return mat_states, mat_actions, mat_rewards, mat_done


    def step(self, mat_states, mat_actions, mat_rewards, mat_done, verbose=False):
        """
        Compute update step for policies and critics.
        """
        
        step_start_time = time.time()
        
        # Use critic function to get advantage.
        values, returns, advantages = [], [], []

        # TODO(jjma): Fix this when making self-play more robust.
        critics = self.critics
        if self.self_play:
            critics = [self.critics[0] for _ in range(len(self.policies))]

        for i, q in enumerate(critics):
            val = q(mat_states[i]).detach()
            ret = get_advantage(0, mat_rewards[i], val, mat_done[i], device=self.device)

            advantage = ret - val

            values.append(val)
            returns.append(ret)
            advantages.append(advantage)

        print('advantage calculated')   
        
        # Use sampled values to fit critic model.
        if self.self_play:
            # TODO(jjma): Currently only supports all symetric players.
            cat_states = torch.cat([mat_states[i] for i in range(len(mat_states))])
            cat_returns = torch.cat(returns)
            critic_update(cat_states, cat_returns, self.critics[0], self.critic_optim[0])
        else:
            for i, q in enumerate(self.critics):
                critic_update(mat_states[i], returns[i], q, self.critic_optim[i])

        # Calculate log probabilities.
        log_probs = []
        for i, p in enumerate(self.policies):
            # Our training wrapper assumes that the policy returns a distribution.
            lp = p(mat_states[i]).log_prob(torch.squeeze(mat_actions[i]))
            log_probs.append(lp)

        print('log probs calculated')   
            
        # Get gradient objective, which is log probabilty times advantage.
        gradient_losses = torch.zeros(len(self.policies), device=self.device)
        for i, (lp, adv) in enumerate(zip(log_probs, advantages)):
            gradient_losses[i] = (-(lp * adv).mean())

        print('gradient losses calculated')
            
        # Compute summed log probabilities for hessian objectives.
        s_log_probs = [torch.zeros_like(lp) for lp in log_probs]
        for lp, slp, mask in zip(log_probs, s_log_probs, mat_done):
            for i in range(slp.size(0)):
                if i == 0:
                    slp[0] = 0.
                else:
                    slp[i] = torch.add(slp[i-1], lp[i-1]) * mask[i-1]

        print('summed log probs calculated')  
                    
        # Compute hessian objectives.
        hessian_losses = torch.zeros(len(self.policies), device=self.device)
        for i in range(len(log_probs)):
            for j in range(len(log_probs)):
                if (i != j):
                    hessian_losses[i] -= (log_probs[i] * log_probs[j] * advantages[i]).mean()

                    term1 = s_log_probs[i][1:] * log_probs[j][1:] * advantages[i][1:]
                    hessian_losses[i] -= term1.sum() / (term1.size(0) - self.batch_size + 1)

                    term2 = log_probs[i][1:] * s_log_probs[j][1:] * advantages[i][1:]
                    hessian_losses[i] -= term2.sum() / (term2.size(0) - self.batch_size + 1)
            
        print('hessian losses calculated')  
            
        # Update the policy parameters.
        self.policy_optim.zero_grad()
        self.policy_optim.step(gradient_losses, hessian_losses, cgd=True)
        
        # Print sampling time for debugging purposes.
        if verbose:
            print('step took:', time.time() - step_start_time)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    
    # Import game utilities from marlenv package.
    import gym, envs, sys
    from network import policy, critic

    # Initialize game environment.
    env = gym.make('python_4p-v1')
    device = torch.device('cuda:1')
    dtype = torch.float32

    p1 = policy().to(device).type(dtype)
    # p1 = policy()
    policies = [p1 for _ in range(4)]
    q = critic().to(device).type(dtype)
    # q = critic()

   
    train_wrap = MultiCoPG(
        env,
        policies,
        [q],
        batch_size=1,
        self_play=True,
        potential=potentials.squared_distance(1000),
        critic_lr=1e-3,
        device=device
    )
    states, actions, rewards, done = train_wrap.sample(verbose=True)
 
    train_wrap.step(states, actions, rewards, done)
