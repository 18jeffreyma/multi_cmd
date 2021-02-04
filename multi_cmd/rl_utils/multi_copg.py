
import numpy as np
import torch
from multi_cmd import optim, potentials
from utils import critic_update, get_advantage


# TODO(jjma): Casework for this?
DEFAULT_DTYPE = torch.float64
torch.set_default_dtype(DEFAULT_DTYPE)


# TODO(jjma): Refine this class structure based on feedback.
class MultiCoPG:
    # TODO(jjma): Horizon, gamma tuning?
    def __init__(
        env,
        policies,
        critics,
        batch_size=32,
        self_play=False,
        potential=potentials.squared_distance(1),
        device=torch.device('cpu')
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

        # Optimizers for policies and critics.
        self.policy_optim = multi_cmd.CMD_RL(
            [p.parameters() for p in self.policies], bregman=potential
        )
        self.critic_optim = [
            torch.optim.Adam(c.parameters(), lr=1e-3) for c in self.critics
        ]

        # Sampling parameters.
        self.batch_size = batch_size

        # Device to be used.
        self.device = device


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
                    obs_gpu = torch.tensor([obs[i]], device=self.device)
                    action = p(obs_gpu).sample().numpy()
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
        mat_states = torch.tensor(mat_states_t, device=self.device).transpose(0, 1)
        mat_actions = torch.tensor(mat_actions_t, device=self.device).transpose(0, 1)
        mat_rewards = torch.tensor(mat_rewards_t, device=self.device).transpose(0, 1)
        mat_done = torch.tensor(mat_done_t, dtype=torch.float, device=self.device).transpose(0, 1)

        return mat_states, mat_actions, mat_rewards, mat_done


    def step(mat_states, mat_actions, mat_rewards, mat_done):
        """
        Compute update step for policies and critics.
        """
        # Use critic function to get advantage.
        values, returns, advantages = [], [], []
        for i, q in enumerate(self.critics):
            val = q(mat_states[i]).detach()
            ret = get_advantage(0, mat_rewards[i], val, mat_done[i], device=self.device)
            advantage = returns - val

            values.append(val)
            returns.append(ret)
            advantages.append(advantage)

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
            lp = p(mat_states[i]).log_prob(mat_actions[i])
            log_probs.append(lp)

        # Get gradient objective, which is log probabilty times advantage.
        gradient_losses = []
        for lp, adv in zip(log_probs, advantages):
            gradient_losses.append(-(lp * adv).mean())

        # Compute summed log probabilities for hessian objectives.
        s_log_probs = [torch.zeros_like(lp) for lp in log_probs]
        for lp, slp, mask in zip(log_probs, s_log_probs, mat_done):
            for i in range(slp.size(0)):
                if i == 0:
                    slp[0] = 0.
                else:
                    slp[i] = torch.add(slp[i-1], lp[i-1]) * mat_done[i-1]

        # Compute hessian objectives.
        hessian_losses = []
        for i in range(len(log_probs)):
            accum = 0.
            for j in range(len(log_probs))
                if (i != j):
                    accum -= (log_probs[i] * log_probs[j] * advantages[i]).mean()

                    term1 = s_log_probs[i][1:] * log_probs[j][1:] * advantages[i][1:]
                    accum -= term1.sum() / (term1.size(0) - self.batch_size + 1)

                    term2 = log_probs[i][1:] * s_log_probs[j][1:] * advantages[i][1:]
                    accum -= term2.sum() / (term2.size(0) - self.batch_size + 1)

            hessian_losses.append(accum)

        # Update the policy parameters.
        self.policy_optim.zero_grad()
        self.policy_optim.step(gradient_losses, hessian_losses, cgd=True)



if __name__ == '__main__':
    # Import game utilities from marlenv package.
    import gym, envs
