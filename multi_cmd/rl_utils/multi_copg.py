
import os
import copy
import time
import numpy as np
import torch
from multi_cmd.optim import cmd_utils, potentials
from multi_cmd.optim import gda_utils

# TODO(jjma): Casework for this?
DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)

def critic_update(state_mat, return_mat, q, optim_q):
    val_loc = q(state_mat)

    critic_loss = (return_mat - val_loc).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()

    critic_loss = critic_loss.detach()


# TODO(jjma): Revisit this?
def get_advantage(
    next_value, reward_mat, value_mat, masks,
    gamma=0.99, tau=0.95, device=torch.device('cpu')
):
    insert_tensor = torch.tensor([[float(next_value)]], device=device)
    value_mat = torch.cat([value_mat, insert_tensor])
    gae = 0
    returns = []

    for i in reversed(range(len(reward_mat))):
        delta = reward_mat[i] + gamma * value_mat[i+1] * masks[i] - value_mat[i]
        gae = delta + gamma * tau * masks[i] * gae
        returns.append(gae + value_mat[i])

    # Reverse ordering.
    returns.reverse()

    vals = torch.cat(returns).reshape(-1, 1)
    return vals


class TrainingWrapper:
    def __init__(
        self,
        env,
        policies,
        critics,
        batch_size=32,
        critic_lr=1e-3,
        tol=1e-3,
        device=torch.device('cpu'),
        dtype=DEFAULT_DTYPE,
        self_play=False,
        gamma=0.99,
        tau=0.95,
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

        # Sampling parameters.
        self.batch_size = batch_size

        # Device to be used.
        self.device = device
        self.dtype = dtype

        # TODO(jjma): Implement self play in a cleaner way.
        self.self_play = self_play
        if self.self_play:
            assert(len(self.critics) == 1)
            self.critic_optim = [
                torch.optim.Adam(self.critics[0].parameters(), lr=critic_lr)
            ]
        else:
            self.critic_optim = [
                torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics
            ]

        # GAE estimation work.
        self.gamma = gamma
        self.tau = tau

    def sample(self, verbose=False):
        """
        :param verbose: Print debugging information if requested.
        Sample observations actions and states. Returns trajectory observations
        actions rewards in single list format (which seperates trajectories
        using done mask).
        """
        # We collect trajectories all into one list (using a done mask) for simplicity.
        num_agents = len(self.policies)
        mat_states_t = []
        mat_actions_t = []
        mat_action_mask_t = []
        mat_rewards_t = []
        mat_done_t = []

        # If verbose, show how long sampling takes.
        if verbose:
            batch_start_time = time.time()

        for j in range(self.batch_size):
            # Reset environment for each trajectory in batch.
            obs = self.env.reset()
            dones = [False for _ in range(num_agents)]

            while (True):
                # Record state...
                mat_states_t.append(obs)
                mat_action_mask_t.append([1. - int(elem) for elem in dones])

                # Since env is usually on CPU, send observation to GPU,
                # sample, then collect back to CPU.
                actions = []
                for i in range(num_agents):
                    policy = self.policies[i]
                    obs_gpu = torch.tensor([obs[i]], device=self.device, dtype=self.dtype)
                    dist = policy(obs_gpu)
                    print('loc:', dist.loc)
                    print('scale:', dist.scale)

                    action = dist.sample().cpu().numpy()
                    # TODO(jjma): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
                    if action.ndim == 1 and action.size == 1:
                        action = action[0]

                    actions.append(action)

                # Advance environment one step forwards.
                obs, rewards, dones, _ = self.env.step(actions)

                # Record actions, rewards, and inverse done mask.
                mat_actions_t.append(actions)
                mat_rewards_t.append(rewards)
                mat_done_t.append([1. - int(elem) for elem in dones])

                # Break once all players are done.
                if all(dones):
                    break

        # Print sampling time for debugging purposes.
        if verbose:
            print('sample took:', time.time() - batch_start_time)

        # Create data on GPU for later update step.
        mat_states = torch.tensor(mat_states_t, dtype=self.dtype, device=self.device).transpose(0, 1)
        mat_actions = torch.tensor(mat_actions_t, dtype=self.dtype, device=self.device).transpose(0, 1)
        mat_action_mask = torch.tensor(mat_action_mask_t, dtype=self.dtype, device=self.device).transpose(0, 1)
        mat_rewards = torch.tensor(mat_rewards_t, dtype=self.dtype, device=self.device).transpose(0, 1)
        mat_done = torch.tensor(mat_done_t, dtype=self.dtype, device=self.device).transpose(0, 1)

        return mat_states, mat_actions, mat_action_mask, mat_rewards, mat_done


# TODO(jjma): Refine this class structure based on feedback.
class MultiSimGD(TrainingWrapper):
    # TODO(jjma): Horizon, gamma tuning?
    def __init__(
        self,
        env,
        policies,
        critics,
        batch_size=32,
        critic_lr=1e-3,
        tol=1e-3,
        device=torch.device('cpu'),
        dtype=DEFAULT_DTYPE,
        self_play=False,
        policy_lr=0.002,
        gamma=0.99,
        tau=0.95,
    ):
        """
        :param env: OpenAI gym instance to train on
        :param policies: List of policies in same order as gym observations.
        :param batch_size: Number of trajectories to collect for each training step.
        :param potential: Bregman potential to use for optimizer.
        :param device: Device to compute on.

        Initialize training wrapper with any gym and policies.
        """
        super(MultiSimGD, self).__init__(
            env,
            policies,
            critics,
            batch_size=batch_size,
            critic_lr=critic_lr,
            tol=tol,
            device=device,
            dtype=dtype,
            self_play=self_play,
            gamma=gamma,
            tau=tau
        )

        # Optimizers for policies and critics.
        self.policy_optim = gda_utils.SGD(
            [p.parameters() for p in self.policies],
            [policy_lr for _ in self.policies],
            device=device
        )


    def step(self, mat_states, mat_actions, mat_action_mask, mat_rewards, mat_done, verbose=False):
        """
        Compute update step for policies and critics.
        """
        if verbose:
            torch.cuda.synchronize()
            step_start_time = time.time()

        # Use critic function to get advantage.
        values, returns, advantages = [], [], []

        # TODO(jjma): Fix this when making self-play more robust.
        critics = self.critics
        if self.self_play:
            critics = [self.critics[0] for _ in range(len(self.policies))]

        # Compute generalized advantage estimation (GAE).
        for i, q in enumerate(critics):
            val = q(mat_states[i]).detach()
            ret = get_advantage(0, mat_rewards[i], val, mat_done[i], 
                                tau=self.tau, gamma=self.gamma, device=self.device)

            advantage = ret - val

            values.append(val)
            returns.append(ret)
            advantages.append(torch.squeeze(advantage))

        # Use sampled values to fit critic model.
        if self.self_play:
            # TODO(jjma): Currently only supports all symmetric players.
            cat_states = torch.cat([mat_states[i] for i in range(len(mat_states))])
            cat_returns = torch.cat(returns)

            critic_update(cat_states, cat_returns, self.critics[0], self.critic_optim[0])
        else:
            for i, q in enumerate(self.critics):
                critic_update(mat_states[i], returns[i], q, self.critic_optim[i])

        # Calculate log probabilities as well as compute gradient pseudoobjectives.
        log_probs = []
        gradient_losses = []
        for i, p in enumerate(self.policies):
            # Our training wrapper assumes that the policy returns a distribution.
            lp_inid = p(mat_states[i]).log_prob(mat_actions[i])
            # TODO(jjma): For games with single action per state, this works.
            lp = lp_inid
            if lp_inid.ndim > 1:
                lp = lp_inid.sum(1)

            # Track log probabilities later to compute Hessian pseudoobjectives.
            log_probs.append(lp)

            # Get gradient objective per player, which is log probabilty times advantage.
            prod = lp * advantages[i] * mat_action_mask[i]
            grad_loss = -prod.sum() / mat_action_mask[i].sum()

            gradient_losses.append(grad_loss)

        # Update the policy parameters.
        self.policy_optim.zero_grad()
        self.policy_optim.step(gradient_losses)

        gradient_losses.clear()

        # Print sampling time for debugging purposes.
        if verbose:
            torch.cuda.synchronize()
            print('step took:', time.time() - step_start_time)


# TODO(jjma): Refine this class structure based on feedback.
class MultiCoPG(TrainingWrapper):
    # TODO(jjma): Horizon, gamma tuning?
    def __init__(
        self,
        env,
        policies,
        critics,
        batch_size=32,
        critic_lr=1e-3,
        tol=1e-6,
        device=torch.device('cpu'),
        dtype=DEFAULT_DTYPE,
        potential=potentials.squared_distance(1),
        self_play=False,
        gamma=0.99,
        tau=0.95,
    ):
        """
        :param env: OpenAI gym instance to train on
        :param policies: List of policies in same order as gym observations.
        :param batch_size: Number of trajectories to collect for each training step.
        :param potential: Bregman potential to use for optimizer.
        :param device: Device to compute on.

        Initialize training wrapper with any gym and policies.
        """
        super(MultiCoPG, self).__init__(
            env,
            policies,
            critics,
            batch_size=batch_size,
            critic_lr=critic_lr,
            tol=tol,
            device=device,
            dtype=dtype,
            self_play=self_play,
            gamma=gamma,
            tau=tau,
        )

        # Optimizers for policies and critics.
        self.policy_optim = cmd_utils.CMD_RL(
            [p.parameters() for p in self.policies],
            bregman=potential,
            tol=tol,
            device=self.device
        )


    def step(self, mat_states, mat_actions, mat_action_mask, mat_rewards, mat_done, verbose=False):
        """
        Compute update step for policies and critics.
        """
        if verbose:
            torch.cuda.synchronize()
            step_start_time = time.time()

        # Use critic function to get advantage.
        values, returns, advantages = [], [], []

        # TODO(jjma): Fix this when making self-play more robust.
        critics = self.critics
        if self.self_play:
            critics = [self.critics[0] for _ in range(len(self.policies))]

        # Compute generalized advantage estimation (GAE).
        for i, q in enumerate(critics):
            val = q(mat_states[i]).detach()
            ret = get_advantage(0, mat_rewards[i], val, mat_done[i], 
                                tau=self.tau, gamma=self.gamma, device=self.device)
            advantage = ret - val

            values.append(val)
            returns.append(ret)
            advantages.append(torch.squeeze(advantage))

        # Use sampled values to fit critic model.
        if self.self_play:
            # TODO(jjma): Currently only supports all symmetric players.
            cat_states = torch.cat([mat_states[i] for i in range(len(mat_states))])
            cat_returns = torch.cat(returns)

            critic_update(cat_states, cat_returns, self.critics[0], self.critic_optim[0])
        else:
            for i, q in enumerate(self.critics):
                critic_update(mat_states[i], returns[i], q, self.critic_optim[i])

        # Calculate log probabilities as well as compute gradient pseudoobjectives.
        log_probs = []
        gradient_losses = []
        for i, p in enumerate(self.policies):
            # Our training wrapper assumes that the policy returns a distribution.
            lp_inid = p(mat_states[i]).log_prob(mat_actions[i])
            # TODO(jjma): For games with single action per state, this works.
            lp = lp_inid
            if lp_inid.ndim > 1:
                lp = lp_inid.sum(1)

            # Track log probabilities later to compute Hessian pseudoobjectives.
            log_probs.append(lp)

            # Get gradient objective per player, which is log probabilty times advantage.
            prod = lp * advantages[i] * mat_action_mask[i]
            grad_loss = -prod.sum() / mat_action_mask[i].sum()

            gradient_losses.append(grad_loss)

        # TODO(jjma): Calculate indices of trajectory. This assumes that all agents
        # have trajectory with splits same to the player with the longest trajectory.
        traj_indices = [0]
        traj_done = True
        for i in range(1, mat_done[0].size(0)):
            # When we start another trajectory, we mark the starting index.
            if traj_done and mat_done[0][i] == 1.:
                traj_indices.append(i)
                traj_done = False

            # Otherwise, when we encounter 0, we know trajectory is over.
            elif mat_done[0][i] == 0.:
                traj_done = True
        # Include last index to compute pairs.
        traj_indices.append(mat_done[0].size(0))

#         # Old slow implementation.
#         s_log_probs = [torch.zeros_like(lp) for lp in log_probs]
#         for lp, slp, mask in zip(log_probs, s_log_probs, mat_done):
#             for i in range(slp.size(0)):
#                 if i == 0:
#                     slp[0] = 0.
#                 else:
#                     slp[i] = torch.add(slp[i-1], lp[i-1]) * mask[i-1]

        # Compute summed log probabilities for Hessian pseudoobjectives.
        s_log_probs = []
        for lp, action_mask in zip(log_probs, mat_action_mask):
            # Compute cumsums over each trajectory.
            traj_cumsums = []
            for i in range(len(traj_indices)-1):
                # Get consecutive trajectory boundary indices for slicing.
                start = traj_indices[i]
                end = traj_indices[i+1]

                # Compute normal cumsum, append 0 to front to align, and
                # slice out trajectory length as givein in CoPG.
                cumsum = torch.cumsum(lp[start:end], dim=0)
                new_cumsum = torch.cat([
                    torch.tensor([0.], device=self.device, dtype=self.dtype),
                    cumsum
                ])[:cumsum.size(0)]

                traj_cumsums.append(new_cumsum)

            s_log_probs.append(torch.cat(traj_cumsums) * action_mask)

        # Compute Hessian objectives.
        hessian_losses = [0. for _ in range(len(self.policies))]
        for i in range(len(log_probs)):
            for j in range(len(log_probs)):
                if (i != j):
                    hessian_losses[i] -= (log_probs[i] * log_probs[j] * advantages[i]).mean()

                    term1 = s_log_probs[i][:-1] * log_probs[j][1:] * advantages[i][1:]
                    hessian_losses[i] -= term1.sum() / (term1.size(0) - self.batch_size + 1)

                    term2 = log_probs[i][1:] * s_log_probs[j][:-1] * advantages[i][1:]
                    hessian_losses[i] -= term2.sum() / (term2.size(0) - self.batch_size + 1)

        # Update the policy parameters.
        self.policy_optim.zero_grad()
        self.policy_optim.step(gradient_losses, hessian_losses, cgd=True)

        gradient_losses.clear()
        hessian_losses.clear()

        # Print sampling time for debugging purposes.
        if verbose:
            torch.cuda.synchronize()
            print('step took:', time.time() - step_start_time)
