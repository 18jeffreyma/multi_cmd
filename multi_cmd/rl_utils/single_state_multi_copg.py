
import os
import copy
import time
import numpy as np
import torch
from multi_cmd.optim import cmd_utils, potentials
from multi_cmd.optim import gda_utils

# TODO(anonymous): Casework for this?
DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)


class SingleStateTrainingWrapper:
    def __init__(
        self,
        env,
        policies,
        batch_size=32,
        tol=1e-6,
        device=torch.device('cpu'),
        dtype=DEFAULT_DTYPE,
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

        # Sampling parameters.
        self.batch_size = batch_size

        # Device to be used.
        self.device = device
        self.dtype = dtype

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
                    
                    action = dist.sample().cpu().numpy()
                    # TODO(anonymous): Pytorch doesn't handle 0-dim tensors (a.k.a scalars well)
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


# TODO(anonymous): Refine this class structure based on feedback.
class SingleStateMultiSimGD(SingleStateTrainingWrapper):
    # TODO(anonymous): Horizon, gamma tuning?
    def __init__(
        self,
        env,
        policies,
        batch_size=32,
        tol=1e-3,
        device=torch.device('cpu'),
        dtype=DEFAULT_DTYPE,
        policy_lr=0.002,
    ):
        """
        :param env: OpenAI gym instance to train on
        :param policies: List of policies in same order as gym observations.
        :param batch_size: Number of trajectories to collect for each training step.
        :param potential: Bregman potential to use for optimizer.
        :param device: Device to compute on.

        Initialize training wrapper with any gym and policies.
        """
        super(SingleStateMultiSimGD, self).__init__(
            env,
            policies,
            batch_size=batch_size,
            tol=tol,
            device=device,
            dtype=dtype,
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

        log_probs = []
        gradient_losses = []
        for i, p in enumerate(self.policies):
            # Our training wrapper assumes that the policy returns a distribution.
            lp_inid = p(mat_states[i]).log_prob(mat_actions[i])

            # TODO(anonymous): For games with single action per state, this works.
            lp = lp_inid
            if lp_inid.ndim > 1:
                lp = lp_inid.sum(1)

            # Track log probabilities later to compute Hessian pseudoobjectives.
            log_probs.append(lp)

            # Get gradient objective per player, which is log probabilty times advantage.
            prod = torch.squeeze(lp) * mat_rewards[i]
            gradient_losses.append(-prod.sum() / mat_action_mask[i].sum())

        # Update the policy parameters.
        self.policy_optim.zero_grad()
        self.policy_optim.step(gradient_losses)

        gradient_losses.clear()

        # Print sampling time for debugging purposes.
        if verbose:
            torch.cuda.synchronize()
            print('step took:', time.time() - step_start_time)


# TODO(anonymous): Refine this class structure based on feedback.
class SingleStateMultiCoPG(SingleStateTrainingWrapper):
    # TODO(anonymous): Horizon, gamma tuning?
    def __init__(
        self,
        env,
        policies,
        batch_size=32,
        tol=1e-6,
        device=torch.device('cpu'),
        dtype=DEFAULT_DTYPE,
        potential=potentials.squared_distance(1),
    ):
        """
        :param env: OpenAI gym instance to train on
        :param policies: List of policies in same order as gym observations.
        :param batch_size: Number of trajectories to collect for each training step.
        :param potential: Bregman potential to use for optimizer.
        :param device: Device to compute on.

        Initialize training wrapper with any gym and policies.
        """
        super(SingleStateMultiCoPG, self).__init__(
            env,
            policies,
            batch_size=batch_size,
            tol=tol,
            device=device,
            dtype=dtype,
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

        log_probs = []
        gradient_losses = []
        for i, p in enumerate(self.policies):
            # Our training wrapper assumes that the policy returns a distribution.
            lp_inid = p(mat_states[i]).log_prob(mat_actions[i])

            # TODO(anonymous): For games with single action per state, this works.
            lp = lp_inid
            if lp_inid.ndim > 1:
                lp = lp_inid.sum(1)

            # Track log probabilities later to compute Hessian pseudoobjectives.
            log_probs.append(lp)

            # Get gradient objective per player, which is log probabilty times advantage.
            prod = torch.squeeze(lp) * mat_rewards[i]
            print(prod.shape)
            gradient_losses.append(-prod.sum() / mat_action_mask[i].sum())

        # Compute summed log probabilities for Hessian pseudoobjectives.
        s_log_probs = []
        for lp in log_probs:
            slp = torch.cumsum(lp.clone(), 0)
            s_log_probs.append(slp)

        # Compute Hessian objectives.
        hessian_losses = [0. for _ in range(len(self.policies))]
        for i in range(len(log_probs)):
            for j in range(len(log_probs)):
                if (i != j):
                    hessian_losses[i] -= (torch.squeeze(log_probs[i] * log_probs[j]) * mat_rewards[i]).mean()

                    # term1 = s_log_probs[i] * log_probs[j] * mat_rewards[i]
                    # hessian_losses[i] -= term1.mean()

                    # term2 = log_probs[i] * s_log_probs[j] * mat_rewards[i]
                    # hessian_losses[i] -= term2.mean()

        # Update the policy parameters.
        self.policy_optim.zero_grad()
        self.policy_optim.step(gradient_losses, hessian_losses, cgd=True)

        gradient_losses.clear()
        hessian_losses.clear()

        # Print sampling time for debugging purposes.
        if verbose:
            torch.cuda.synchronize()
            print('step took:', time.time() - step_start_time)
