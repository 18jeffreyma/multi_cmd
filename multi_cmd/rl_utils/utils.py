import torch

def critic_update(state_mat, return_mat, q, optim_q):
    val_loc = q(state_mb)
    critic_loss = (return_mat - state_mat).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()

# TODO(jjma): Revisit this?
def get_advantage(
    next_value, reward_mat, value_mat, masks,
    gamma=0.99, tau=0.95, device=torch.device('cpu')
):
    value_mat = torch.cat([value_mat, torch.tensor([[next_value]], device=device)])
    gae = 0
    returns = []

    for i in reversed(range(len(reward_mat))):
        delta = reward_mat[i] + gamma * value_mat[i+1] * masks[i] - value_mat[i]
        gae = delta + gamma * tau * masks[i] * gae
        returns.append(gae + value_mat[i])

    # Reverse ordering.
    returns.reverse()
    return torch.cat(returns)
