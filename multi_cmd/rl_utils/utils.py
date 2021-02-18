import torch

def critic_update(state_mat, return_mat, q, optim_q):
    val_loc = q(state_mat)
    critic_loss = (return_mat - val_loc).pow(2).mean()

    optim_q.zero_grad()
    critic_loss.backward()
    optim_q.step()

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
    return torch.cat(returns).reshape(-1, 1)
