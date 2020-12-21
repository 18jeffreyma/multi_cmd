import torch
import torch.autograd as autograd

from multi_cmd import utils

def compute_update(loss, param, lr=1, retain_graph=True):
    grad_param = autograd.grad(loss, param,
                               retain_graph=retain_graph,
                               allow_unused=True)
    grad_vec = utils.grad_tuple_to_vec(grad_param, param)
    return -lr * grad_vec
