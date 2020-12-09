import torch
import torch.autograd as autograd
from utils import *

def compute_update(loss, param, lr=1, retain_graph=True):
    grad_param = autograd.grad(loss, param,
                               retain_graph=retain_graph,
                               allow_unused=True)
    grad_vec = grad_tuple_to_vec(grad_param, param)
    return -lr * grad_vec
