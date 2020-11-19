import torch
import torch.autograd as autograd

ADD_FUNC = lambda x, y: x + y
SUB_FUNC = lambda x, y: x - y

def vec_list_dot(v1, v2):
    assert(len(v1) == len(v2))
    return sum([torch.dot(elem1, elem2) for elem1, elem2 in zip(v1, v2)])

def vec_list_op(v1, v2, f):
    assert(len(v1) == len(v2))
    return [f(elem1, elem2) for elem1, elem2 in zip(v1, v2)]

def vec_list_map(v, f):
    return list(map(f, v))

def grad_tuple_to_vec(grad_tuple, param):
    assert(len(grad_tuple) == 1)
    if grad_tuple[0] is None:
        return torch.zeros_like(param, requires_grad=True).view(-1)
    else:
        return torch.cat([g.contiguous().view(-1) for g in grad_tuple])