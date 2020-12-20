import torch
import torch.autograd as autograd

ADD_FUNC = lambda x, y: x + y
SUB_FUNC = lambda x, y: x - y
ADD_FUNC_INPLACE = lambda x, y: x.data.add_(y)
SUB_FUNC_INPLACE = lambda x, y: x.data.sub_(y)

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
    
def zero_grad(params):
    for param in params:
        if param.grad is not None:
            p.grad.detach()
            p.grad.zero_()
                
def Hvp_vec(grad_vec, params, vec, retain_graph=True):
    '''
    :param grad_vec: tensor of which the Hessian vector product will be computed
    :param params: list of params, w.r.t which the Hessian will be computed
    :param vec: the "vector" in Hessian vector product
    :param retain_graph: save if set to True
    
    Computes Hessian vector product.
    '''
    if torch.isnan(grad_vec).any():
        raise ValueError('Gradvec nan')
    if torch.isnan(vec).any():
        raise ValueError('vector nan')

    grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph,
                              allow_unused=True)
    grad_list = []
    for i, p in enumerate(params):
        if grad_grad[i] is None:
            grad_list.append(torch.zeros_like(p).view(-1))
        else:
            grad_list.append(grad_grad[i].contiguous().view(-1))
    hvp = torch.cat(grad_list)
    if torch.isnan(hvp).any():
        raise ValueError('hvp Nan')
    return hvp