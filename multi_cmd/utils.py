import torch
import torch.autograd as autograd


ADD_FUNC = lambda x, y: x + y
SUB_FUNC = lambda x, y: x - y
INPLACE_ADD_FUNC = lambda x, y: x.data.add_(y)
INPLACE_SUB_FUNC = lambda x, y: x.data.sub_(y)


def vec_list_dot(v1, v2):
    assert(len(v1) == len(v2))
    return sum([
        sum([
            torch.dot(elem1, elem2) for elem1, elem2 in zip(v1_elem, v2_elem)
        ]) for v1_elem, v2_elem in zip(v1, v2)
    ])


def vec_list_op(v1, v2, f):
    assert(len(v1) == len(v2))
    return [
        [f(elem1, elem2) for elem1, elem2 in zip(v1_elem, v2_elem)]
        for v1_elem, v2_elem in zip(v1, v2)
    ]


def vec_list_map(v, f):
    return [[f(elem) for elem in v_elem] for v_elem in v]


def grad_tuple_to_vec(
    grad_tuple, param_tensors, neg=False
):
    """
    Converts gradient tuple from tuple to list of tensors, replacing None
    gradient with zero gradient.
    """
    grad_elems = []

    for i, elem in enumerate(grad_tuple):
        if elem is None:
            grad_elems.append(
                torch.zeros_like(param_tensors[i], requires_grad=True))
        else:
            grad_elems.append(elem if not neg else -elem)

    return grad_elems


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

# def Hvp_vec(grad_vec, params, vec, retain_graph=True):
#     '''
#     :param grad_vec: tensor of which the Hessian vector product will be computed
#     :param params: list of params, w.r.t which the Hessian will be computed
#     :param vec: the "vector" in Hessian vector product
#     :param retain_graph: save if set to True
#
#     Computes Hessian vector product.
#     '''
#     if torch.isnan(grad_vec).any():
#         raise ValueError('Gradvec nan')
#     if torch.isnan(vec).any():
#         raise ValueError('vector nan')
#
#     grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph,
#                               allow_unused=True)
#     grad_list = []
#     for i, p in enumerate(params):
#         if grad_grad[i] is None:
#             grad_list.append(torch.zeros_like(p).view(-1))
#         else:
#             grad_list.append(grad_grad[i].contiguous().view(-1))
#     hvp = torch.cat(grad_list)
#     if torch.isnan(hvp).any():
#         raise ValueError('hvp Nan')
#     return hvp
