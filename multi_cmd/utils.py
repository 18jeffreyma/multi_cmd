"""Helper methods for operating on lists of lists of Tensors."""
import torch
import torch.autograd as autograd


ADD_FUNC = lambda x, y: torch._foreach_add(x, y)
SUB_FUNC = lambda x, y: torch._foreach_sub(x, y)
INPLACE_ADD_FUNC = lambda x, y: torch._foreach_add_(x, y)
INPLACE_SUB_FUNC = lambda x, y: torch._foreach_sub_(x, y)

INPLACE_ALPHA_ADD_FUNC = lambda alpha: (
    lambda x, y: torch._foreach_add_(x, y, alpha=alpha))
INPLACE_ALPHA_SUB_FUNC = lambda alpha: (
    lambda x, y: torch._foreach_sub_(x, y, alpha=alpha))


def player_list_dot(v1, v2):
    """Compute dot product of two lists of lists of Tensors."""
    # assert(len(v1) == len(v2))
    return sum([
        sum([
            torch.dot(elem1, elem2) for elem1, elem2 in zip(v1_elem, v2_elem)
        ]) for v1_elem, v2_elem in zip(v1, v2)
    ])


def player_list_op(v1, v2, foreach_f):
    """Map a function mapping two lists of elements to a list of elements."""
    # assert(len(v1) == len(v2))
    return tuple(
        foreach_f(v1_elem, v2_elem) for v1_elem, v2_elem in zip(v1, v2)
    )


def player_list_map(v, f):
    """Map a function over each element in a list of list of elements."""
    return tuple(tuple(f(elem) for elem in v_elem) for v_elem in v)


def filter_none_grad(
    grad_tuple, param_tensors, neg=False, detach=False
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
            if detach:
                elem = elem.detach()
            grad_elems.append(elem if not neg else -elem)

    return tuple(grad_elems)
