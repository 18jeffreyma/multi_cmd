import torch
import torch.autograd as autograd

from . import utils
from . import potentials

def avp(
    loss_list,
    param_list,
    vector_list,
    bregman=potentials.squared_distance(1),
    transpose=False,
    retain_graph=True,
    detach=True,
):
    """
    :param vector_list: list of vectors for each player
    :param loss_list: list of objective functions for each player
    :param param_list: list of parameter vectors for each player
    :param bregman: dictionary representing bregman potential to use
    :param transpose: compute product against transpose if set
    :param retain_graph: save

    Computes right product of metamatrix with a vector of player vectors.
    """
    # TODO(jjma): add error handling and assertions
    assert(len(loss_list) == len(param_list))
    assert(len(loss_list) == len(vector_list))

    prod_list = [torch.zeros_like(param) for param in param_list]

    for i, row_param in enumerate(param_list):
        for j, (col_param, vector_elem) in enumerate(zip(param_list, vector_list)):

            if i == j:
                prod_list[i] += bregman['Dxx_vp'](row_param, vector_elem)
                continue

            loss = loss_list[i] if not transpose else loss_list[j]

            grad_param = autograd.grad(loss, col_param,
                                       create_graph=retain_graph,
                                       retain_graph=retain_graph,
                                       allow_unused=True)
            grad_param_vec = utils.grad_tuple_to_vec(grad_param, col_param)

            # Compute relevant mixed Hessian for metamatrix multiplication.
            hvp_vec = utils.Hvp_vec(grad_param_vec, [row_param], vector_elem)

            if detach:
                hvp_vec = hvp_vec.detach()

            prod_list[i] += hvp_vec

    return prod_list

def metamatrix_conjugate_gradient(
    loss_list,
    param_list,
    vector_list=None,
    bregman=potentials.squared_distance(1),
    n_steps=5,
    tol=1e-6,
    atol=1e-6,
    retain_graph=True,
    detach=True,
):
    """
    :param loss_list: list of loss tensors for each player
    :param param_list: list of player params to compute gradients from
    :param vector_list: initial guess for update solution
    :param bregman: dict representing a Bregman potential to be used
    :param n_steps: number of iteration steps for conjugate gradient
    :param tol: relative residual tolerance threshold from initial vector guess
    :param atol: absolute residual tolerance threshold

    Compute solution to meta-matrix game form using conjugate gradient method. Since
    the metamatrix A is not p.s.d, we multiply both sides by the transpose to
    ensure p.s.d.

    In other words, note that solving Ax = b (where A is meta matrix, x is
    vector of update vectors and b is learning rate times vector of gradients
    is the same as solving A'x = b' (where A' = (A^T)A and b' = (A^T)b.
    """

    b = []
    for loss, param in zip(loss_list, param_list):

        grad_param = autograd.grad(loss, param,
                                   retain_graph=retain_graph,
                                   allow_unused=True)
        grad_vec = utils.grad_tuple_to_vec(grad_param, param)
        b.append(-grad_vec)

    # Multiplying both sides by transpose to ensure p.s.d.
    # r = A^t * b (before we subtract)
    r = avp(loss_list, param_list, b, bregman=bregman, transpose=True)

    # Set relative residual threshold based on norm of b.
    norm_At_b = utils.vec_list_dot(r, r)
    residual_tol = tol * norm_At_b

    # If no guess provided, start from zero vector.
    if vector_list is None:
        vector_list = [torch.zeros(param.shape[0]) for param in param_list]

    else:
        # Compute initial residual if a guess is given.
        A_x = avp(loss_list, param_list, vector_list,
                  bregman=bregman, transpose=False)
        At_A_x = avp(loss_list, param_list, A_x,
                     bregman=bregman, transpose=True)

        r = utils.vec_list_op(r, At_A_x, utils.SUB_FUNC)

    # Early exit if solution already found.
    rdotr = utils.vec_list_dot(r, r)
    if rdotr < residual_tol or rdotr < atol:
        return vector_list, 0

    # Define p and measure current candidate vector
    p = [r_elem.clone().detach() for r_elem in r]

    # Use conjugate gradient to find vector solution
    for i in range(n_steps):

        A_p = avp(loss_list, param_list, p,
                  bregman=bregman, transpose=False)
        At_A_p = avp(loss_list, param_list, A_p,
                     bregman=bregman, transpose=True)

        alpha = rdotr / utils.vec_list_dot(p, At_A_p)

        alpha_mul = lambda x: alpha * x
        alpha_p = utils.vec_list_map(p, alpha_mul)
        alpha_At_A_p = utils.vec_list_map(At_A_p, alpha_mul)

        # Update candidate solution and residual
        utils.vec_list_op(vector_list, alpha_p, utils.ADD_FUNC_INPLACE)
        utils.vec_list_op(r, alpha_At_A_p, utils.SUB_FUNC_INPLACE)

        # Calculate new residual metric
        new_rdotr = utils.vec_list_dot(r, r)

        # Break if solution is within threshold
        if new_rdotr < atol or new_rdotr < residual_tol:
            break

        # Otherwise, update and continue
        beta = new_rdotr / rdotr
        beta_p = utils.vec_list_map(p, lambda x: beta * x)

        p = utils.vec_list_op(r, beta_p, utils.ADD_FUNC)
        rdotr = new_rdotr

    # Detach, since we no longer need derivatives.
    if detach:
        vector_list = [param.detach().requires_grad_() for param in vector_list]

    return vector_list, i


def exp_map(param_list, nash_list,
            bregman=potentials.squared_distance(1),
            in_place=True,
            detach=True):
    """
    :param param_list: list of player params before update
    :param nash_list: nash equilibrium solutions computed from minimization step

    Map dual system coordinate solution back to primal, accounting
    for feasibility constraints specified in Bregman potential.
    """
    with torch.no_grad():
        def combine(param, nash):
            return bregman['Dx'](bregman['Dx_inv'](param) + bregman['Dxx_vp'](param, nash))

        mapped = utils.vec_list_op(param_list, nash_list, combine)

        # Detach, since we no longer need derivatives.
        if detach:
            mapped = [param.detach().requires_grad_() for param in mapped]

        return mapped


# TODO(jjma): make this user interface cleaner.
class CMD(object):
    """Optimizer class for the CMD algorithm."""
    def __init__(self, player_list,
                 bregman=potentials.squared_distance(1),
                 tol=1e-6, atol=1e-6,
                 device=torch.device('cpu')
                ):
        self.bregman = bregman
        self.state = {'step': 0,
                      'player_list': player_list,
                      'tol': tol, 'atol': atol,
                      'last_dual_soln': None,
                      'last_dual_soln_n_iter': 0}
        # TODO(jjma): set this device in CMD algorithm.
        self.device = device

    def zero_grad(self):
        for player in player_list:
            utils.zero_grad(player)

    def state_dict(self):
        return self.state

    def player_list(self):
        return self.state['player_list']

    def step(self, loss_list):
        nash_list, n_iter = metamatrix_conjugate_gradient(
            loss_list,
            self.state['player_list'],
            vector_list=self.state['last_dual_soln'],
            bregman=self.bregman,
            tol=self.state['tol'],
            atol=self.state['atol']
        )

        self.state['step'] += 1
        self.state['last_dual_soln'] = nash_list
        self.state['last_dual_soln_n_iter'] = n_iter

        mapped_list = exp_map(self.state['player_list'],
                              nash_list,
                              bregman=self.bregman)

        # In-place update parameters to properly work as optimizer.
        for player, mapped in zip(self.state['player_list'], mapped_list):
            player.data = mapped
