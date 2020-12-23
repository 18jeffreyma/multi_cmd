import torch
import torch.autograd as autograd

from multi_cmd import utils
from multi_cmd import potentials

def avp(
    hessian_loss_list,
    player_list,
    vector_list,
    bregman=potentials.squared_distance(1),
    transpose=False,
    retain_graph=True,
    detach=True,
):
    """
    :param hessian_loss_list: list of objective functions for hessian computation
    :param vector_list: list of list of vectors for each player
    :param player_list: list of lists of player tensors to compute gradients from
    :param bregman: dictionary representing bregman potential to use
    :param transpose: compute product against transpose if set
    :param retain_graph: save

    Computes right product of metamatrix with a vector of player vectors.
    """
    # TODO(jjma): add error handling and assertions
    # assert(len(hessian_loss_list) == len(player_list))
    # assert(len(hessian_loss_list) == len(vector_list))

    prod_list = utils.player_list_map(player_list, lambda p: torch.zeros_like(p))

    for i, row_params in enumerate(player_list):
        for j, (col_params, vector_elem) in enumerate(zip(player_list, vector_list)):
            if i == j:
                # Diagonal element is the Bregman term.

                # TODO(jjma): Check if all Bregman potentials can be evaluated
                # element-wise; if so, we can evaluate this tensor by tensor as
                # below.
                bregman_tuple = tuple(bregman['Dxx_vp'](row_param, v_elem)
                                      for row_param, v_elem in zip(row_params, vector_elem))
                torch._foreach_add_(prod_list[i], bregman_tuple)

                continue

            # Otherwise, we construct our Hessian vector products.
            loss = hessian_loss_list[i] if not transpose else hessian_loss_list[j]

            grad_raw = autograd.grad(loss, col_params,
                                     create_graph=retain_graph,
                                     retain_graph=retain_graph,
                                     allow_unused=True)
            grad_tuple = utils.filter_none_grad(grad_raw, col_params)

            grad_dot_prod = sum(torch.dot(grad, vec)
                                for grad, vec in zip(grad_tuple, vector_elem))

            hvp_raw = autograd.grad(grad_dot_prod, row_params,
                                      create_graph=retain_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=True)
            hvp_tuple = utils.filter_none_grad(hvp_raw, row_params)

            torch._foreach_add_(prod_list[i], hvp_tuple)

    return prod_list

def metamatrix_conjugate_gradient(
    grad_loss_list,
    hessian_loss_list,
    player_list,
    vector_list=None,
    bregman=potentials.squared_distance(1),
    n_steps=5,
    tol=1e-6,
    atol=1e-6,
    retain_graph=True,
    detach=True,
):
    """
    :param grad_loss_list: list of loss tensors for each player to compute gradient
    :param hessian_loss_list: list of loss tensors for each player to compute hessian
    :param player_list: list of lists of player tensors to compute gradients from
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
    for loss, param_tensors in zip(grad_loss_list, player_list):
        # Get vector list of negative gradients.
        grad_param_tuple = autograd.grad(loss, param_tensors,
                                         retain_graph=retain_graph,
                                         allow_unused=True)
        grad_vec_list = utils.filter_none_grad(grad_param_tuple,
                                               param_tensors,
                                               neg=True,
                                               detach=True)
        b.append(grad_vec_list)
    b = tuple(b)

    # Multiplying both sides by transpose to ensure p.s.d.
    # r = A^t * b (before we subtract)
    r = avp(hessian_loss_list, player_list, b, bregman=bregman, transpose=True)

    # Set relative residual threshold based on norm of b.
    norm_At_b = utils.player_list_dot(r, r)
    residual_tol = tol * norm_At_b

    # If no guess provided, start from zero vector.
    if vector_list is None:
        vector_list = utils.player_list_map(player_list,
                                            lambda p: torch.zeros_like(p))
    else:
        # Compute initial residual if a guess is given.
        A_x = avp(hessian_loss_list, player_list, vector_list,
                  bregman=bregman, transpose=False)
        At_A_x = avp(hessian_loss_list, player_list, A_x,
                     bregman=bregman, transpose=True)

        utils.player_list_op(r, At_A_x, utils.INPLACE_SUB_FUNC)

    # Early exit if solution already found.
    rdotr = utils.player_list_dot(r, r)
    if rdotr < residual_tol or rdotr < atol:
        return vector_list, 0

    # Define p and measure current candidate vector
    p = utils.player_list_map(r, lambda elem: elem.clone().detach())

    # Use conjugate gradient to find vector solution
    for i in range(n_steps):

        A_p = avp(hessian_loss_list, player_list, p,
                  bregman=bregman, transpose=False)
        At_A_p = avp(hessian_loss_list, player_list, A_p,
                     bregman=bregman, transpose=True)

        with torch.no_grad():
            alpha = rdotr / utils.player_list_dot(p, At_A_p)

            # Update candidate solution and residual, where:
            # (1) x_new = x + alpha * p
            # (2) r_new = r - alpha * A' * p
            utils.player_list_op(vector_list, p,
                                 utils.INPLACE_ALPHA_ADD_FUNC(alpha))
            utils.player_list_op(r, At_A_p,
                                 utils.INPLACE_ALPHA_SUB_FUNC(alpha))

            # Calculate new residual metric
            new_rdotr = utils.player_list_dot(r, r)

            # Break if solution is within threshold
            if new_rdotr < atol or new_rdotr < residual_tol:
                break

            # Otherwise, update and continue.
            # (3) p_new = r_new + beta * p
            beta = new_rdotr / rdotr
            beta_p = utils.player_list_map(p, lambda x: beta * x)
            p = utils.player_list_op(r, beta_p, utils.ADD_FUNC)

            rdotr = new_rdotr

    return vector_list, i


def exp_map(player_list, nash_list,
            bregman=potentials.squared_distance(1),
            in_place=True
):
    """
    :param player_list: list of player params before update
    :param nash_list: nash equilibrium solutions computed from minimization step

    Map dual system coordinate solution back to primal, accounting
    for feasibility constraints specified in Bregman potential.
    """
    with torch.no_grad():
        def combine(param, nash):
            return bregman['Dx'](bregman['Dx_inv'](param) +
                                 bregman['Dxx_vp'](param, nash))

        mapped = tuple(
            tuple(
                combine(p, n) for p, n in zip(param, nash)
            ) for param, nash in zip(player_list, nash_list)
        )

    return mapped


# TODO(jjma): make this user interface cleaner.
class CMD(object):
    """Optimizer class for the CMD algorithm with differentiable player objectives."""
    def __init__(self, player_list,
                 bregman=potentials.squared_distance(1),
                 tol=1e-6, atol=1e-6,
                 device=torch.device('cpu')
                ):
        """
        :param player_list: list (per player) of list of Tensors, representing parameters
        :param bregman: dict representing Bregman potential to be used
        """
        self.bregman = bregman

        # In case, parameter generators are provided.
        player_list = tuple(tuple(elem) for elem in player_list)

        # Store optimizer state.
        self.state = {'step': 0,
                      'player_list': player_list,
                      'tol': tol, 'atol': atol,
                      'last_dual_soln': None,
                      'last_dual_soln_n_iter': 0}
        # TODO(jjma): set this device in CMD algorithm.
        self.device = device

    def zero_grad(self):
        for player in self.state['player_list']:
            utils.zero_grad(player)

    def state_dict(self):
        return self.state

    def player_list(self):
        return self.state['player_list']

    def step(self, loss_list):
        # Compute dual solution first, before mapping back to primal.
        nash_list, n_iter = metamatrix_conjugate_gradient(
            loss_list,
            loss_list,
            self.state['player_list'],
            vector_list=self.state['last_dual_soln'],
            bregman=self.bregman,
            tol=self.state['tol'],
            atol=self.state['atol']
        )

        # Store state for use in next nash computation..
        self.state['step'] += 1
        self.state['last_dual_soln'] = nash_list
        self.state['last_dual_soln_n_iter'] = n_iter

        # Map dual solution back into primal space.
        mapped_list = exp_map(self.state['player_list'],
                              nash_list,
                              bregman=self.bregman)

        # Update parameters in place to update players as optimizer.
        for player_params, mapped_params in zip(self.state['player_list'], mapped_list):
            for param, mapped_param in zip(player_params, mapped_params):
                param.data = mapped_param


class CMD_RL(CMD):
    """RL optimizer using CMD algorithm, using derivation from CoPG paper."""
    def __init__(self, player_list,
                 bregman=potentials.squared_distance(1),
                 tol=1e-6, atol=1e-6,
                 device=torch.device('cpu')
                ):
        """
        :param player_list: list (per player) of list of Tensors, representing parameters
        :param bregman: dict representing Bregman potential to be used
        """
        super(CMD_RL, self).__init__(player_list,
                                     bregman=bregman,
                                     tol=tol, atol=atol,
                                     device=device)

    def step(self, grad_loss_list, hessian_loss_list):
        """
        CMD algorithm using derivation for gradient and hessian term from CoPG.
        """
        # TODO(jjma): Add documentation.

        # Compute dual solution first, before mapping back to primal.
        nash_list, n_iter = metamatrix_conjugate_gradient(
            grad_loss_list,
            hessian_loss_list,
            self.state['player_list'],
            vector_list=self.state['last_dual_soln'],
            bregman=self.bregman,
            tol=self.state['tol'],
            atol=self.state['atol']
        )

        # Store state for use in next nash computation..
        self.state['step'] += 1
        self.state['last_dual_soln'] = nash_list
        self.state['last_dual_soln_n_iter'] = n_iter

        # Map dual solution back into primal space.
        mapped_list = exp_map(self.state['player_list'],
                              nash_list,
                              bregman=self.bregman)

        # Update parameters in place to update players as optimizer.
        for player_params, mapped_params in zip(self.state['player_list'], mapped_list):
            for param, mapped_param in zip(player_params, mapped_params):
                param.data = mapped_param
