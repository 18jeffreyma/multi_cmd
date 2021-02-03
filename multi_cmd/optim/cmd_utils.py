import torch
import torch.autograd as autograd
import time

from multi_cmd.optim import potentials

def zero_grad(params):
    """Given some list of Tensors, zero and reset gradients."""
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

def flatten_filter_none(grad_list, param_list,
                        detach=False,
                        neg=False,
                        device=torch.device('cpu')):
    """
    Given a list of Tensors with possible None values, returns single Tensor
    with None removed and flattened.
    """
    filtered = []
    for grad, param in zip(grad_list, param_list):
        if grad is None:
            filtered.append(torch.zeros(param.numel(), device=device, requires_grad=True))
        else:
            filtered.append(grad.contiguous().view(-1))

    result = torch.cat(filtered) if not neg else -torch.cat(filtered)

    # Use this only if higher order derivatives are not needed.
    if detach:
        result.detach_()

    return result

"""Normal metamatrix vector prodduct."""
def avp(
    hessian_loss_list,
    player_list,
    player_list_flattened,
    vector_list_flattened,
    bregman=potentials.squared_distance(1),
    transpose=False,
    device=torch.device('cpu'),
    verbose=True
):
    """
    :param hessian_loss_list: list of objective functions for hessian computation
    :param player_list: list of list of params for each player to compute gradients from
    :param player_list_flattened: list of flattened player tensors (without gradients)
    :param vector_list_flattened: list of flattened vectors for each player
    :param bregman: dictionary representing bregman potential to use
    :param transpose: compute product against transpose if set

    Computes right product of metamatrix with a vector of player vectors.
    """
    # TODO(jjma): add error handling and assertions
    # assert(len(hessian_loss_list) == len(player_list))
    # assert(len(hessian_loss_list) == len(vector_list))
    prod_list = [torch.zeros_like(v, device=device) for v in vector_list_flattened]

    for i, row_params in enumerate(player_list):
        for j, (col_params, vector_elem) in enumerate(zip(player_list, vector_list_flattened)):

            if i == j:
                # Diagonal element is the Bregman term.
                prod_list[i] += bregman['Dxx_vp'](player_list_flattened[i], vector_elem)
                continue

            # Otherwise, we construct our Hessian vector products. Variable
            # retain_graph must be set to true, or we cant compute multiple
            # subsequent Hessians any more.
            
            loss = hessian_loss_list[i] if not transpose else hessian_loss_list[j]

            if verbose:
                start_time = time.time()

            grad_raw = autograd.grad(loss, col_params,
                                     create_graph=True,
                                     retain_graph=True,
                                     allow_unused=True)
            
            if verbose:
                print("Grad took:", time.time() - start_time)
                
            grad_flattened = flatten_filter_none(grad_raw, col_params,
                                                 device=device)

            # Don't need any higher order derivatives, so create_graph = False.
            hvp_raw = autograd.grad(grad_flattened, row_params,
                                    grad_outputs=vector_elem,
                                    create_graph=False,
                                    retain_graph=True,
                                    allow_unused=True)
            hvp_flattened = flatten_filter_none(hvp_raw, row_params,
                                                device=device)

            prod_list[i] += hvp_flattened

    return prod_list

"""Antisymmetric matrix vector prodduct."""
def antivp(
    hessian_loss_list,
    player_list,
    player_list_flattened,
    vector_list_flattened,
    bregman=potentials.squared_distance(1),
    transpose=False,
    device=torch.device('cpu')
):
    """
    :param hessian_loss_list: list of objective functions for hessian computation
    :param player_list: list of list of params for each player to compute gradients from
    :param player_list_flattened: list of flattened player tensors (without gradients)
    :param vector_list_flattened: list of flattened vectors for each player
    :param bregman: dictionary representing bregman potential to use
    :param transpose: compute product against transpose if set

    Computes right product of antisymmetric metamatrix with a vector of player vectors.
    """
    # TODO(jjma): add error handling and assertions
    # assert(len(hessian_loss_list) == len(player_list))
    # assert(len(hessian_loss_list) == len(vector_list))
    prod_list = [torch.zeros_like(v, device=device) for v in vector_list_flattened]

    for i, row_params in enumerate(player_list):
        for j, (col_params, vector_elem) in enumerate(zip(player_list, vector_list_flattened)):
            if i == j:
                # Diagonal element is the Bregman term.
                prod_list[i] += bregman['Dxx_vp'](player_list_flattened[i], vector_elem)
                continue

            # Otherwise, we construct our Hessian vector products. Variable
            # retain_graph must be set to true, or we cant compute multiple
            # subsequent Hessians any more.
            left_loss, right_loss = hessian_loss_list[i], hessian_loss_list[j]
            if transpose:
                left_loss, right_loss = right_loss, left_loss

            # Anti-symmetric decomposition (1/2)(A - A^T)...
            left_grad_raw = autograd.grad(left_loss, col_params,
                                          create_graph=True,
                                          retain_graph=True,
                                          allow_unused=True)
            left_grad_flattened = flatten_filter_none(left_grad_raw, col_params,
                                                      device=device)

            left_hvp_raw = autograd.grad(left_grad_flattened, row_params,
                                         grad_outputs=vector_elem,
                                         create_graph=False,
                                         retain_graph=True,
                                         allow_unused=True)
            left_hvp_flattened = flatten_filter_none(left_hvp_raw, row_params,
                                                     device=device)

            right_grad_raw = autograd.grad(right_loss, col_params,
                                           create_graph=True,
                                           retain_graph=True,
                                           allow_unused=True)
            right_grad_flattened = flatten_filter_none(right_grad_raw, col_params,
                                                       device=device)

            right_hvp_raw = autograd.grad(right_grad_flattened, row_params,
                                          grad_outputs=vector_elem,
                                          create_graph=False,
                                          retain_graph=True,
                                          allow_unused=True)
            right_hvp_flattened = flatten_filter_none(right_hvp_raw, row_params,
                                                      device=device)

            prod_list[i] += 0.5 * (left_hvp_flattened - right_hvp_flattened)

    return prod_list


def metamatrix_conjugate_gradient(
    grad_loss_list,
    hessian_loss_list,
    player_list,
    player_list_flattened,
    vector_list_flattened=None,
    bregman=potentials.squared_distance(1),
    mvp=avp,
    n_steps=20,
    tol=1e-6,
    atol=1e-6,
    device=torch.device('cpu')
):
    """
    :param grad_loss_list: list of loss tensors for each player to compute gradient
    :param hessian_loss_list: list of loss tensors for each player to compute hessian
    :param player_list: list of list of params for each player to compute gradients from
    :param player_list_flattened: list of flattened player tensors (without gradients)
    :param vector_list_flattened: initial guess for update solution
    :param bregman: dict representing a Bregman potential to be used
    :param n_steps: number of iteration steps for conjugate gradient
    :param tol: relative residual tolerance threshold from initial vector guess
    :param atol: absolute residual tolerance threshold

    Compute solution to meta-matrix game form using preconditioned conjugate
    gradient method. Since the metamatrix A is not p.s.d, we multiply both sides
    by the transpose to ensure p.s.d.

    In other words, note that solving Ax = b (where A is meta matrix, x is
    vector of update vectors and b is learning rate times vector of gradients
    is the same as solving A'x = b' (where A' = (A^T)A and b' = (A^T)b.
    """

    b = []
    for loss, param_tensors in zip(grad_loss_list, player_list):
        # Get vector list of negative gradients.
        grad_raw = autograd.grad(loss, param_tensors,
                                 retain_graph=True,
                                 allow_unused=True)
        grad_flattened = flatten_filter_none(grad_raw, param_tensors,
                                             neg=True, detach=True, device=device)
        b.append(grad_flattened)

    # Multiplying both sides by transpose to ensure p.s.d.
    # r = A^t * b (before we subtract)
    r = mvp(hessian_loss_list, player_list, player_list_flattened, b,
            bregman=bregman, transpose=True, device=device)

    # Set relative residual threshold based on norm of b.
    norm_At_b = sum(torch.dot(r_elem, r_elem) for r_elem in r)
    residual_tol = tol * norm_At_b

    # If no guess provided, start from zero vector.
    if vector_list_flattened is None:
        vector_list_flattened = [torch.zeros_like(p, device=device)
                                 for p in player_list_flattened]
    else:
        # Compute initial residual if a guess is given.
        A_x = mvp(hessian_loss_list, player_list, player_list_flattened, vector_list_flattened,
                  bregman=bregman, transpose=False, device=device)
        At_A_x = mvp(hessian_loss_list, player_list, player_list_flattened, A_x,
                     bregman=bregman, transpose=True, device=device)
        
        # torch._foreach_sub_(r, At_A_x)
        for r_elem, At_A_x_elem in zip(r, At_A_x):
            r_elem -= At_A_x_elem
      

    # Use preconditioner if available...
    z = r
    if 'Dxx_inv_vp' in bregman:
        z = [bregman['Dxx_inv_vp'](params, r_elems)
             for params, r_elems in zip(player_list_flattened, r)]

    # Early exit if solution already found.
    rdotr = sum(torch.dot(r_elem, r_elem) for r_elem in r)
    rdotz = sum(torch.dot(r_elem, z_elem) for r_elem, z_elem in zip(r, z))
    if rdotr < residual_tol or rdotr < atol:
        return vector_list_flattened, 0, rdotr

    # Define p and measure current candidate vector.
    p = [z_elem.clone().detach() for z_elem in z]

    # Use conjugate gradient to find vector solution.
    for i in range(n_steps):
        A_p = mvp(hessian_loss_list, player_list, player_list_flattened, p,
                  bregman=bregman, transpose=False, device=device)
        At_A_p = mvp(hessian_loss_list, player_list, player_list_flattened, A_p,
                     bregman=bregman, transpose=True, device=device)

        with torch.no_grad():
            alpha = torch.div(rdotz, sum(torch.dot(e1, e2) for e1, e2 in zip(p, At_A_p)))

            # Update candidate solution and residual, where:
            # (1) x_new = x + alpha * p
            # (2) r_new = r - alpha * A' * p

            # torch._foreach_add_(vector_list_flattened, p, alpha=alpha)
            # torch._foreach_sub_(r, At_A_p, alpha=alpha)
            
            for vlf_elem, p_elem in zip(vector_list_flattened, p):
                vlf_elem += p_elem * alpha
            for r_elem, At_A_P_elem in zip(r, At_A_p):
                r_elem -= At_A_P_elem * alpha
            
            # Calculate new residual metric
            new_rdotr = sum(torch.dot(r_elem, r_elem) for r_elem in r)

            # Break if solution is within threshold
            if new_rdotr < atol or new_rdotr < residual_tol:
                break

            # If preconditioner provided, use it...
            z = r
            if 'Dxx_inv_vp' in bregman:
                z = [bregman['Dxx_inv_vp'](params, r_elems)
                     for params, r_elems in zip(player_list_flattened, r)]
            new_rdotz = sum(torch.dot(r_elem, z_elem) for r_elem, z_elem in zip(r, z))

            # Otherwise, update and continue.
            # (3) p_new = r_new + beta * p
            
            # torch._foreach_add(z, p, alpha=beta)
            beta = torch.div(new_rdotz, rdotz)
            p = [z_elem + p_elem * beta for z_elem, p_elem in zip(z, p)]

            rdotr = new_rdotr
            rdotz = new_rdotz

    return vector_list_flattened, i+1, rdotr


def exp_map(player_list_flattened, nash_list_flattened,
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
        mapped = [bregman['Dx'](bregman['Dx_inv'](param) + bregman['Dxx_vp'](param, nash))
                  for param, nash in zip(player_list_flattened, nash_list_flattened)]

    return mapped


# TODO(jjma): make this user interface cleaner.
class CMD(object):
    """Optimizer class for the CMD algorithm with differentiable player objectives."""
    def __init__(self, player_list,
                 antisymetric=False,
                 bregman=potentials.squared_distance(1),
                 tol=1e-4, atol=1e-5,
                 n_steps=10,
                 device=torch.device('cpu')
    ):
        """
        :param player_list: list (per player) of list of Tensors, representing parameters
        :param bregman: dict representing Bregman potential to be used
        """
        self.bregman = bregman

        # In case, parameter generators are provided.
        player_list = [list(elem) for elem in player_list]

        # Conjugate gradient will provably converge in number of params steps.
        if n_steps is None:
            n_steps = sum([sum([elem.numel() for elem in param_list])
                           for param_list in player_list])

        # Store optimizer state.
        self.state = {'step': 0,
                      'player_list': player_list,
                      'tol': tol, 'atol': atol,
                      'n_steps': n_steps,
                      'last_dual_soln': None,
                      'last_dual_soln_n_iter': 0,
                      'last_dual_residual': 0.,
                      'antisymetric': antisymetric}
        # TODO(jjma): set this device in CMD algorithm.
        self.device = device
        self.mvp = avp if not antisymetric else antivp

    def zero_grad(self):
        for player in self.state['player_list']:
            zero_grad(player)

    def state_dict(self):
        return self.state

    def player_list(self):
        return self.state['player_list']

    def step(self, loss_list):
        # Compute flattened player list for some small optimization.
        player_list = self.state['player_list']
        player_list_flattened = [flatten_filter_none(player, player,
                                                     detach=True, device=self.device)
                                 for player in player_list]

        # Compute dual solution first, before mapping back to primal.
        # Use dual solution as initial guess for numerical speed.
        nash_list_flattened, n_iter, res = metamatrix_conjugate_gradient(
            loss_list,
            loss_list,
            player_list,
            player_list_flattened,
            vector_list_flattened=self.state['last_dual_soln'],
            mvp=self.mvp,
            bregman=self.bregman,
            tol=self.state['tol'],
            atol=self.state['atol'],
            n_steps=self.state['n_steps'],
            device=self.device
        )

        # Store state for use in next nash computation..
        self.state['step'] += 1
        self.state['last_dual_soln'] = nash_list_flattened
        self.state['last_dual_soln_n_iter'] = n_iter
        self.state['last_dual_residual'] = res

        # Map dual solution back into primal space.
        mapped_list_flattened = exp_map(player_list_flattened,
                                        nash_list_flattened,
                                        bregman=self.bregman)

        # Update parameters in place to update players as optimizer.
        for player, mapped_flattened in zip(self.state['player_list'], mapped_list_flattened):
            idx = 0
            for p in player:
                p.data = mapped_flattened[idx: idx + p.numel()].reshape(p.shape)
                idx += p.numel()

# TODO(jjma): May need to fix this optimizer for self-play, specifically update
# method, since we need to update data and not just replace it. If this were
# self play, only one of the updates calculated from nash would carry through.
class CMD_RL(CMD):
    """RL optimizer using CMD algorithm, using derivation from CoPG paper."""
    def __init__(self, player_list,
                 bregman=potentials.squared_distance(1),
                 antisymetric=False,
                 tol=1e-8, atol=1e-6,
                 n_steps=None,
                 device=torch.device('cpu')
                ):
        """
        :param player_list: list (per player) of list of Tensors, representing parameters
        :param bregman: dict representing Bregman potential to be used
        """
        super(CMD_RL, self).__init__(player_list,
                                     bregman=bregman,
                                     antisymetric=antisymetric,
                                     tol=tol, atol=atol,
                                     n_steps=n_steps,
                                     device=device)

    def step(self, grad_loss_list, hessian_loss_list, cgd=False):
        """
        CMD algorithm using derivation for gradient and hessian term from CoPG.
        """
        # Compute flattened player list for some small optimization.
        player_list = self.state['player_list']
        player_list_flattened = [flatten_filter_none(player, player,
                                                     detach=True, device=self.device)
                                 for player in player_list]

        # Compute dual solution first, before mapping back to primal.
        # Use dual solution as initial guess for numerical speed.
        nash_list_flattened, n_iter, res = metamatrix_conjugate_gradient(
            grad_loss_list,
            hessian_loss_list,
            player_list,
            player_list_flattened,
            vector_list_flattened=self.state['last_dual_soln'],
            mvp=self.mvp,
            bregman=self.bregman,
            tol=self.state['tol'],
            atol=self.state['atol'],
            n_steps=self.state['n_steps'],
            device=self.device
        )

        # Store state for use in next nash computation..
        self.state['step'] += 1
        self.state['last_dual_soln'] = nash_list_flattened
        self.state['last_dual_soln_n_iter'] = n_iter
        self.state['last_dual_residual'] = res

        # Edge case to enable self play in CGD case (since we can compute
        # element-wise in place operations in CGD).
        if (cgd):
            for player, nash_flattened in zip(self.state['player_list'], nash_list_flattened):
                idx = 0
                for p in player:
                    p.data += nash_flattened[idx: idx + p.numel()].reshape(p.shape)
                    idx += p.numel()
            return

        # Map dual solution back into primal space.
        mapped_list_flattened = exp_map(player_list_flattened,
                                        nash_list_flattened,
                                        bregman=self.bregman)

        # Update parameters in place to update players as optimizer.
        for player, mapped_flattened in zip(self.state['player_list'], mapped_list_flattened):
            idx = 0
            for p in player:
                p.data = mapped_flattened[idx: idx + p.numel()].reshape(p.shape)
                idx += p.numel()
