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


def avp(
    loss_list,
    param_list,
    vector_list,
    lr_list=None,
    transpose=False,
    retain_graph=True,
    detach=True,
):
    """
    :param vector_list: list of vectors for each player
    :param loss_list: list of objective functions for each player
    :param param_list: list of parameter vectors for each player
    :param lr: list of learning rates for each player
    :param transpose: compute product against transpose if set
    :param retain_graph: save
    
    Computes left product of metamatrix with a vector of player vectors.
    """
    # TODO(jjma): add error handling and assertions
    assert(len(loss_list) == len(param_list))
    assert(len(loss_list) == len(vector_list))
    assert((lr_list is not None) and (len(loss_list) == len(lr_list)))
    
    prod_list = [torch.zeros(param.shape[0]) for param in param_list]
    lr_list = lr_list if lr_list else [1] * len(param_list)
    
    for i, row_param in enumerate(param_list):
        for j, (col_param, vector_elem) in enumerate(zip(param_list, vector_list)):
            if i == j:
                prod_list[i] += vector_elem
                continue

            loss = loss_list[i] if not transpose else loss_list[j]
                
            grad_param = autograd.grad(loss, col_param, 
                                       create_graph=retain_graph,
                                       retain_graph=retain_graph,
                                       allow_unused=True)
            grad_param_vec = torch.cat([g.contiguous().view(-1) for g in grad_param])
            
            if torch.isnan(grad_param_vec).any():
                raise ValueError('grad_param_vec nan')
            
            grad_vec_prod = torch.dot(grad_param_vec, vector_elem)
            hvp = autograd.grad(grad_vec_prod, row_param, 
                                retain_graph=retain_graph, 
                                allow_unused=True)
            hvp_vec = torch.cat([g.contiguous().view(-1) for g in hvp])
            
            if torch.isnan(hvp_vec).any():
                raise ValueError('hvp_vec nan')
            
            if detach:
                hvp_vec = hvp_vec.detach_()
            
            lr = lr_list[i] if not transpose else lr_list[j]
            prod_list[i] += hvp_vec * lr
    
    return prod_list

def metamatrix_conjugate_gradient(
    loss_list,
    param_list,
    vector_list=None,
    lr_list=None,
    n_steps=10,
    tol=1e-5,
    atol=1e-10
):
    """
    :param loss_list: list of loss tensors for each player
    :param param_list: list of player params to compute gradients from
    :param vector_list: initial guess for update solution
    :param lr_list: list of learning rates per player
    :param n_steps: number of iteration steps for conjugate gradient
    :param tol: relative residual tolerance threshold from initial vector guess
    :param tol: absolute residual tolerance threshold
    
    Compute solution to meta-matrix game form using conjugate gradient method. Since
    the metamatrix A is not p.s.d, we multiply both sides by the transpose to 
    ensure p.s.d.
    
    In other words, note that solving Ax = b (where A is meta matrix, x is 
    vector of update vectors and b is learning rate times vector of gradients 
    is the same as solving A'x = b' (where A' = (A^T)A and b' = (A^T)b.
    """
    lr_list = lr_list if lr_list else [1] * len(param_list)
       
    b = [-lr * autograd.grad(loss, param,
                             retain_graph=retain_graph,
                             allow_unused=True)
         for lr, loss, param in zip(lr_list, loss_list, param_list)]
    
    # Multiplying both sides by transpose to ensure p.s.d.
    # r = A^t * b (before we subtract)
    r = avp(loss_list, param_list, b, lr_list=lr_list, transpose=True)
    
    if vector_list is None:
        vector_list = [torch.zeros(param.shape[0]) for param in param_list]
       
    else:
        A_x = avp(loss_list, param_list, vector_list, 
                  lr_list=lr_list, transpose=False)
        At_A_x = avp(loss_list, param_list, A_x, 
                     lr_list=lr_list, transpose=True)
        
        r = vec_list_op(r, At_A_x, SUB_FUNC)
    
    # Define p and measure current candidate vector
    p = [r_elem.clone().detach() for r_elem in r]
    rdotr = vec_list_dot(r, r)
    
    # Set relative residual threshold
    residual_tol = tol * rdotr
    if rdotr < residual_tol or rdotr < atol:
        return vector_list, 1
    
    # Use conjugate gradient to find vector solution
    for i in range(n_steps):
        A_p = avp(loss_list, param_list, p, 
                  lr_list=lr_list, transpose=False)
        
        alpha = rdotr / vec_list_dot(p, A_p)
        
        alpha_mul = lambda x: alpha * x
        alpha_p = vec_list_map(p, alpha_mul)
        alpha_Ap = vec_list_map(A_p, alpha_mul)
        
        # Update candidate solution and residual
        vector_list = vec_list_op(vector_list, alpha_p, ADD_FUNC)
        r = vec_list_op(r, alpha_Ap, SUB_FUNC)
        
        new_rdotr = vec_list_dot(r, r)
        
        # Break if solution is within threshold
        if new_rdotr < atol or new_rdotr < residual_tol:
            break
        
        # Otherwise, update and continue
        beta = new_rdotr / rdotr
        beta_p = vec_list_map(p, lambda x: beta * x)
        
        p = vec_list_op(r, beta_p, ADD_FUNC)
        rdotr = new_rdotr
        
    return vector_list, i
        
       
    