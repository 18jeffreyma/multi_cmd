import torch


def squared_distance(alpha):
    # This recovers our original CGD algorithm.
    def Dx(x_dual):
        return x_dual * alpha
    
    def Dx_inv(x_dual):
        return x_dual / alpha
    
    def Dxx_vp(x_primal, vec):
        # Hessian is just the identity matrix times 1/alpha.
        return vec / alpha 
    
    return {'Dx': Dx, 'Dx_inv': Dx_inv, 'Dxx_vp': Dxx_vp}
    
    
def shannon_entropy(alpha):
    def Dx(x_dual):
        return torch.exp(x_dual * alpha)
    
    def Dx_inv(x_primal):
        return torch.log(x_primal) / alpha
    
    def Dxx_vp(x_primal, vec):
        # Hessian is just the identity matrix times alpha.
        return vec / x_primal / alpha
    
    return {'Dx': Dx, 'Dx_inv': Dx_inv, 'Dxx_vp': Dxx_vp}
