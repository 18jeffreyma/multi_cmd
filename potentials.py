import torch


def squared_distance(alpha):
    # This recovers our original CGD algorithm.
    def Dx(x_dual):
        return x_dual / alpha
    
    def Dx_inv(x_dual):
        return x_dual * alpha
    
    def Dxx_vp(x_primal, vec):
        # Does not need to be in-place.
        return vec * alpha 
    
    return {'Dx': Dx, 'Dx_inv': Dx_inv, 'Dxx_vp': Dxx_vp}
    
    
def shannon_entropy(alpha):
    def Dx(x_dual):
        return torch.exp(x_dual / alpha)
    
    def Dx_inv(x_primal):
        return alpha * torch.log(x_primal)
    
    def Dxx_vp(x_primal, vec):
        # Does not need to be in-place.
        return vec / x_primal * alpha
    
    return {'Dx': Dx, 'Dx_inv': Dx_inv, 'Dxx_vp': Dxx_vp}
