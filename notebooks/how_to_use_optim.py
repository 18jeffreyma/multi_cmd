# How to use PCGD optimizer.

# Import CMD and CMD RL optimizers.
from multi_cmd.optim.cmd_utils import SGD
from multi_cmd.optim.cmd_utils import CMD
from multi_cmd.optim.cmd_utils import CMD_RL

# See multi_cmd/optim/potential.py for Bregman potential implementations.
from multi_cmd.optim import potentials

# Initialize CMD optimizer (use this for cases where loss is differentiable).
# The CMD_RL optimizer differs from CMD in that the CMD_RL optimizer can have
# seperate objectives for gradient and hessian loss calculations

# List of players, where each player is a list of their parameter tensors.
player_list_cgd = [
    [torch.tensor([x], requires_grad=True)], 
    [torch.tensor([y], requires_grad=True)], 
    [torch.tensor([z], requires_grad=True)]
]
player_list_gda = [
    [torch.tensor([x], requires_grad=True)], 
    [torch.tensor([y], requires_grad=True)],
    [torch.tensor([z], requires_grad=True)]
]


# Simple linear price, linear cost game.
def player_payoffs(quantity_list):
    quantity_tensor = torch.stack(sum(quantity_list, []))
    price = torch.max(1. - torch.sum(quantity_tensor),
                      torch.tensor(0., requires_grad=True))
                      
    payoffs = []
    for i, quantity in enumerate(quantity_tensor):
        # Negative, since CGD minimizes player objectives.
        payoffs.append(- (quantity * price - 0.1 * quantity))
        
    return payoffs

# Initialize optimizers with parameters per player, as well as bregman potential for PCGD and LR for SimGD.
cgd_optim = cmd_utils.CMD(player_list_cgd, bregman=potentials.shannon_entropy(1/0.001), antisymetric=True)
gda_optim = gda_utils.SGD(player_list_gda, lr_list=[0.001, 0.001, 0.001])

# Update parameters by calling step with a list of calculated losses with gradients, one for each player.
num_iterations = 500
for i in range(num_iterations):
    cgd_payoffs = player_payoffs(player_list_cgd)
    cgd_optim.step(cgd_payoffs)
    
    gda_payoffs = player_payoffs(player_list_gda)
    gda_optim.step(gda_payoffs)

print('timesteps, lr:', (num_iterations, lr))
print('final cgd quantities:', [elem[0].detach() for elem in player_list_cgd])
print('final cgd payoffs:', [elem[0].detach() for elem in cgd_payoffs])

print('final gda quantities:', [elem[0].detach() for elem in player_list_gda])
print('final gda payoffs:', [elem[0].detach() for elem in gda_payoffs])

