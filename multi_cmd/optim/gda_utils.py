import torch
import torch.autograd as autograd

def zero_grad(params):
    """Given some list of Tensors, zero and reset gradients."""
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

# TODO(jjma): make this user interface cleaner.
class SGD(object):
    """Optimizer class for simultaneous SGD"""
    def __init__(self, player_list,
                 lr_list=None,
                 device=torch.device('cpu')
                ):
        """
        :param player_list: list (per player) of list of Tensors, representing parameters
        :param lr_list: list of learning rates per player optimizer.
        """
        # Store optimizer state.
        player_list = [list(elem) for elem in player_list]
        self.state = {'step': 0,
                      'player_list': player_list,
                      'lr_list': lr_list}
        # TODO(jjma): set this device in CMD algorithm.
        self.device = device

    def zero_grad(self):
        for player in self.state['player_list']:
            zero_grad(player)

    def state_dict(self):
        return self.state

    def step(self, loss_list):
        print('step')
        grad_list = [
            autograd.grad(loss, player, retain_graph=True)
            for loss, player in zip(loss_list, self.state['player_list'])
        ]

        for grad, player, lr in zip(grad_list, self.state['player_list'], self.state['lr_list']):
            for player_elem, grad_elem in zip(player, grad):
                player_elem.data -= grad_elem * lr

            # torch._foreach_add_(player, grad, alpha=-lr)
