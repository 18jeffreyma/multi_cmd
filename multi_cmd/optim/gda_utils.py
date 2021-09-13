import torch
import torch.autograd as autograd

def zero_grad(params):
    """Given some list of Tensors, zero and reset gradients."""
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

<<<<<<< HEAD
# TODO(jjma): make this user interface cleaner.
=======
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
        result.detach()

    return result

# TODO(anonymous): make this user interface cleaner.
>>>>>>> f6389a590292eab5a39dbdd2ee6b8b91cf73c6de
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
        # TODO(anonymous): set this device in CMD algorithm.
        self.device = device

    def zero_grad(self):
        for player in self.state['player_list']:
            zero_grad(player)

    def state_dict(self):
        return self.state

    def step(self, loss_list):
        print('step')
        grad_list = [
            autograd.grad(loss, player, retain_graph=True, allow_unused=True)
            for loss, player in zip(loss_list, self.state['player_list'])
        ]

        print("largest gradient:", max(torch.max(tensor) for tensor in grad_list[0]))
        print("largest gradient:", max(torch.max(tensor) for tensor in grad_list[1]))
        print("largest gradient:", max(torch.max(tensor) for tensor in grad_list[2]))



        for grad, player, lr in zip(grad_list, self.state['player_list'], self.state['lr_list']):
            for player_elem, grad_elem in zip(player, grad):
                if grad_elem is not None:
                    player_elem.data -= grad_elem * lr

            # torch._foreach_add_(player, grad, alpha=-lr)
