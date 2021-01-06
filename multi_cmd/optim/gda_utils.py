import torch


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

        # Create new SGD optimizer per player.
        optim_list = []
        for player, lr in zip(player_list, lr_list):
            optim_list.append(torch.optim.SGD(player, lr=lr))

        # Store optimizer state.
        self.state = {'step': 0,
                      'optim_list': optim_list}
        # TODO(jjma): set this device in CMD algorithm.
        self.device = device

    def zero_grad(self):
        for optim in self.state['optim_list']:
            optim.zero_grad()

    def state_dict(self):
        return self.state

    def step(self, loss_list):
        for optim, loss in zip(self.state['optim_list'], loss_list):
            optim.zero_grad()
            loss.backward()
            optim.step()


# TODO(jjma): Write a formal optimizer for CGD.
