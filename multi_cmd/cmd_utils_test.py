import torch
import unittest

from multi_cmd import cmd_utils
from multi_cmd import potentials

class TestCGDUtils(unittest.TestCase):

    def setUp(self):
        super(TestCGDUtils, self).setUp()

        self.x = torch.tensor([1.0, 1.0], requires_grad=True)
        self.y = torch.tensor([1.0, 1.0], requires_grad=True)
        self.z = torch.tensor([1.0, 1.0], requires_grad=True)

        self.bregman = potentials.squared_distance(1)

    def testTwoPlayerMetaMatrixProductSqDist(self):
        """
        Test metamatrix product with respect to two players, which can be calculated
        analytically.
        """
        # Player objective functions are f(x,y) = (x^2)(y^2) = -g(x,y)
        x_loss = torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))
        y_loss = - torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))

        b1 = torch.tensor([1.0, 1.0])
        b2 = torch.tensor([1.0, 1.0])

        result1 = cmd_utils.avp([x_loss, y_loss], [self.x, self.y], [b1, b2], bregman=self.bregman,
                            transpose=False, retain_graph=True)
        result2 = cmd_utils.avp([x_loss, y_loss], [self.x, self.y], [b1, b2], bregman=self.bregman,
                            transpose=True, retain_graph=True)

        expected1 = expected1 = [torch.tensor([9., 9.]), torch.tensor([-7., -7.])]
        expected2 = [torch.tensor([-7., -7.]), torch.tensor([9., 9.])]

        for a, b in zip(result1, expected1):
            self.assertTrue(torch.all(torch.eq(a, b)))

        for a, b in zip(result2, expected2):
            self.assertTrue(torch.all(torch.eq(a, b)))

    def testTwoPlayerConjugateGradientSqDist(self):
        """
        We test in the two player case with the following objective functions. Since the two
        player case is the same as presented in the original CGD paper, this conjugate gradient
        should return the same result as calculated by hand from the closed form solution in
        the paper.
        """

        # Player objective functions are f(x,y) = (x^2)(y^2) = -g(x,y)
        x_loss = torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))
        y_loss = - torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))

        result, n_iter = cmd_utils.metamatrix_conjugate_gradient(
            [x_loss, y_loss], [self.x, self.y], bregman=self.bregman)
        expected = [torch.tensor([-0.5538, -0.5538]), torch.tensor([-0.4308, -0.4308])]

        for a, b in zip(result, expected):
            self.assertTrue(torch.all(torch.isclose(a, b, atol=1e-03,)))

    def testTwoPlayerExpMapSqDist(self):
        """
        For the squared distance Bregman potential, we test that our implementation is correct.
        """
        nash_list = [torch.tensor([-1., -1.]), torch.tensor([-1., -1.])]

        result = cmd_utils.exp_map([self.x, self.y], nash_list)
        expected = [torch.tensor([0., 0.]), torch.tensor([0, 0])]

        for a, b in zip(result, expected):
            self.assertTrue(torch.all(torch.eq(a, b)))


    def testTwoPlayerOptimSqDist(self):
        """
        For the squared distance Bregman potential, we test that our implementation is correct.
        """
        def test_payoff(param_list):
            return [
                torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2)),
                -torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))
            ]

        player_list = [self.x, self.y]
        optim = cmd_utils.CMD(player_list, bregman=self.bregman)
        optim.step(test_payoff(player_list))

        expected = [torch.tensor([1 - 0.5538, 1 - 0.5538]), torch.tensor([1 - 0.4308, 1 - 0.4308])]

        for a, b in zip(player_list, expected):
            self.assertTrue(torch.all(torch.isclose(a, b, atol=1e-03,)))

    # TODO(jjma): add tests for 3 player case

if __name__ == '__main__':
    unittest.main()
