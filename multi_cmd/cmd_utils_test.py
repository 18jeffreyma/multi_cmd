import torch
import unittest
import time

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

        start = time.time()
        result1 = cmd_utils.avp([x_loss, y_loss], [[self.x], [self.y]], [self.x, self.y], [b1, b2],
                                    bregman=self.bregman,
                                    transpose=False)
        result2 = cmd_utils.avp([x_loss, y_loss], [[self.x], [self.y]], [self.x, self.y], [b1, b2],
                                    bregman=self.bregman,
                                    transpose=True)
        result3 = cmd_utils.atvp([x_loss, y_loss], [[self.x], [self.y]], [self.x, self.y], [b1, b2],
                                    bregman=self.bregman)

        expected1 = [torch.tensor([9., 9.]), torch.tensor([-7., -7.])]
        expected2 = [torch.tensor([-7., -7.]), torch.tensor([9., 9.])]


        for a, b in zip(result1, expected1):
            self.assertTrue(torch.all(torch.eq(a, b)))

        for a, b in zip(result2, expected2):
            self.assertTrue(torch.all(torch.eq(a, b)))

        for a, b in zip(result3, expected2):
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
            [x_loss, y_loss],
            [x_loss, y_loss],
            [[self.x], [self.y]],
            [self.x, self.y],
            bregman=self.bregman)

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

        player_list = [[self.x], [self.y]]
        optim = cmd_utils.CMD(player_list, bregman=self.bregman)
        optim.step(test_payoff(player_list))

        result = [elem[0] for elem in player_list]
        expected = [torch.tensor([1 - 0.5538, 1 - 0.5538]), torch.tensor([1 - 0.4308, 1 - 0.4308])]
        for a, b in zip(result, expected):
            self.assertTrue(torch.all(torch.isclose(a, b, atol=1e-03,)))


    def testTwoPlayerOptimShannonEntropyE2E(self):
        """
        Perform an E2E test that CMD algorithm is generally implemented correctly
        on a bilinear, two player objective.
        """
        # TODO(jjma): Clean up this test case.
        x_param = torch.tensor([0.3], requires_grad=True)
        y_param = torch.tensor([0.4], requires_grad=True)
        param_list = [[x_param], [y_param]]

        alpha = 1
        num_iter = 20

        x_player = []
        y_player = []

        # Testing on the example f(x,y) = a(x-0.1)(y-0.1) = -g(x, y)
        def payoff_func(param_list):
            x_loss = alpha * (torch.sum(param_list[0][0]) - 0.1) * (torch.sum(param_list[1][0]) - 0.1)
            y_loss = - alpha * (torch.sum(param_list[0][0]) - 0.1) * (torch.sum(param_list[1][0]) - 0.1)

            return [x_loss, y_loss]

        # Initialize optimizer.
        optim = cmd_utils.CMD(param_list, bregman=potentials.shannon_entropy(1))

        for n in range(num_iter):
            x_player.append(float(param_list[0][0].data[0]))
            y_player.append(float(param_list[1][0].data[0]))

            optim.step(payoff_func(param_list))

        # Values taken from a correct implementation.
        expected_x = [0.3, 0.2137, 0.1494, 0.1046, 0.0743, 0.0539, 0.0402, 0.0307,
                      0.0242, 0.0195, 0.0161, 0.0137, 0.0119, 0.0105, 0.0095, 0.0087,
                      0.0082, 0.0077, 0.0074, 0.0072]
        expected_y = [0.4, 0.4413, 0.458, 0.4563, 0.4423, 0.4209, 0.3956, 0.3687,
                      0.3419, 0.3152, 0.2897, 0.2657, 0.2433, 0.2224, 0.2032, 0.1855,
                      0.1692, 0.1543, 0.1406, 0.1282]

        # Check that values are correct.
        for actual, expected in zip(x_player, expected_x):
            self.assertAlmostEqual(actual, expected, places=3)

        for actual, expected in zip(y_player, expected_y):
            self.assertAlmostEqual(actual, expected, places=3)

    # TODO(jjma): add tests for 3 player case

if __name__ == '__main__':
    unittest.main()
