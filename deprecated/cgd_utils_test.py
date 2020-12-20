import torch
import unittest
import cgd_utils

class TestCGDUtils(unittest.TestCase):
    
    def setUp(self):
        super(TestCGDUtils, self).setUp()
        
        self.x = torch.tensor([1.0, 1.0], requires_grad=True)
        self.y = torch.tensor([1.0, 1.0], requires_grad=True)
        self.z = torch.tensor([1.0, 1.0], requires_grad=True)

    def testTwoPlayerMetaMatrixProduct(self):
        """
        Test metamatrix product with respect to two players, which can be calculated
        analytically.
        """
        # Player objective functions are f(x,y) = (x^2)(y^2) = -g(x,y)
        x_loss = torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))
        y_loss = - torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))
        
        b1 = torch.tensor([1.0, 1.0])
        b2 = torch.tensor([1.0, 1.0])
        
        result1 = cgd_utils.avp([x_loss, y_loss], [self.x, self.y], [b1, b2], [1, 1], 
                            transpose=False, retain_graph=True)
        result2 = cgd_utils.avp([x_loss, y_loss], [self.x, self.y], [b1, b2], [1, 1], 
                            transpose=True, retain_graph=True)
        
        expected1 = [torch.tensor([9., 9.]), torch.tensor([-7., -7.])]
        expected2 = [torch.tensor([-7., -7.]), torch.tensor([9., 9.])]
        
        for a, b in zip(result1, expected1):
            self.assertTrue(torch.all(torch.eq(a, b)))
        
        for a, b in zip(result2, expected2):
            self.assertTrue(torch.all(torch.eq(a, b)))
            
    def testTwoPlayerConjugateGradient(self):
        """
        We test in the two player case with the following objective functions. Since the two 
        player case is the same as presented in the original CGD paper, this conjugate gradient 
        should return the same result as calculated by hand from the closed form solution in 
        the paper.
        """
        
        # Player objective functions are f(x,y) = (x^2)(y^2) = -g(x,y)
        x_loss = torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))
        y_loss = - torch.sum(torch.pow(self.x, 2)) * torch.sum(torch.pow(self.y, 2))
        
        soln, n_iter = cgd_utils.metamatrix_conjugate_gradient(
            [x_loss, y_loss], [self.x, self.y], lr_list=[0.01, 0.01])
        expected = [torch.tensor([-0.0429, -0.0429]), torch.tensor([0.0366, 0.0366])]
        
        for a, b in zip(soln, expected):
            self.assertTrue(torch.all(torch.isclose(a, b, atol=1e-03,)))

    # TODO(jjma): add tests for 3 player case
    
if __name__ == '__main__':
    unittest.main()