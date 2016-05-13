import unittest2 as unittest
import numpy as np

import src.cubic_reg


class TestGradHess(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: x[0]**2*x[1]**2 + x[0]**2 + x[1]**2
        self.grad = lambda x: [2*x[0]*x[1]**2+2*x[0], 2*x[0]**2*x[1]+2*x[1]]
        self.hess = lambda x: np.asarray([[2*x[1]**2+2, 4*x[0]*x[1]], [4*x[0]*x[1], 2*x[0]**2+2]])
        self.x0 = [1,2]
        self.cr = src.cubic_reg.CubicRegularization(self.x0, self.f)

    def test_gradient(self):
        self.assertAlmostEqual(self.grad(self.x0)[0], self.cr.gradient(self.x0)[0], places=5)
        self.assertAlmostEqual(self.grad(self.x0)[1], self.cr.gradient(self.x0)[1], places=5)

    def test_hessian(self):
        self.assertAlmostEqual(self.hess(self.x0)[0,0], self.cr.hessian(self.x0)[0,0], places=5)
        self.assertAlmostEqual(self.hess(self.x0)[0,1], self.cr.hessian(self.x0)[0,1], places=5)
        self.assertAlmostEqual(self.hess(self.x0)[1,0], self.cr.hessian(self.x0)[1,0], places=5)
        self.assertAlmostEqual(self.hess(self.x0)[1,1], self.cr.hessian(self.x0)[1,1], places=5)





