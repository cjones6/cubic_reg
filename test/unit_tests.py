import unittest2 as unittest
import numpy as np

import src.cubic_reg


class TestInitializations(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: x[0]**2*x[1]**2 + x[0]**2 + x[1]**2
        self.grad = lambda x: np.asarray([2*x[0]*x[1]**2+2*x[0], 2*x[0]**2*x[1]+2*x[1]])
        self.hess = lambda x: np.asarray([[2*x[1]**2+2, 4*x[0]*x[1]], [4*x[0]*x[1], 2*x[0]**2+2]])
        self.x0 = [1,2]
        # self.cr = src.cubic_reg.CubicRegularization(self.x0, self.f)

    # def test_gradient(self):
    #     self.assertAlmostEqual(self.grad(self.x0)[0], self.cr.gradient(self.x0)[0], places=5)
    #     self.assertAlmostEqual(self.grad(self.x0)[1], self.cr.gradient(self.x0)[1], places=5)
    #
    # def test_hessian(self):
    #     self.assertAlmostEqual(self.hess(self.x0)[0,0], self.cr.hessian(self.x0)[0,0], places=5)
    #     self.assertAlmostEqual(self.hess(self.x0)[0,1], self.cr.hessian(self.x0)[0,1], places=5)
    #     self.assertAlmostEqual(self.hess(self.x0)[1,0], self.cr.hessian(self.x0)[1,0], places=5)
    #     self.assertAlmostEqual(self.hess(self.x0)[1,1], self.cr.hessian(self.x0)[1,1], places=5)

    def test_lambdaplus(self):
        self.cr = src.cubic_reg.CubicRegularization(self.x0, self.f, gradient=self.grad, hessian=self.hess)
        self.assertAlmostEqual(np.sqrt(73)-7, self.cr.compute_lambda_nplus(self.x0), places=10)


class TestSubproblem(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: x[0] ** 2 * x[1] ** 2 + x[0] ** 2 + x[1] ** 2
        self.grad = lambda x: np.asarray([2 * x[0] * x[1] ** 2 + 2 * x[0], 2 * x[0] ** 2 * x[1] + 2 * x[1]])
        self.hess = lambda x: np.asarray([[2 * x[1] ** 2 + 2, 4 * x[0] * x[1]], [4 * x[0] * x[1], 2 * x[0] ** 2 + 2]])
        self.x0 = [1, 2]
        self.cr = src.cubic_reg.CubicRegularization(self.x0, self.f, L=2, gradient=self.grad, hessian=self.hess)
        self.aux_problem = src.cubic_reg.AuxiliaryProblem(self.x0, self.cr.gradient, self.cr.hessian, self.cr.L, self.cr.lambda_nplus, self.cr.kappa_easy)

    def test_solution(self):
        xnew = self.aux_problem.solve()
        self.assertAlmostEqual(1.47073, xnew[0], places=4)
        self.assertAlmostEqual(0.0431533, xnew[1], places=4)


class TestCubicReg(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: x[0] ** 2 * x[1] ** 2 + x[0] ** 2 + x[1] ** 2
        self.grad = lambda x: np.asarray([2 * x[0] * x[1] ** 2 + 2 * x[0], 2 * x[0] ** 2 * x[1] + 2 * x[1]])
        self.hess = lambda x: np.asarray([[2 * x[1] ** 2 + 2, 4 * x[0] * x[1]], [4 * x[0] * x[1], 2 * x[0] ** 2 + 2]])
        self.x0 = [1, 2]
        self.cr = src.cubic_reg.CubicRegularization(self.x0, self.f, L=2, gradient=self.grad, hessian=self.hess)

    def test_cr(self):
        x_new, intermediate_points = self.cr.cubic_reg()
        self.assertAlmostEqual(0, x_new[0], places=4)
        self.assertAlmostEqual(0, x_new[1], places=4)