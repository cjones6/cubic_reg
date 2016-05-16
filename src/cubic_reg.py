from __future__ import division
import numpy as np
import scipy.linalg


class CubicRegularization():
    def __init__(self, x0, f=None, gradient=None, hessian=None, L=None, L0=None, kappa_easy=0.0001, maxiter=1000, conv_tol=1e-6, epsilon=2*np.sqrt(np.finfo(float).eps)):
        """
        :param gradient: function that returns the gradient
        :param hessian: function that returns the hessian
        :return:
        """
        self.f = f
        self.x0 = x0
        self.maxiter = maxiter
        self.conv_tol = conv_tol # Convergence tolerance
        self.epsilon = epsilon # Sqrt(machine precision)
        self.L = L
        self.L0 = L0
        self.kappa_easy = kappa_easy
        self.n = len(x0)
        assert(f is not None or (gradient is not None and hessian is not None and L is not None))  # TODO convert to exception

        # Estimate the gradient, hessian, and find a lower bound L0 for L if necessary
        if gradient is not None:
            self.gradient = gradient
        else:
            self.gradient = self.approx_grad
        if hessian is not None:
            self.hessian = hessian
        else:
            self.hessian = self.approx_hess
        if L0 is None and L is None:
            self.L0 = np.linalg.norm(self.hessian(self.x0)-self.hessian(self.x0+np.ones_like(self.x0)), ord=2)/np.linalg.norm(np.ones_like(self.x0))+self.epsilon
        self.lambda_nplus = self.compute_lambda_nplus(self.x0)

    @staticmethod
    def std_basis(size, idx):
        ei = np.zeros(size)
        ei[idx] = 1
        return ei

    def approx_grad(self, x):
        return [(self.f(x+self.epsilon*self.std_basis(self.n, i))-self.f(x-self.epsilon*self.std_basis(self.n, i)))/(2*self.epsilon) for i in range(0, self.n)]

    def approx_hess(self, x):
        grad_x0 = self.gradient(x)
        hessian = np.zeros((self.n,self.n))
        for j in range(0, self.n):
            grad_x_plus_eps = self.gradient(x + self.epsilon*self.std_basis(self.n, j))
            for i in range(0, self.n):
                hessian[i,j] = (grad_x_plus_eps[i]-grad_x0[i])/self.epsilon
        return hessian

    def compute_lambda_nplus(self, x):
        lambda_n = scipy.linalg.eigh(self.hessian(x), eigvals_only=True, eigvals=(0, 0))
        return max(-lambda_n[0], 0)

    def check_convergence(self, x_new):
        if np.linalg.norm(self.gradient(x_new))**2 <= self.conv_tol:  # TODO change convergence criterion
            return True
        else:
            return False

    def cubic_reg(self):
        k = 0
        converged = False
        x_new = self.x0
        mk = self.L0
        intermediate_points = [x_new]
        while k < self.maxiter and converged is False:
            x_old = x_new
            x_new, mk = self.find_x_new(mk, x_old)
            self.lambda_nplus = self.compute_lambda_nplus(x_new)
            converged = self.check_convergence(x_new)
            intermediate_points.append(x_new)
        return x_new, intermediate_points

    def find_x_new(self, mk, x_old):
        if self.L is not None:
            aux_problem = AuxiliaryProblem(x_old, self.gradient, self.hessian, self.L, self.lambda_nplus, self.kappa_easy)
            x_new = aux_problem.solve()
            return x_new, self.L
        else:
            decreased = False
            while not decreased:
                mk *= 2
                aux_problem = AuxiliaryProblem(x_old, self.gradient, self.hessian, mk, self.lambda_nplus, self.kappa_easy)
                x_new = aux_problem.solve()
                decreased = (self.f(x_new)-self.f(x_old) <= 0)
            mk = max(0.5 * mk, self.L0)
            return x_new, mk


class AuxiliaryProblem():
    def __init__(self, x, gradient, hessian, M, lambda_nplus, kappa_easy):
        self.x = x
        self.gradient = gradient
        self.hessian = hessian
        self.M = M
        self.lambda_nplus = lambda_nplus
        self.kappa_easy = kappa_easy
        self.H_lambda = lambda x: self.hessian(self.x)+x*np.identity(np.size(self.hessian(self.x), 0))

    def eigendecomposition(self):
        eig_vals, V = np.linalg.eigh(self.hessian)
        self.eigenvalues, self.eigenvectors = eig_vals, V

    def change_basis(self):
        return np.linalg.solve(self.eigenvectors, self.gradient)

    def compute_s(self, lambduh):
        L = scipy.linalg.cholesky(self.H_lambda(lambduh))
        s = scipy.linalg.cho_solve((L, False), -self.gradient(self.x))
        return s, L

    def update_lambda(self, lambduh, s, L):
        w = scipy.linalg.solve_triangular(L.T, s, lower=True)
        norm_s = np.linalg.norm(s)
        phi = 1/norm_s-self.M/(2*lambduh)
        phi_prime = np.linalg.norm(w)**2/(norm_s**3)+self.M/(2*lambduh**2)
        return lambduh - phi/phi_prime

    def converged(self, s, lambduh):
        r = 2*lambduh/self.M
        if abs(np.linalg.norm(s)-r) <= self.kappa_easy:  # TODO choose better convergence criterion
            return True
        else:
            return False

    def compute_lambduh(self, lambda_const=0.00001):
        lambduh = 0 if self.lambda_nplus == 0 else self.lambda_nplus+lambda_const
        s, L = self.compute_s(lambduh)
        r = 2*lambduh/self.M
        if np.linalg.norm(s) <= r:
            if lambduh == 0 or np.linalg.norm(s) == r:
                hard_case = False
                return lambduh, s, hard_case
            else:
                raise NotImplementedError
        if lambduh == 0:
            lambduh += lambda_const
        while not self.converged(s, lambduh):
            lambduh = self.update_lambda(lambduh, s, L)  # TODO fix this so it doesn't run forever if it doesn't converge
            s, L = self.compute_s(lambduh)
        hard_case = False
        return lambduh, s, hard_case

    def solve(self):
        lambduh, s, hard_case = self.compute_lambduh()
        if not hard_case:
            h = np.linalg.solve(self.H_lambda(lambduh), -self.gradient(self.x))
            return h+self.x
        else:
            raise NotImplementedError




