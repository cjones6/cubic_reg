from __future__ import division
import numpy as np
import scipy.linalg


class CubicRegularization():
    def __init__(self, x0, f=None, gradient=None, hessian=None, L=None, L0=None, kappa_easy=0.1, maxiter=1000, conv_tol=1e-5, epsilon=np.sqrt(np.finfo(float).eps)):
        """
        :param gradient: function that returns the gradient
        :param hessian: function that returns the hessian
        :return:
        """
        self.f = f
        self.x0 = x0
        self.maxiter = maxiter
        self.conv_tol = conv_tol # Convergence tolerance
        self.epsilon = epsilon # Machine precision
        self.L = L
        self.L0 = L0
        self.kappa_easy = kappa_easy
        self.n = len(x0)
        assert(f is not None or (gradient is not None and hessian is not None and L is not None))
        if gradient is not None:
            self.gradient = gradient
        else:
            self.gradient = self.approx_grad(f)
        if hessian is not None:
            self.hessian = hessian
        else:
            self.hessian = self.approx_hess(f)
        self.lambda_nplus = self.compute_lambda_nplus()

    @staticmethod
    def std_basis(size, idx):
        ei = np.zeros(size)
        ei[idx] = 1
        return ei

    def approx_grad(self, f):
        return lambda x: [(f(x+self.epsilon*self.std_basis(self.n, i))-f(x-self.epsilon*self.std_basis(self.n, i)))/(2*self.epsilon) for i in range(0, self.n)]

    def approx_hess(self, f):
        pass

    def compute_lambda_nplus(self):
        lambda_n = scipy.linalg.eigh(self.hessian, eigvals_only=True, eigvals=0)
        return max(-lambda_n, 0)

    def check_convergence(self, x_new):
        if np.linalg.norm(self.gradient(x_new))**2 <= self.conv_tol:
            return True
        else:
            return False

    def cubic_reg(self):
        k = 0
        converged = 0
        x_new = self.x0
        mk = self.L0
        while k < self.maxiter and converged is False:
            x_old = x_new
            x_new, mk = self.find_x_new(mk, x_old)
            converged = self.check_convergence(x_new)

    def find_x_new(self, mk, x_old):
        if self.L is not None:
            aux_problem = AuxiliaryProblem(x_old, self.gradient, self.hessian, self.L, self.lambda_nplus, self.kappa_easy)
            x_new = aux_problem.solve()
            return x_new
        else:
            raise(NotImplementedError)


class AuxiliaryProblem():
    def __init__(self, x, gradient, hessian, M, lambda_nplus, kappa_easy):
        self.x = x
        self.gradient = gradient
        self.hessian = hessian
        self.M = M
        self.lambda_nplus = lambda_nplus
        self.kappa_easy = kappa_easy
        self.H_lambda = lambda x: self.hessian+x*np.identity(np.size(self.hessian, 0))

    def eigendecomposition(self):
        eig_vals, V = np.linalg.eigh(self.hessian)
        self.eigenvalues, self.eigenvectors = eig_vals, V

    def change_basis(self):
        return np.linalg.solve(self.eigenvectors, self.gradient)

    def compute_s(self, lambduh):
        L = scipy.linalg.cholesky(self.H_lambda(lambduh))
        s = scipy.linalg.cho_solve((L, False), -self.gradient)
        return s, L

    def update_lambda(self, lambduh, s, L):
        w = scipy.linalg.solve_triangular(L, s)
        norm_s = np.linalg.norm(s)
        phi = 1/norm_s-self.M/(2*lambduh)
        phi_prime = np.linalg.norm(w)**2/(norm_s**3)-self.M/(2*lambduh**2)
        return lambduh - phi/phi_prime

    def converged(self, s, lambduh):
        r = 2*lambduh/self.M
        if abs(np.linalg.norm(s)-r) <= self.kappa_easy*r:
            return True
        else:
            return False

    def compute_lambduh(self):
        lambduh = 0 if self.lambda_nplus==0 else self.lambda_nplus+0.00001  # TODO not sure if the sign here is correct
        s, L = self.compute_s(lambduh)
        r = 2*lambduh/self.M
        if np.linalg.norm(s) <= r:
            if lambduh == 0 or np.linalg.norm(s) == r:
                hard_case = True
                return lambduh, s, hard_case
            else:
                raise NotImplementedError
        while not self.converged(s, lambduh):
            lambduh = self.update_lambda(lambduh, s, L)
            s, L = self.compute_s(lambduh)
        hard_case = False
        return lambduh, s, hard_case

    def solve(self):
        lambduh, s, hard_case = self.compute_lambduh()
        if not hard_case:
            h = np.linalg.solve(self.H_lambda(lambduh), -self.gradient)
            return h+self.x
        else:
            raise NotImplementedError




