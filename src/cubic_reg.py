from __future__ import division
import numpy as np
import scipy.linalg


class CubicRegularization():
    def __init__(self, x0, f=None, gradient=None, hessian=None, L=None, L0=None, kappa_easy=0.0001, maxiter=1000, conv_tol=1e-6, epsilon=2*np.sqrt(np.finfo(float).eps)):
        """

        :param x0: Starting point for cubic regularization algorithm
        :param f: Function to be minimized
        :param gradient: Gradient of f (input as a function)
        :param hessian: Hessian of f (input as a function)
        :param L: Lipschitz constant on the Hessian
        :param L0:
        :param kappa_easy:
        :param maxiter: Maximum number of cubic regularization iterations
        :param conv_tol:
        :param epsilon:
        """
        self.f = f
        self.gradient = gradient
        self.hessian = hessian
        self.x0 = x0
        self.maxiter = maxiter
        self.conv_tol = conv_tol # Convergence tolerance
        self.epsilon = epsilon # Sqrt(machine precision)
        self.L = L
        self.L0 = L0
        self.kappa_easy = kappa_easy
        self.n = len(x0)

        self._check_inputs()
        # Estimate the gradient, hessian, and find a lower bound L0 for L if necessary
        if gradient is None:
            self.gradient = self.approx_grad
        if hessian is None:
            self.hessian = self.approx_hess
        if L0 is None and L is None:
            self.L0 = np.linalg.norm(self.hessian(self.x0)-self.hessian(self.x0+np.ones_like(self.x0)), ord=2)/np.linalg.norm(np.ones_like(self.x0))+self.epsilon
        self.lambda_nplus = self._compute_lambda_nplus(self.x0)

    def _check_inputs(self):
        """
        Ensure that the inputs are of the right form and all necessary inputs have been supplied
        """
        if not isinstance(self.x0, (tuple, list, np.ndarray)):
            raise TypeError('Invalid input type for x0')
        if len(self.x0) < 1:
            raise ValueError('x0 must have length > 0')
        if not (self.f is not None or (self.gradient is not None and self.hessian is not None and self.L is not None)):
            raise AttributeError('You must specify f and/or each of the following: gradient, hessian, and L')
        if not((not self.L or self.L > 0)and (not self.L0 or self.L0 > 0) and self.kappa_easy > 0 and self.maxiter > 0 and self.conv_tol > 0 and self.epsilon > 0):
            raise ValueError('All inputs that are constants must be larger than 0')
        if self.f is not None:
            try:
                self.f(self.x0)
            except TypeError:
                raise TypeError('x0 is not a valid input to function f')
        if self.gradient is not None:
            try:
                self.gradient(self.x0)
            except TypeError:
                raise TypeError('x0 is not a valid input to the gradient. Is the gradient a function with input dimension length(x0)?')
        if self.hessian is not None:
            try:
                self.hessian(self.x0)
            except TypeError:
                raise TypeError('x0 is not a valid input to the hessian. Is the hessian a function with input dimension length(x0)?')

    @staticmethod
    def _std_basis(size, idx):
        """
        Compute the idx'th standard basis vector
        :param size: Length of the vector
        :param idx: Index of value 1 in the vector
        :return: ei: Standard basis vector with 1 in the idx'th position
        """
        ei = np.zeros(size)
        ei[idx] = 1
        return ei

    def approx_grad(self, x):
        """
        Approximate the gradient of the function self.f at x
        :param x: Point at which the gradient will be approximated
        :return: Estimated gradient at x
        """
        return np.asarray([(self.f(x + self.epsilon * self._std_basis(self.n, i)) - self.f(x - self.epsilon * self._std_basis(self.n, i))) / (2 * self.epsilon) for i in range(0, self.n)])

    def approx_hess(self, x):
        """
        Approximate the hessian of the function self.x at x
        :param x: Point at which the Hessian will be approximated
        :return: Estimated Hessian at x
        """
        grad_x0 = self.gradient(x)
        hessian = np.zeros((self.n, self.n))
        for j in range(0, self.n):
            grad_x_plus_eps = self.gradient(x + self.epsilon * self._std_basis(self.n, j))
            for i in range(0, self.n):
                hessian[i,j] = (grad_x_plus_eps[i]-grad_x0[i])/self.epsilon
        return hessian

    def _compute_lambda_nplus(self, x):
        """
        Compute max(-1*smallest eigenvalue of hessian of f at x, 0)
        :param x: Point at which the hessian will be computed for use in max()
        :return: max(-1*smallest eigenvalue of hessian of f at x, 0)
        """
        lambda_n = scipy.linalg.eigh(self.hessian(x), eigvals_only=True, eigvals=(0, 0))
        return max(-lambda_n[0], 0)

    def _check_convergence(self, x_new):
        """
        Check whether the cubic regularization algorithm has converged using the criteria from Nesterov and Polyak (2006)
        :param x_new:
        :return:
        """
        if np.linalg.norm(self.gradient(x_new))**2 <= self.conv_tol:  # TODO change convergence criterion
            return True
        else:
            return False

    def cubic_reg(self):
        """
        Run the cubic regularization algorithm
        :return:
        """
        k = 0
        converged = False
        x_new = self.x0
        mk = self.L0
        intermediate_points = [x_new]
        while k < self.maxiter and converged is False:
            x_old = x_new
            x_new, mk = self._find_x_new(x_old, mk)
            self.lambda_nplus = self._compute_lambda_nplus(x_new)
            converged = self._check_convergence(x_new)
            intermediate_points.append(x_new)
        return x_new, intermediate_points

    def _find_x_new(self, x_old, mk):
        """
        Determine what M_k should be and compute the next point for the cubic regularization algorithm
        :param x_old: Previous point
        :param mk: Previous value of M_k (will start with this if L isn't specified)
        :return: x_new: New point
        :return: mk: New value of M_k
        """
        if self.L is not None:
            aux_problem = _AuxiliaryProblem(x_old, self.gradient, self.hessian, self.L, self.lambda_nplus, self.kappa_easy)
            x_new = aux_problem.solve()
            return x_new, self.L
        else:
            decreased = False
            while not decreased:
                mk *= 2
                aux_problem = _AuxiliaryProblem(x_old, self.gradient, self.hessian, mk, self.lambda_nplus, self.kappa_easy)
                x_new = aux_problem.solve()
                decreased = (self.f(x_new)-self.f(x_old) <= 0)
            mk = max(0.5 * mk, self.L0)
            return x_new, mk


class _AuxiliaryProblem:
    def __init__(self, x, gradient, hessian, M, lambda_nplus, kappa_easy):
        """

        :param x: Current point
        :param gradient: Gradient (as a function)
        :param hessian: Hessian (as a function)
        :param M:
        :param lambda_nplus: max(-1*smallest eigenvalue of hessian of f at x, 0)
        :param kappa_easy:
        """
        self.x = x
        self.gradient = gradient
        self.hessian = hessian
        self.M = M
        self.lambda_nplus = lambda_nplus
        self.kappa_easy = kappa_easy
        self.H_lambda = lambda x: self.hessian(self.x)+x*np.identity(np.size(self.hessian(self.x), 0))
        self.lambda_const = (1+self.lambda_nplus)*np.sqrt(np.finfo(float).eps)

    def _compute_s(self, lambduh):
        try:
            L = scipy.linalg.cholesky(self.H_lambda(lambduh))
        except:
            # See p. 516 of Solving the Trust-Region Problem using the Lanczos Method by Gould, Lucidi, Roma, Toint (1999)
            self.lambda_const *= 2
            s, L = self._compute_s(self.lambda_nplus + self.lambda_const)
        s = scipy.linalg.cho_solve((L, False), -self.gradient(self.x))
        return s, L

    def _update_lambda(self, lambduh, s, L):
        """

        :param lambduh:
        :param s:
        :param L:
        :return:
        """
        w = scipy.linalg.solve_triangular(L.T, s, lower=True)
        norm_s = np.linalg.norm(s)
        phi = 1/norm_s-self.M/(2*lambduh)
        phi_prime = np.linalg.norm(w)**2/(norm_s**3)+self.M/(2*lambduh**2)
        return lambduh - phi/phi_prime

    def _converged(self, s, lambduh):
        """
        Check whether the algorithm from the subproblem has converged
        :param s:
        :param lambduh:
        :return:
        """
        r = 2*lambduh/self.M
        if abs(np.linalg.norm(s)-r) <= self.kappa_easy:  # TODO choose better convergence criterion
            return True
        else:
            return False

    def solve(self):
        """
        Solve the cubic regularization subproblem.
        :return: s+self.x: Next point for the cubic regulatization algorithm
        """
        if self.lambda_nplus == 0:
            lambduh = 0
        else:
            lambduh = self.lambda_nplus + self.lambda_const
        s, L = self._compute_s(lambduh)
        r = 2*lambduh/self.M
        if np.linalg.norm(s) <= r:
            if lambduh == 0 or np.linalg.norm(s) == r:
                return s+self.x
            else:
                Lambda, U = np.linalg.eigh(self.H_lambda(self.lambda_nplus))
                s_cri = -U.T.dot(np.linalg.pinv(np.diag(Lambda))).dot(U).dot(self.gradient(self.x))
                alpha = max(np.roots([np.dot(U[:, 0], U[:, 0]), 2*np.dot(U[:, 0], s_cri), np.dot(s_cri, s_cri)-4*self.lambda_nplus**2/self.M**2]))
                s = s_cri + alpha*U[:, 0]
                return s+self.x
        if lambduh == 0:
            lambduh += self.lambda_const
        while not self._converged(s, lambduh):
            lambduh = self._update_lambda(lambduh, s, L)  # TODO fix this so it doesn't run forever if it doesn't converge
            s, L = self._compute_s(lambduh)
        return s+self.x
