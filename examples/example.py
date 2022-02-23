import matplotlib.pyplot as plt
import numpy as np

import src.cubic_reg


class Function:
    def __init__(self, function='bimodal', method='adaptive', hessian_update='broyden'):
        self.method = method
        if function == 'bimodal':
            self.f = lambda x: -(x[0] ** 2 + 3*x[1] ** 2)*np.exp(1-x[0] ** 2 - x[1] ** 2)
            self.grad = None
            self.hess = None
            self.x0 = [1, 0]  # Start at saddle point!
        elif function == 'simple':
            self.f = lambda x: x[0]**2*x[1]**2 + x[0]**2 + x[1]**2
            self.grad = lambda x: np.asarray([2*x[0]*x[1]**2 + 2*x[0], 2*x[0]**2*x[1] + 2*x[1]])
            self.hess = lambda x: np.asarray([[2*x[1]**2 + 2, 4*x[0]*x[1]], [4*x[0]*x[1], 2*x[0]**2 + 2]])
            self.x0 = np.array([1, 2])
        elif function == 'quadratic':
            self.f = lambda x: x[0]**2+x[1]**2
            self.grad = lambda x: np.asarray([2*x[0], 2*x[1]])
            self.hess = lambda x: np.asarray([[2, 0], [0, 2]])*1.0
            self.x0 = [2, 2]
        if self.method == 'adaptive':
            self.cr = src.cubic_reg.AdaptiveCubicReg(self.x0, f=self.f, gradient=self.grad, hessian=self.hess,
                                                     hessian_update_method=hessian_update, conv_tol=1e-4)
        else:
            self.cr = src.cubic_reg.CubicRegularization(self.x0, f=self.f, gradient=self.grad, hessian=self.hess,
                                                        conv_tol=1e-4)

    def run(self):
        if self.method == 'adaptive':
            x_opt, intermediate_points, n_iter, flag = self.cr.adaptive_cubic_reg()
        else:
            x_opt, intermediate_points, n_iter, flag = self.cr.cubic_reg()
        return x_opt, intermediate_points, n_iter

    def plot_points(self, intermediate_points):
        xlist = np.linspace(-3.0, 3.0, 50)
        ylist = np.linspace(-3.0, 3.0, 50)
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.zeros_like(X)
        for i in range(0, len(X)):
            for j in range(0, len(X)):
                Z[i, j] = self.f((X[i, j], Y[i, j]))
        points = np.asarray(intermediate_points)
        plt.clf()
        cp = plt.contour(X, Y, Z)
        plt.clabel(cp, inline=True, fontsize=10)
        plt.plot(points[:, 0], points[:, 1])
        plt.title('Contour plot of function and path of cubic regularization algorithm')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.show()


if __name__ == '__main__':
    # Choose a function to run it on, and a method to use (original cubic reg or adaptive cubic reg)
    # Function choices: 'bimodal', 'simple', 'quadratic'
    # Method choices: 'adaptive', 'original'
    # If you choose method='adaptive', you can choose hessian updates from 'broyden', 'rank_one', and 'exact'.
    bm = Function(function='bimodal', method='adaptive', hessian_update='broyden')
    # Run the algorithm on the function
    x_opt, intermediate_points, n_iter = bm.run()
    print('Argmin of function:', x_opt)
    # Plot the path of the algorithm
    bm.plot_points(intermediate_points)
