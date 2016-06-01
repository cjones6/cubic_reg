import matplotlib.pyplot as plt
import numpy as np

import src.cubic_reg


class Function:
    def __init__(self, function='bimodal'):
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
        self.cr = src.cubic_reg.CubicRegularization(self.x0, f=self.f, gradient=self.grad, hessian=self.hess)

    def run(self):
        x_opt, intermediate_points, n_iter = self.cr.cubic_reg()
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
    bm = Function(function='bimodal')
    x_opt, intermediate_points, n_iter = bm.run()
    print 'Argmin of function:', x_opt
    bm.plot_points(intermediate_points)