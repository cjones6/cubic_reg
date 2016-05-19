import matplotlib.pyplot as plt
import numpy as np

import src.cubic_reg

class NiceFunction():
    def __init__(self):
        self.f = lambda x: x[0] ** 2 * x[1] ** 2 + x[0] ** 2 + x[1] ** 2
        self.grad = lambda x: np.asarray([2 * x[0] * x[1] ** 2 + 2 * x[0], 2 * x[0] ** 2 * x[1] + 2 * x[1]])
        self.hess = lambda x: np.asarray([[2 * x[1] ** 2 + 2, 4 * x[0] * x[1]], [4 * x[0] * x[1], 2 * x[0] ** 2 + 2]])
        self.x0 = [1, 2]
        self.cr = src.cubic_reg.CubicRegularization(self.x0, self.f, L=2, gradient=self.grad, hessian=self.hess)

    def run(self):
        x_new, intermediate_points = self.cr.cubic_reg()
        return x_new, intermediate_points

    def plot_points(self, intermediate_points):
        points = np.asarray(intermediate_points)
        plt.clf()
        plt.plot(points[:,0], points[:,1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        plt.savefig('x_vs_y_ex1.png')

    def plot_func_value(self, intermediate_points):
        f_vals = [self.f(intermediate_points[i]) for i in range(0, len(intermediate_points))]
        plt.clf()
        plt.plot(range(0, len(f_vals)), f_vals)
        plt.xlabel('Iteration')
        plt.ylabel('Function value')
        plt.savefig('function_vs_iter.png')


class BadFunction():
    def __init__(self):
        self.f = lambda x: -(x[0] ** 2 + 3*x[1] ** 2)*np.exp(1-x[0] ** 2 - x[1] ** 2)
        self.grad = None
        self.hess = None
        # self.grad = lambda x: np.asarray([2 * x[0] * x[1] ** 2 + 2 * x[0], 2 * x[0] ** 2 * x[1] + 2 * x[1]])
        # self.hess = lambda x: np.asarray([[2 * x[1] ** 2 + 2, 4 * x[0] * x[1]], [4 * x[0] * x[1], 2 * x[0] ** 2 + 2]])
        #self.x0 = [1, 0.5]
        self.x0 = [1, 0]
        self.cr = src.cubic_reg.CubicRegularization(self.x0, self.f, gradient=self.grad, hessian=self.hess)

    def run(self):
        x_new, intermediate_points = self.cr.cubic_reg()
        return x_new, intermediate_points

if __name__ == '__main__':
    #nf = NiceFunction()
    nf = BadFunction()
    x, ip = nf.run()
    # nf.plot_points(ip)
    # nf.plot_func_value(ip)
    with open('example_path.txt', 'w') as outfile:
        for i in range(0, np.size(ip, 0)):
            outstr = str(ip[i][0])+'\t'+str(ip[i][1])+'\n'
            outfile.write(outstr)