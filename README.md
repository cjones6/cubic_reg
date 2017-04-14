# cubic_reg

This code implements two algorithms: 

1. Nesterov and Polyak's (2006) cubic regularization algorithm; and 
2. Cartis et al's (2011) adaptive cubic regularization algorithm.

Cubic regularization solves unconstrained minimization problems by minimizing a cubic upper bound to the function at each iteration.

See the example.py file for an example of how to use them and the comments in cubic_reg.py for all of the possible input options.
Briefly, you can load the file and numpy using
```
import numpy as np
import src.cubic_reg
```
specify your function, the gradient, Hessian, and initial point (the gradient and Hessian can be None)
```
f = lambda x: x[0] ** 2 * x[1] ** 2 + x[0] ** 2 + x[1] ** 2
grad = lambda x: np.asarray([2 * x[0] * x[1] ** 2 + 2 * x[0], 2 * x[0] ** 2 * x[1] + 2 * x[1]])
hess = lambda x: np.asarray([[2 * x[1] ** 2 + 2, 4 * x[0] * x[1]], [4 * x[0] * x[1], 2 * x[0] ** 2 + 2]])
x0 = np.array([1, 2]
```
and then use cubic regularization by running
```
cr = src.cubic_reg.CubicRegularization(x0, f=f, gradient=grad, hessian=hess, conv_tol=1e-4)
x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
```
To run adaptive cubic regularization instead, you can set
```
cr = src.cubic_reg.AdaptiveCubicReg(x0, f=f, gradient=grad, hessian=hess, hessian_update_method='broyden', conv_tol=1e-4)
x_opt, intermediate_points, n_iter, flag = cr.adaptive_cubic_reg()
```
There are many other options you can specify and parameters you can control.

References:
- Nesterov, Y., & Polyak, B. T. (2006). Cubic regularization of Newton method and its global performance.
  Mathematical Programming, 108(1), 177-205.
- Cartis, C., Gould, N. I., & Toint, P. L. (2011). Adaptive cubic regularisation methods for unconstrained optimization.
  Part I: motivation, convergence and numerical results. Mathematical Programming, 127(2), 245-295.
- Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods (Vol. 1). Siam.
- Gould, N. I., Lucidi, S., Roma, M., & Toint, P. L. (1999). Solving the trust-region subproblem using the Lanczos
  method. SIAM Journal on Optimization, 9(2), 504-525.

