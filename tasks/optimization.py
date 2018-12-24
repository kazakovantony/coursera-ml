# SciPy for minimization use gradient or non gradient method based on task - local optimization
# (not all tasks have gradient in each point) and final result


import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as op


def equation(x):
    return np.sin(x/5) * np.exp(x/10) + 5 * np.exp(-x/2)


z = np.linspace(1, 30)
plt.plot(z, equation(z))
plt.show()
x0 = [30]
res = op.minimize(equation, x0, method='BFGS')
print(equation(res.x[0]))  # ans1 = 1.75


# Global optimization, differential evolution


res = op.differential_evolution(equation, bounds=[(1, 30)])
print(equation(res.x[0]))  # ans2 = 25.88


# Optimization for non shape function (gradient defined not anywhere)


def not_shape_equation(x):
    return int(np.sin(x/5) * np.exp(x/10) + 5 * np.exp(-x/2))


x = np.arange(1, 50, 1)
y = np.arange(1, 50, 1)

for idx, el in enumerate(x):
    y[idx] = not_shape_equation(el)

plt.plot(x, y)
plt.show()
x0 = [30]
res = op.minimize(not_shape_equation, x0, method='BFGS')
print(not_shape_equation(res.x[0]))  # ans3 = -5, -11
res = op.differential_evolution(not_shape_equation, bounds=[(1, 30)])
print(not_shape_equation(res.x[0]))
