import numpy as np
import matplotlib.pylab as plt

# base function: f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)
# w0 + w1x1 = f(x) kof matrix : 1x1 [w0, w1] * [X^0, X^1] (1x1) = f(x)
# points: 1, 8, 15 - find w kof w0 + w1 * 1 = f1, w0 + w1 * 8 = f2, w0 + w1 * 15 + w2 * 15^2 = f3
# points: 1,4,10,15 - w0 + w1 * 1 = f1, w0 + w1 * 4 = f2, w0 + w1 * 10 + w2 * 10^2 = f3,
# w0 + w1 * 15 + w2 * 15^2 + w3 * 15^3 = f4


def equation(x):
    return np.sin(x/5) * np.exp(x/10) + 5 * np.exp(-x/2)


def approximate_equation(y):
    return matrix_resolution[0] + matrix_resolution[1] * y + matrix_resolution[2] * np.power(y, 2) +\
           matrix_resolution[3] * np.power(y, 3)


f1 = equation(1)
f2 = equation(4)
f3 = equation(10)
f4 = equation(15)

a = np.array([[1, 1, 1, 1], [1, 4, np.power(4, 2), np.power(4, 3)], [1, 10, np.power(10, 2), np.power(10, 3)],
              [1, 15, np.power(15, 2), np.power(15, 3)]])
b = np.array([f1, f2, f3, f4])

matrix_resolution = np.linalg.solve(a, b)
print(matrix_resolution)

z = np.linspace(1, 15)
plt.plot(z, equation(z))
plt.show()


z = np.linspace(1, 15)
plt.plot(z, approximate_equation(z))
plt.show()
