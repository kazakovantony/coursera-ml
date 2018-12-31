import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from math import sqrt

b = 3

pareto_rv = sts.pareto(b)

mean = sts.pareto.mean(b)
variance = sts.pareto.var(b)


# generates 'number_of_samples' samples of size 'samples_size' and returns an array of their means
def number_of_samples(central_n, samples):
    result = np.asarray([])
    for i in range(samples):
        result = np.append(result, np.mean(pareto_rv.rvs(central_n)))
    return result


sample = pareto_rv.rvs(1000)
plt.hist(sample, bins=100, normed=True, label='Выборочное')

x = np.linspace(1, 10, 100)
pdf = pareto_rv.pdf(x)
plt.plot(x, pdf, color='r', label='Теоретическая плотность')

plt.ylabel('Выборочное средние')
plt.xlabel('X')
plt.legend()
plt.show()

n = 5

plt.hist(number_of_samples(n, 1000), bins=20, normed=True, label='Выборочное')
plt.ylabel('Выборочное средние')
plt.xlabel('X')

norm_random_variates = sts.norm(mean, sqrt(variance / n))
x = np.linspace(0, 6, 100)
pdf = norm_random_variates.pdf(x)

plt.plot(x, pdf, color='r', label='Теоретическая плотность')
plt.legend()
plt.show()

n = 10

plt.hist(number_of_samples(n, 1000), bins=20, normed=True, label='Выборочное')
plt.ylabel('Выборочное средние')
plt.xlabel('X')

norm_random_variates = sts.norm(mean, sqrt(variance / n))
x = np.linspace(0, 6, 100)
pdf = norm_random_variates.pdf(x)

plt.plot(x, pdf, color='r', label='Теоретическая плотность')
plt.legend()
plt.show()

n = 50

plt.hist(number_of_samples(n, 1000), bins=20, normed=True, label='Выборочное')
plt.ylabel('Выборочное средние')
plt.xlabel('X')

norm_random_variates = sts.norm(mean, sqrt(variance / n))
x = np.linspace(0, 6, 100)
pdf = norm_random_variates.pdf(x)

plt.plot(x, pdf, color='r', label='Теоретическая плотность')
plt.legend()
plt.show()


# Вывод: точность аппроксимации выборочных средних, распределения парето,
# нормальным распределением с ростом n повышается, как и горовится в центральной предельной теореме
