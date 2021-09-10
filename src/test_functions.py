import numpy as np
import math
import json

def ackley_nD(x, a = 20, b = 0.2, c = 2*math.pi):

    sum1 = 0
    sum2 = 0

    for i in range(0, len(x)):
        xi = x[i]
        sum1 = sum1 + (xi * xi)
        sum2 = sum2 + np.cos(c*xi)

    t1 = -a * np.exp(-b * np.sqrt(sum1/len(x)))
    t2 = -np.exp(sum2/len(x))

    return t1+t2+a+math.exp(1)


def six_hump_camel_2D(X):
    x = X[:, 0]
    y = X[:, 1]

    x2 = np.power(x, 2)
    x4 = np.power(x, 4)
    y2 = np.power(y, 2)

    return ((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2) + (x * y) + ((-4.0 + 4.0 * y2) * y2)


def six_hump_camel_2D_2input(x, y):
    x2 = np.power(x, 2)
    x4 = np.power(x, 4)
    y2 = np.power(y, 2)

    return ((4.0 - 2.1 * x2 + (x4 / 3.0)) * x2) + (x * y) + ((-4.0 + 4.0 * y2) * y2)


def forresterEtAl(X):
    term_a = 6*X-2
    term_b = 12*X-4
    return np.power(term_a,2) * np.sin(term_b)


def bohachevsky_2D(X):
    x1 = X[:,0]
    x2 = X[:,1]

    t1 = np.power(x1,2)
    t2 = 2*np.power(x2,2)
    t3 = -0.3 * np.cos(3*math.pi*x1)
    t4 = -0.4 * np.cos(4*math.pi*x2)

    return t1+t2+t3+t4+0.7


if __name__ == '__main__':

    with open('example/new_content.json', "r") as f:
        res = json.load(f)

    six_hump_camel_2D_2input(res['x'], res['y'])
    print(res)

    with open('example/new_new_content.json', "w") as f:
        json.dump(res, f)
