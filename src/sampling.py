import numpy as np
from skopt.space import Space
from skopt.sampler import Lhs


def latin_hypercube(domain, n_samples, method="maximin"):
    space_domain = []
    for i in range(len(domain)):
        space_domain.append(tuple(domain[i]))

    space = Space(space_domain)
    lhs = Lhs(criterion=method, iterations=10000)
    samples = lhs.generate(space.dimensions, n_samples)

    return samples

#TODO add lola-voronoi and iterative sampling strategies as interface


if __name__ == '__main__':
    domain = [[-100,100], [-100,100]]
    x = latin_hypercube(domain, 30)

    print(x)