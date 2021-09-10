from sampling import latin_hypercube
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import logging
import plotting
from test_functions import six_hump_camel_2D

# adapted from https://github.com/FuhgJan/StateOfTheArtAdaptiveSampling/blob/master/src/adaptive_techniques/LOLA_function.m and
# gitlab.com/energyincities/besos/-/blob/master/besos/

#TODO add 1D case
#TODO code samples unexpected locations. check what goes wrong
class LolaVoronoi():

    def __init__(self, model, train_X, train_y, test_X, test_y, dom, f, n_init = 20, n_iteration = 10, n_per_iteration = 5):
        self.model = model
        self.dimension = len(train_X[0,:])
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.dom = dom
        self.n_init = n_init
        self.n_iteration = n_iteration
        self.n_per_iteration = n_per_iteration
        self.f = f

        self.score = np.empty((self.n_iteration+1))

        if np.ndim(self.test_X) == 1:
            self.score[0] = r2_score(self.test_y, self.model.predict(self.test_X.reshape(-1,1)))
        else:
            self.score[0] = r2_score(self.test_y, self.model.predict(self.test_X))

    def update_model(self):
        self.model.fit(self.train_X, self.train_y)


    def run_sequential_design(self):
        self.N, self.S = initialize_samples(self.train_X)
        self.sample()
        self.update_model()

        for i in range(self.n_iteration):
            P_new = select_new_datapoints()
            y = self.model(P_new)
            P = np.concatenate(P, P_new)
            self.update_model()



    def apply(self):
        self.N, self.S = initialize_samples(self.train_X)
        self.sample()
        self.update_model()

        for i in range(self.n_iteration):
            for train_new in self.new_data:
                self.N, self.S = update_neighbourhood(self.N, self.train_X, self.S, train_new)
                N_new, S_new = initialize_samples(self.train_X, train_new)
                self.N = np.append(self.N, N_new, axis=2)
                self.S = np.append(self.S, S_new, axis=0)
            self.train_X = np.append(self.train_X, self.new_data, axis=0)
            res = self.f(self.new_data)
            self.train_y = np.append(self.train_y, self.new_data_y, axis=0)
            self.sample()
            self.update_model()
            self.score[i+1] = r2_score(self.test_y, self.model.predict(self.test_X))

            print(f"iteration {i} finished: score {self.score[i+1]}")


    def sample(self):
        lola_est = lola_score(self.N, self.train_X, self.model)
        voronoi, samples = estimate_voronoi_volume(self.train_X, self.dom)
        hybrid_score = lola_voronoi_score(lola_est, voronoi)
        idx_new = np.argsort(hybrid_score)
        data_sorted = self.train_X[idx_new,:]

        ind = 2
        self.new_data = np.empty([self.n_per_iteration, self.train_X.shape[1]])

        for i in range(self.n_per_iteration):
            candidates = in_voronoi_region(data_sorted[-i-1,:], self.train_X, samples)

            while len(candidates) <= 1:
                candidates = in_voronoi_region(data_sorted[-i-ind,:], self.train_X, samples)
                ind += 1

            #new X data sample
            self.new_data[i,:] = select_new_sample(data_sorted[-i-1,:], self.N[:,:, idx_new][:,:,-i-1], candidates)

        #output from function evaluation on newly selected X samples
        self.new_data_y = self.f(self.new_data)

#N: Neighbours for points in train_X
#S: Scores for neighbourhood

def select_new_sample(d, neighhbours, candidates):
    neighhbours = np.append(neighhbours, [d], axis =0)
    dist_max = 0

    for candidate in candidates:
        d = 0
        for neighbour in neighhbours:
            d = d+np.linalg.norm(candidate-neighbour)
        if d>dist_max:
            dist_max = d
            candidate_max = candidate

    return candidate_max



def in_voronoi_region(d, train_X, samples):
    mask = train_X != d
    train_X_temp = train_X[mask[:,0], :]
    candidates = np.empty([1, train_X.shape[1]])

    for s in samples:
        for train_i in train_X_temp:
            if np.linalg.norm(s-train_i) <= np.linalg.norm(s-d):
                break
            else:
                continue
        if np.all(train_i == train_X[-1]):
            candidates = np.append(candidates, [s], axis=0)

    return candidates


def initialize_samples(train_X, train_new = None):
    m = 2 * len(train_X[0,:])
    d = train_X.shape[1]

    if np.any(train_new == None):
        n = len(train_X)
        Neighbourhood_points = np.empty((m,d,n))
        Score_points = np.empty((n))
        train_ref = (train_X*1.0)
    else:
        if np.ndim(train_new) == 1:
            train_ref = np.expand_dims(train_new, axis=0)
        else:
            train_ref = train_new * 1.0
        n_new = len(train_ref)
        Neighbourhood_points = np.empty([m,d,n_new])
        Score_points = np.empty([n_new])

    for i in range(len(train_ref)):
        mask = train_X != train_ref[i,:]
        train_norefpoint = train_X[mask[:,0],:]
        Neighbourhood_points[:,:,i] = train_norefpoint[0:m,:]
        Score_points[i] = neighbourhood_score(Neighbourhood_points[:,:,i],train_ref[i,:], np.ndim(train_new))

    ind = 0

    for cand in train_X:
        ind += 1

        Neighbourhood_points, Score_points = update_neighbourhood(Neighbourhood_points, train_ref, Score_points, cand)

    return Neighbourhood_points, Score_points


def lola_voronoi_score(E, V):
    return V + E / sum(E)


def update_neighbourhood(neighbours, train_X, scores, candidates):
    if np.ndim(train_X) == 1:
        train_X = np.expand_dims(train_X, axis=0)
        neighbours = np.reshape(neighbours, (neighbours.shape[1], neighbours.shape[1], 1))
        scores = np.expand_dims(scores, axis=0)

    if np.ndim(candidates) == 1:
        candidates = np.expand_dims(candidates, axis=0)

    m = 2*len(train_X[0,:])
    ind = 0

    for candidate in candidates:
        for p in train_X:
            if sum(p == candidate) < len(train_X[0,:]):
                neighbours_temp = np.dstack([neighbours[:,:,ind]]*m)
                scores_temp = np.zeros((m))

                for i in range(m):
                    if sum(sum(neighbours_temp[:,:,i] == candidate)) < len(train_X[0,:]):
                        neighbours_temp[i,:,i] = candidate
                    else:
                        pass
                    scores_temp[i] = neighbourhood_score(neighbours_temp[:,:,i], p, np.ndim(candidates))
                min_ind = np.argmin(scores_temp)

                neighbours[:,:,ind] = neighbours_temp[:,:, min_ind]
                scores[ind] = scores_temp[min_ind]
                ind += 1
            else:
                pass

    return neighbours, scores


def neighbourhood_score(neighbourhood, candidates, dim):
    m = len(neighbourhood)

    cand_dist = np.empty((m))
    min_dist = np.empty((m))

    C = 0
    for i in range(m):
        C = C + np.linalg.norm(neighbourhood[i,:] - candidates)
    C = C / m

    for i in range(m):
        for j in range(m):
            if i==j:
                pass
            else:
                cand_dist[i] = np.linalg.norm(neighbourhood[i,:] - neighbourhood[j,:])
        min_dist[i] = min(cand_dist)

    if dim > 1:
        A = 1 / m * sum(min_dist)
        R = A / (np.sqrt(2) * C)
    elif dim == 1:
        #todo check and fix
        R = np.empy((m))
        for i in range(m):
            R[i] = 1-(np.abs(neighbourhood[i,:] + neighbourhood[i+1]) /
                      (np.abs(neighbourhood[i] + np.abs(neighbourhood[i+1]) + np.abs(neighbourhood[i]-neighbourhood[i+1]))))


    return R/C



def lola_score(neighbours, train_X, model):
    n = len(train_X)
    idx = 0
    E = np.empty([n])

    for p in train_X:
        grad = gradient(neighbours[:,:,idx], p, model)
        E[idx] = nonlinearity_measure(grad, neighbours[:,:,idx], p, model)
        idx+=1

    return E

def nonlinearity_measure(grad, neighbours, p, model):
    E = 0

    for i in range(len(neighbours)):
        E = E+abs(model.predict([neighbours[i,:]]) - (model.predict([p]) + np.dot(grad, (neighbours[i,:] - p))))

    return E

'''
 Exploration using Voronoi approximation: identify regions where sample density is low. low Voronoi volume implies low sampling density.
 Crombecq, Karel; Gorissen, Dirk; Deschrijver, Dirk; Dhaene, Tom (2011) A novel hybrid sequential design strategy for global surrogate modeling of computer experiments
 Estimation of Voronoi cell size, see algorithm 2.
 How large should n be approximate V-size?

 alternative would be to calculated voronoi cell via Delauney tesselation. more expensive, not necessary according to paper.

 alternative for finding nearest, using kdtrees:
   from scipy.spatial import KDTree
   kdt = KDTree(P.T)
   kdt.query(PQ.T)
'''
def estimate_voronoi_volume(samples, domain, n = 500):

    V = np.zeros(len(samples))
    S = latin_hypercube(domain, n)

    for s in S:
        d = np.inf
        idx = 0
        for p in samples:
            if np.linalg.norm(p-s) < d:
                d = np.linalg.norm(p-s)
                idx_fin = idx
            idx += 1
        V[idx_fin] = V[idx_fin] + 1 / len(S)

    return V, np.asarray(S)


def gradient(N, p, model):
    m = len(N)
    d = len(p)

    P_mtrx = np.empty((m,d))
    F_mtrx = np.empty((m))

    for i in range(m):
        P_mtrx[i,:] = N[i,:] - p
        F_mtrx[i] = model.predict(N[i,:].reshape(1,-1))

    grad = np.linalg.lstsq(P_mtrx, np.transpose(F_mtrx), rcond=None)[0].reshape((1,d))

    return grad

def test():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
    from sklearn.manifold import MDS
    from sklearn.model_selection import train_test_split
    from test_functions import bohachevsky_2D, six_hump_camel_2D, six_hump_camel_2D_2input
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from plotting import plot_function_custom

    x0_range = [-3, 3]
    x1_range = [-2, 2]

    grid_size = 30
    X1_s = np.random.uniform(x0_range[0], x0_range[1], grid_size)
    X2_s = np.random.uniform(x1_range[0], x1_range[1], grid_size)
    X = np.stack((X1_s,X2_s),-1)
    #plot_function_custom(six_hump_camel_2D, X, plot_sample_locations=True, show=True)


    indices = np.random.permutation(X1_s.shape[0])
    train_idx, test_idx = indices[:round(len(indices)*0.8)], indices[round(len(indices)*0.8):]
    train_X= X[train_idx,:]
    test_X = X[test_idx,:]
    train_y = six_hump_camel_2D(train_X)
    test_y = six_hump_camel_2D(test_X)

    gp = GaussianProcessRegressor()
    gp.fit(train_X, train_y)
    p = gp.predict(X)
    p2 = gp.predict(train_X)

    #plot_function_custom(six_hump_camel_2D, X, y=p, plot_sample_locations=True, show=True)

    n_iters = 10
    n_per_iters = 10
    lv = LolaVoronoi(gp, train_X, train_y, test_X, test_y, [x0_range, x1_range], bohachevsky_2D, n_iteration=n_iters, n_per_iteration=n_per_iters)
    lv.apply()
    #
    end_sample = (lv.n_iteration * lv.n_per_iteration)
    #
    cmap = plotting.get_cmap(lv.n_iteration)
    plot = plotting.plot_function_custom(six_hump_camel_2D, train_X,
                                          y=train_y, show=True)
    #
    plot2 = plotting.plot_function_custom(six_hump_camel_2D, lv.train_X, lv.train_y, show=True)

    gp2 = GaussianProcessRegressor()
    gp2.fit(train_X, train_y)
    for j in range(n_iters):
        X1_new = np.random.uniform(x0_range[0], x0_range[1], n_per_iters)
        X2_new = np.random.uniform(x1_range[0], x1_range[1], n_per_iters)
        X_new = np.stack((X1_new, X2_new),-1)
        new_y = six_hump_camel_2D(X_new)
        train_X = np.append(train_X, X_new, axis=0)
        train_y = np.append(train_y, new_y, axis=0)
        gp2.fit(train_X, train_y)
        print(f"r2: {r2_score(test_y, gp2.predict(test_X))}")
    plot3 = plotting.plot_function_custom(six_hump_camel_2D, train_X, train_y, show=True)

    gp2 = GaussianProcessRegressor()
    gp2.fit(lv.train_X, lv.train_y)
    p2 = gp2.predict(X)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #surface = ax.plot_surface(train_x1, train_x2, , cmap=cm.coolwarm)
    surface = ax.plot_surface(x1, x2, p2.reshape(grid_size, grid_size), cmap=cm.coolwarm)
    ax.scatter(new_points_x1, new_points_x2, t)
    fig.colorbar(surface)
    plt.show()

    print(lv.score)



def test2():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
    from sklearn.manifold import MDS
    from sklearn.model_selection import train_test_split
    from test_functions import forresterEtAl
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from plotting import plot_function_custom,add_samples_to_plot

    X_range = np.linspace(0,1,1000)
    y_range = forresterEtAl(X_range)
    domain = [0,1]
    n_points = 10
    X = np.random.uniform(domain[0], domain[1], n_points)

    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:round(len(indices)*0.8)], indices[round(len(indices)*0.8):]
    train_X= np.sort(X[train_idx])
    test_X = np.sort(X[test_idx])
    train_y = forresterEtAl(train_X)
    test_y = forresterEtAl(test_X)

    gp = GaussianProcessRegressor()
    gp.fit(train_X.reshape(-1,1), train_y)

    p = gp.predict(test_X.reshape(-1,1))
    print(f"test R2: {r2_score(test_y, p)}")

    plot = plot_function_custom(six_hump_camel_2D, train_X, y=gp.predict(train_X.reshape(-1,1)), plot_sample_locations=True, show=False)
    #plot = add_samples_to_plot(plot, test_X, p, 'r' )
    plot.plot(X_range, y_range, 'r')

    n_iters = 5
    n_per_iters = 3
    lv = LolaVoronoi(gp, train_X.reshape(-1,1), train_y.reshape(-1,1), test_X.reshape(-1,1), test_y.reshape(-1,1), [[domain[0], domain[1]]], forresterEtAl, n_iteration=n_iters,
                     n_per_iteration=n_per_iters)
    lv.apply()

    plot = add_samples_to_plot(plot, lv.train_X[-n_iters*n_per_iters:], forresterEtAl(lv.train_X[-n_iters*n_per_iters:]), 'g')
    plot.show()


def test_fun(X):
    x1 = X[:,0]
    x2 = X[:,1]
    return np.sin(x1-3)*np.cos(x2/4)

if __name__ == '__main__':
    test2()