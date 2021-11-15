from test_functions import shekel
import lolaVoronoi
import numpy as np
from surrogate_model import NN
from sklearn.metrics import mean_squared_error
import math
from lolaVoronoi import LolaVoronoi


def shekel_benchmark():
    domain = [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]
    n_points = 200

    X1 = np.random.uniform(domain[0][0], domain[0][1], n_points)
    X2 = np.random.uniform(domain[1][0], domain[1][1], n_points)
    X3 = np.random.uniform(domain[2][0], domain[2][1], n_points)
    X4 = np.random.uniform(domain[3][0], domain[3][1], n_points)

    X = np.stack([X1, X2, X3, X4], -1)

    train_X = X[0:10, :]
    test_X = X[1:, :]
    train_y = shekel(train_X)
    test_y = shekel(test_X)

    nn = NN()
    nn.create_model(input_dim=(4,))
    nn.fit(train_X, train_y)
    y_hat = nn.predict(test_X)
    rmse = math.sqrt(mean_squared_error(test_y, y_hat))

    print(f"RMSE with initial trainingset of size {train_X.shape[0]}: {rmse}")
    experiment = [10, 50, 100, 150, 200]

    # results = run_lv_experiment(nn, train_X, train_y, test_X, test_y, domain, experiment)
    # print(results)

    results_random = run_random_experiment(nn, shekel, train_X, train_y, test_X, test_y, domain, experiment)
    print(results_random)


def run_random_experiment(model, test_fun, train_X, train_y, test_X, test_y, domain, n_samples_list):
    results = {}

    for i in n_samples_list:
        X1 = np.random.uniform(domain[0][0], domain[0][1], i)
        X2 = np.random.uniform(domain[1][0], domain[1][1], i)
        X3 = np.random.uniform(domain[2][0], domain[2][1], i)
        X4 = np.random.uniform(domain[3][0], domain[3][1], i)
        new_X = np.stack([X1, X2, X3, X4], -1)

        for j in range(0, i):
            new_y = test_fun(np.expand_dims(new_X[j,:], axis=0))
            model.update(np.expand_dims(new_X[j,:], axis=0), new_y)

        y_hat = model.predict(test_X)
        rmse = math.sqrt(mean_squared_error(test_y, y_hat))
        results[i] = rmse
        print(f"RMSE with trainingset of size {train_X.shape[0]+i}: {rmse}")

    return results


def run_lv_experiment(model, train_X, train_y, test_X, test_y, domain, n_samples_list):
    n_per_iter = 1
    results = {}
    for i in n_samples_list:
        print(f"Running LV with {i} iterations")
        lv = LolaVoronoi(
            model,
            train_X,
            train_y,
            test_X,
            test_y,
            [[domain[0], domain[1], domain[2], domain[3]]],
            shekel,
            n_iteration=i,
            n_per_iteration=n_per_iter,
            metric='rmse'
        )
        lv.run_sequential_design()
        y_hat = lv.model.predict(test_X)
        rmse = math.sqrt(mean_squared_error(test_y, y_hat))
        results[i] = rmse
        print(f"RMSE with trainingset of size {lv.train_X.shape[0]}: {rmse}")

    return results

if __name__ == '__main__':
    shekel_benchmark()



