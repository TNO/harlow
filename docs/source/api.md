(sec:api)=
# API reference

## Adaptive sampling

### Sampling base class
```{eval-rst}
.. autoclass:: harlow.sampling_baseclass.Sampler
    :members:
    :undoc-members:
    :show-inheritance:
```


### Probabilistic sampler
```{eval-rst}
.. autoclass:: harlow.probabilistic_sampling.Probabilistic_sampler
    :members:
    :undoc-members:
    :show-inheritance:
```


### LOLA-Voronoi
```{eval-rst}
.. autoclass:: harlow.lola_voronoi.LolaVoronoi
    :members:
    :undoc-members:
    :show-inheritance:
```


## Surrogating

### Surrogate base class
```{eval-rst}
.. autoclass:: harlow.surrogate_model.Surrogate
    :members:
    :undoc-members:
    :show-inheritance:
```


### Gaussian Process w/ sklearn
```{eval-rst}
.. autoclass:: harlow.surrogate_model.GaussianProcess
    :members:
```


### Gaussian Process w/ tensorflow
```{eval-rst}
.. autoclass:: harlow.surrogate_model.GaussianProcessTFP
    :members:
    :undoc-members:
    :show-inheritance:
```



### Neural Network
```{eval-rst}
.. autoclass:: harlow.surrogate_model.NN
    :members:
    :undoc-members:
    :show-inheritance:
```


### Probabilistic Neural Network
```{eval-rst}
.. autoclass:: harlow.surrogate_model.Prob_NN
    :members:
    :undoc-members:
    :show-inheritance:
```

### Bayesian Neural Network
```{eval-rst}
.. autoclass:: harlow.surrogate_model.Bayesian_NN
    :members:
    :undoc-members:
    :show-inheritance:
```


## Utilities
```{eval-rst}
.. autofunction:: harlow.helper_functions.latin_hypercube_sampling
```
