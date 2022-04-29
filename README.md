# adaptive_sampling

Adaptive surrogate modelling.

`f_target(x) ~ f_surrogate(x)`

## On using the repository

* Install dependencies and the code from this repo:

```commandline
pip install -e .
```

* Install with additional dependencies for building documentation:

```commandline
pip install -e .[docs]
```

* To build the documentation locally:

```commandline
sphinx-build -b html docs/source docs/build
```

* To view the documentation open `docs/build/index.html`



* All code within this repository is expected to be run with the
    working directory set as the root directory of the repository.
