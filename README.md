# harlow

[![CI](https://github.com/JanKoune/harlow/actions/workflows/push.yml/badge.svg)](https://github.com/JanKoune/harlow/actions)
[![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JanKoune/812e7f877bc9e67a4b692669ddc71030/raw/harlow_master_coverage.json)](https://en.wikipedia.org/wiki/Code_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
