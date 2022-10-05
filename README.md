# harlow

[![Docs](https://github.com/TNO/harlow/actions/workflows/build-docs.yml/badge.svg?branch=master)](https://tno.github.io/harlow/)
[![CI](https://github.com/TNO/harlow/actions/workflows/push.yml/badge.svg)](https://github.com/TNO/harlow/actions)
[![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/JanKoune/812e7f877bc9e67a4b692669ddc71030/raw/harlow_master_coverage.json)](https://en.wikipedia.org/wiki/Code_coverage)
[![PyPI version](https://img.shields.io/pypi/v/harlow)](https://pypi.org/project/harlow/)
![python versions](https://img.shields.io/pypi/pyversions/harlow)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Adaptive surrogate modelling.

`f_target(x) ~ f_surrogate(x)`

Harlow is an Adaptive Surrogate Modeling package, in Python.
The package offers a wide range of GPU-trainable Surrogate Models for single input-output and multi input-output pairs. Additionaly a series of Adaptive Samplers is implemented that work with multivariate data providing real-time web-logging. The package offers an intergration and benchmark test suite with numerous test functions and a real case study along with visualization functionality.

#### DISCLAIMER: This repository is in development. There's no guarantee in terms of code quality or output.

## On using the repository

* Install the latest stable version using pip:

```commandline
pip install harlow
```

* Or clone and install dependencies and the code from this repo:

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
