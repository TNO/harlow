Benchmark INTRO
=============
Currently there is only LOLA & FLOLA working properly.

## Please install before running
On the adaptive_sampling venv (should exist) after a successful built:
```
pip install scikit-fuzzy tensorboardX
```

- Example of usage. Run FLOLA sampler on a 8D problam with 15 initial points writing 3 metric functions (RMSE, RRSE, MAE) for 500 adaptive steps by adding 1 point every step.

```
python benchmark_samplers.py -s FLOLA -p 8 -i 15 -m all -st 500 -n 1
```
**NOTE** - **LOLA/Prob/Random samplers do not use all this arguments**

**TODO** Similar you can use `-s LOLA/Prob/Random` to run LOLA, Probabilistic and Random (Latin Hypercube) samplers respectively.

- To visualize the results, open a terminal and navigate (cd) to the tests/benchmark_tests. Then just `tensorboard --logdir runs`
