Benchmark INTRO
=============
Currently there is only LOLA & FLOLA working properly.

## Please install before running
On the adaptive_sampling venv (should exist) after a successful built:
```
pip install scikit-fuzzy tensorboardX
```

- Example of usage 
```
python benchmark_samplers.py -s FLOLA -i 50 -p 6
```
This will run the FLOLA adaptive sampling with 50 initial points on the 6D problem. Similar you can use `-s LOLA` to run LOLA.

- To visualize the results, open a terminal and navigate (cd) to the tests/benchmark_tests. Then just `tensorboard --logdir runs`