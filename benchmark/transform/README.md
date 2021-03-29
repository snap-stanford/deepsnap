# Run Transformation Benchmark

This folder contains benchmarks for the transformation of the PageRank and Clutering Coefficient.

Run the following command as an examples.

```sh
# Benchmark PageRank
python pagerank_clustering.py --print_run --num_runs=10 --netlib=sx --bench_task=page

# Benchmark Clustering Coefficient
python pagerank_clustering.py --print_run --num_runs=10 --netlib=sx --bench_task=cluster
```

Using `pyinstrument` to benchmark, need to install it at first
```
pip install pyinstrument
```
then run
```
pyinstrument ego_net.py --netlib=sx --num_runs=1 --print_run
```