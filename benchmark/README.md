# Run Benchmark

To benchmark PageRank and Clutering Coefficient, run following code:

```sh
# Benchmark PageRank
python pagerank_clustering.py --print_run --num_runs=10 --netlib=sx --bench_task=page

# Benchmark Clustering Coefficient
python pagerank_clustering.py --print_run --num_runs=10 --netlib=sx --bench_task=cluster
```