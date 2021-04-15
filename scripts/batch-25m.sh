#!/bin/bash    
for size in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
do
    julia src/run_experiments.jl --lever_size $size --lever_type strike --dataset ml-25m --n_test_negatives 50 --cutoff "2019-11-01T00:00:00.0" &
done
wait
