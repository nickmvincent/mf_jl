# Matrix Factorization Recommendation in Julia

This repo implements ideas from two papers
* a nearly line-by-line translation of the dot product matrix factorization recommender code from Rendle et al.
* the concept of "data leverage" from Vincent et al.

Specifically, this code can replicate the dot product results from Rendle et al. (to summarize, Rendle and colleagues found the simple dot product approach can outperform neural methods)

Additionally, it has many additional features for simulating different types of data leverage
* data strikes/conscious data contribution
* data poisoning

## Other features
* Incorporate Strict Global Timeline

from 

On Offline Evaluation of Recommender Systems

Yitong Ji, Aixin Sun, Jie Zhang, Chenliang Li https://arxiv.org/abs/2010.11060

e.g. "2000-12-29T23:42:56.4"
julia run_experiments.jl --cutoff "2000-12-29T23:42:56.4"


## Useful one-liners
* `julia src/run_experiments.jl --lever_type strike --dataset ml-25m --n_test_negatives 50 --cutoff "2019-11-01T00:00:00.0" —-load_results`
    - collect all results from ml-25m
    - cutoff chosen to give a rough 90:10 split and be cleaner.
* 
* `julia src/run_experiments.jl --lever_type strike --dataset ml-1m --n_test_negatives 100 --cutoff "2000-12-29T23:42:56.4" --load_results` 
    - collect all results for ml-1m. If any are missing, it will run experiments (slow). Best to use a script/* script to run multiple experiments in parallel.
    - cutoff chosen to give an exact 90:10 split.
* `julia src/run_experiments.jl --lever_type strike --dataset ml-1m --n_test_negatives 0 --cutoff "2000-12-29T23:42:56.4" --load_model` 
    - use prev trained model to eval on ALL Examples.
* `julia src/run_experiments.jl --lever_type strike --lever_size 0.1 --dataset ml-1m --n_test_negatives 100 --cutoff "2000-12-29T23:42:56.4"` 
    - run a single strike experiment.