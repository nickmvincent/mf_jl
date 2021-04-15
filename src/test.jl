include("./mf_simple.jl")

epochs, embedding_dim, frac, learning_rate, regularization, n_negatives = (
	20, 16, 0.1, 0.002, 0.005, 8
)
lever_size = 0
res = main(
    epochs=epochs, learning_rate=learning_rate, regularization=regularization,
    frac=frac, n_negatives=8, lever_size=lever_size, outname="results/test.csv", model_filename="models/test.jld"
)

res = main(
    epochs=epochs, learning_rate=learning_rate, regularization=regularization,
    frac=frac, n_negatives=8, lever_size=lever_size, outname="results/test.csv", model_filename="models/test.jld", load_model=true
)