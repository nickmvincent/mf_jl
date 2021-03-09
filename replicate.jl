### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b8109bf0-8074-11eb-0582-576739ba04c8
include("./mf_simple.jl")

# ╔═╡ d356e0de-8074-11eb-1f34-918fca9d7521
epochs, embedding_dim, frac, learning_rate, regularization, n_negatives = (
	256, 16, 1.0, 0.002, 0.005, 8
)

# ╔═╡ da594fe2-8074-11eb-09a1-8b233a4d1b6e
lever_size = 0

# ╔═╡ bfd57090-8074-11eb-0b28-8d5f91f7acf5
res = main(
    epochs=epochs, learning_rate=learning_rate, regularization=regularization,
    frac=frac, n_negatives=8, lever_size=lever_size, outname="results/replicate.csv", model_filename="models/replicate.jld"
)


# ╔═╡ e5889fb0-8074-11eb-1500-c1f5f236b745


# ╔═╡ Cell order:
# ╠═b8109bf0-8074-11eb-0582-576739ba04c8
# ╠═d356e0de-8074-11eb-1f34-918fca9d7521
# ╠═da594fe2-8074-11eb-09a1-8b233a4d1b6e
# ╠═bfd57090-8074-11eb-0b28-8d5f91f7acf5
# ╠═e5889fb0-8074-11eb-1500-c1f5f236b745
