### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b8109bf0-8074-11eb-0582-576739ba04c8
include("./mf_simple.jl")

# ╔═╡ d356e0de-8074-11eb-1f34-918fca9d7521
load_config = DataLoadingConfig(
	100, false,#,"2000-12-29T23:42:56.4",
	"ml-1m", 1.0
)
lever = Lever(0, "All", false, "strike")
epochs = 100
trn_config = TrainingConfig(
	epochs, 15, # because bias is one of the dimensions
	0.005, #reg
    8, 
	0.002, #lr
    0.1
)

# ╔═╡ da594fe2-8074-11eb-09a1-8b233a4d1b6e

# ╔═╡ bfd57090-8074-11eb-0b28-8d5f91f7acf5
outpath = "results/replication"
modelname = "$outpath/$epochs.jld"
mkpath(outpath)

res = main(
    load_config, trn_config, outpath,
    lever=lever,
    load_model=false,
    modelname=modelname
)

# ╔═╡ e5889fb0-8074-11eb-1500-c1f5f236b745


# ╔═╡ Cell order:
# ╠═b8109bf0-8074-11eb-0582-576739ba04c8
# ╠═d356e0de-8074-11eb-1f34-918fca9d7521
# ╠═da594fe2-8074-11eb-09a1-8b233a4d1b6e
# ╠═bfd57090-8074-11eb-0b28-8d5f91f7acf5
# ╠═e5889fb0-8074-11eb-1500-c1f5f236b745
