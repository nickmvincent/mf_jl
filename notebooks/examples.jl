### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 54c8ef58-8acd-11eb-1c4c-876d8ab8ce28
using JLD

# ╔═╡ 29d821ac-8acc-11eb-2eba-4311c5383058
include("./mf_simple.jl")

# ╔═╡ 2ed58c08-8acc-11eb-2f12-f5b29ec74f57
include("./data.jl")

# ╔═╡ 4011fd12-8acc-11eb-2920-4b7a1ffe9ca9
load_config = DataLoadingConfig(
	0, "2000-12-29T23:42:56.4",
	"ml-1m", 1.0
)



# ╔═╡ 433ed42e-8acc-11eb-2098-7d3d289c7b80
trn_config = TrainingConfig(
	20, 15, # because bias is one of the dimensions
	0.005, 8, 
	0.002, 0.1
)

# ╔═╡ 0696ec9a-8acd-11eb-3416-171e87717eb0
dataset = "ml-1m"

# ╔═╡ 0497e2e4-8acd-11eb-23db-e926d7f69f83
if dataset == "ml-1m"
	filename = "ml-1m/ratings.dat"
	item_filename = "ml-1m/movies.dat"
	delim = "::"
	datarow = 1
elseif dataset == "ml-25m"
	filename = "ml-25m/ratings.csv"
	item_filename = "ml-25m/movies.csv"
	delim = ","
	datarow = 2 
end

# ╔═╡ 1086bb86-8acd-11eb-146b-b3a52ebe8a43
lever = Lever(0, "All", false, "strike")


# ╔═╡ 3022549c-8acc-11eb-1bcb-91f7abce7251
train_df, test_hits_df, test_negatives, n_users, n_items, items, all_genres = load_custom(
        load_config,
        filename, lever, delim=delim,
        item_filename=item_filename, datarow=datarow,
		
    )

# ╔═╡ 2a4d3fac-8acf-11eb-0304-3fdce0675f41
user = 3757

# ╔═╡ 9dc1d8a6-8acc-11eb-1984-5b0e745aea38
d = load("/Users/nicholasvincent/workspaces/mf_jl/results/ml-1m/50-2000-12-29T23:42:56.4/strike-All-0.0/16-0.002-0.005-8/topk/$user.jld")

# ╔═╡ e4f2a5ae-8acd-11eb-11bb-ad0b0ab10c48
item2name = Dict()

# ╔═╡ 82860c44-8acd-11eb-2ede-e91548d79d5a
for row in eachrow(items)
	item2name[row.item] = row.name
end

# ╔═╡ 57f273aa-8acf-11eb-25d7-677803ee5637
map(x -> item2name[x], test_hits_df[test_hits_df.user .== user, :].item)

# ╔═╡ 03777450-8ace-11eb-39c3-ebaddc2f4e34
seen = map(x -> item2name[x], train_df[train_df.user .== user, :].item)

# ╔═╡ 4df825fe-8acd-11eb-1159-b5bff16567fa
recs = map(x-> item2name[x], [x[2] for x in d["topk"]])

# ╔═╡ 3623cfe6-8acf-11eb-3c58-7b73a1153169


# ╔═╡ Cell order:
# ╠═29d821ac-8acc-11eb-2eba-4311c5383058
# ╠═2ed58c08-8acc-11eb-2f12-f5b29ec74f57
# ╠═4011fd12-8acc-11eb-2920-4b7a1ffe9ca9
# ╠═433ed42e-8acc-11eb-2098-7d3d289c7b80
# ╠═0696ec9a-8acd-11eb-3416-171e87717eb0
# ╠═0497e2e4-8acd-11eb-23db-e926d7f69f83
# ╠═1086bb86-8acd-11eb-146b-b3a52ebe8a43
# ╠═3022549c-8acc-11eb-1bcb-91f7abce7251
# ╠═54c8ef58-8acd-11eb-1c4c-876d8ab8ce28
# ╠═2a4d3fac-8acf-11eb-0304-3fdce0675f41
# ╠═9dc1d8a6-8acc-11eb-1984-5b0e745aea38
# ╠═e4f2a5ae-8acd-11eb-11bb-ad0b0ab10c48
# ╠═82860c44-8acd-11eb-2ede-e91548d79d5a
# ╠═57f273aa-8acf-11eb-25d7-677803ee5637
# ╠═03777450-8ace-11eb-39c3-ebaddc2f4e34
# ╠═4df825fe-8acd-11eb-1159-b5bff16567fa
# ╠═3623cfe6-8acf-11eb-3c58-7b73a1153169
