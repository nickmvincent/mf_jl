### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ e8ff5b3e-8045-11eb-0f2e-2170e77dc8ed
using Random

# ╔═╡ 5ab98ade-8045-11eb-1b16-c7163c589b09
include("./mf_simple.jl")

# ╔═╡ 68c34ae0-8045-11eb-2e08-5971ea54f0e0
include("./data.jl")

# ╔═╡ 70267780-8045-11eb-25c3-05a54abde4b6
x =5

# ╔═╡ cacc56a0-8045-11eb-00e1-4fa676705ed7
begin
epochs = 20
embedding_dim = 16
stdev = 0.1
frac = 1.0
lever_size = 0
lever_genre = ""
lever_type = "strike"
learning_rate = 0.01
regularization = 0.005
n_negatives = 8
filename = "ml-1m/ratings.dat"
item_filename = "ml-1m/movies.dat"
delim = "::"
strat = "leave_out_last"
end

# ╔═╡ 9aa33ca0-8045-11eb-0ea8-95b75e358290
lever = Lever(lever_size, lever_genre, lever_type)

# ╔═╡ 8c2a6720-8045-11eb-1437-8f8f047397da
train, test_hits, test_negatives, n_users, n_items, items, all_genres = load(
        filename, lever, delim=delim, strat=strat, frac=frac, item_filename=item_filename
   )

# ╔═╡ c648f160-8045-11eb-15f4-ebc2f4bd5b52


# ╔═╡ c0f9a940-8048-11eb-1ac6-05e5562d56e2
begin
	tot = 0
	n = size(train, 1)
	for genre in all_genres
	    matches = items[items[:, genre], "item"]
	    mask = [x in matches for x in train.item]
	    s = sum(mask)
	    frac = round(s / n, digits=3)
	    tot += s
	    print("$genre $s $frac \n")
	end 
	print("\n")
	print(tot, "\n")
end

# ╔═╡ Cell order:
# ╠═5ab98ade-8045-11eb-1b16-c7163c589b09
# ╠═68c34ae0-8045-11eb-2e08-5971ea54f0e0
# ╠═70267780-8045-11eb-25c3-05a54abde4b6
# ╠═cacc56a0-8045-11eb-00e1-4fa676705ed7
# ╠═9aa33ca0-8045-11eb-0ea8-95b75e358290
# ╠═e8ff5b3e-8045-11eb-0f2e-2170e77dc8ed
# ╠═8c2a6720-8045-11eb-1437-8f8f047397da
# ╠═c648f160-8045-11eb-15f4-ebc2f4bd5b52
# ╠═c0f9a940-8048-11eb-1ac6-05e5562d56e2
