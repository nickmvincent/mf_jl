### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ e8ff5b3e-8045-11eb-0f2e-2170e77dc8ed
using Random, StatsBase, CSV, DataFrames

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
lever_genre = "All"
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
train, test_hits, test_negatives, n_users, n_items, items, all_genres = load_custom(
        filename, lever, delim=delim, strat=strat, frac=frac, item_filename=item_filename
   )

# ╔═╡ c9cb1e10-8218-11eb-0025-531cd3f2aece
d = StatsBase.countmap(train.item)

# ╔═╡ 1d2c26a0-821c-11eb-2f5f-7ba9393fb171


# ╔═╡ 32e42600-821c-11eb-324b-13b90dd2012d
test_hits

# ╔═╡ 53fd5aa0-821c-11eb-0771-81507025fbfa
d[-1] = 0

# ╔═╡ 8a1fff20-821c-11eb-04f9-df81cb447d1c
size(test_hits, 1)	

# ╔═╡ f313f70e-8218-11eb-3032-759312d35453
begin
cnt = 0
for i in 1:size(test_hits, 1)	
	user, target, hit = test_hits[i, :]
	all_items = copy(test_negatives[i, :])
	append!(all_items, target)
	counts = [d[item] for item in all_items]
	indices = sortperm(counts, rev=true)
	ranked = all_items[indices]
	if target in ranked[1:10]
		global cnt += 1
	end
	
end
end

# ╔═╡ 83783340-821c-11eb-1e61-177ba5027c95
cnt / size(test_hits,1)

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

# ╔═╡ d3781ab2-8107-11eb-0134-818785d91a3b
tmp = CSV.read("results/0.5_100_epoch.csv", DataFrame);

# ╔═╡ d6a16e30-8107-11eb-282d-29e76dd47d46
tmp[:, ["epoch", "hr"]]

# ╔═╡ c7b2ba80-8109-11eb-0d47-6d4e3514381e
tmp2 = CSV.read("replicate.csv", DataFrame)

# ╔═╡ e2b5189e-8109-11eb-0c0b-e5e59f632b99
tmp2[:, ["epoch", "hr"]]

# ╔═╡ a142aee0-8218-11eb-26eb-9f8a09191d4a


# ╔═╡ Cell order:
# ╠═5ab98ade-8045-11eb-1b16-c7163c589b09
# ╠═68c34ae0-8045-11eb-2e08-5971ea54f0e0
# ╠═70267780-8045-11eb-25c3-05a54abde4b6
# ╠═cacc56a0-8045-11eb-00e1-4fa676705ed7
# ╠═9aa33ca0-8045-11eb-0ea8-95b75e358290
# ╠═e8ff5b3e-8045-11eb-0f2e-2170e77dc8ed
# ╠═8c2a6720-8045-11eb-1437-8f8f047397da
# ╠═c9cb1e10-8218-11eb-0025-531cd3f2aece
# ╠═1d2c26a0-821c-11eb-2f5f-7ba9393fb171
# ╠═32e42600-821c-11eb-324b-13b90dd2012d
# ╠═53fd5aa0-821c-11eb-0771-81507025fbfa
# ╠═8a1fff20-821c-11eb-04f9-df81cb447d1c
# ╠═f313f70e-8218-11eb-3032-759312d35453
# ╠═83783340-821c-11eb-1e61-177ba5027c95
# ╠═c0f9a940-8048-11eb-1ac6-05e5562d56e2
# ╠═d3781ab2-8107-11eb-0134-818785d91a3b
# ╠═d6a16e30-8107-11eb-282d-29e76dd47d46
# ╠═c7b2ba80-8109-11eb-0d47-6d4e3514381e
# ╠═e2b5189e-8109-11eb-0c0b-e5e59f632b99
# ╠═a142aee0-8218-11eb-26eb-9f8a09191d4a
