### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b81173a0-7d71-11eb-05de-252115b9905c
using Plots, DataFrames, Printf, CSV

# ╔═╡ a9045180-7bbd-11eb-334b-d9a2fc106895
include("./mf_simple.jl")

# ╔═╡ ed417ac0-7d71-11eb-3433-2bb5d1bde552


# ╔═╡ 22804ef0-7d72-11eb-13b0-832f251c034a


# ╔═╡ c5e179d0-7dd0-11eb-18a1-95b44b195b59
epochs, embedding_dim, frac, learning_rate, regularization, n_negatives = (
	20, 16, 0.2, 0.002, 0.005, 8
)

# ╔═╡ 921ebc80-7e0b-11eb-0ee7-43707e956e9a
nice_dfs = []

# ╔═╡ 13f3bfd0-7d71-11eb-3275-e5bde57f6cc9
Threads.@threads for strike_size in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	for strike_genre in ["", "Comedy"]
		outname = "results/e=$epochs-dim=$embedding_dim-frac=$frac-strike=$strike_size-genre=$strike_genre-lr=$learning_rate-reg=$regularization-neg=$n_negatives.csv"
		print(outname, "\n")
		#results = CSV.read(outname, DataFrame)
		if isfile(outname)
			results = CSV.read(outname, DataFrame)
		else
			results = main(
				epochs=epochs, strike_size=strike_size, frac=frac,
				strike_genre=strike_genre
			)
		end
		cols = ["strike_size", "n_train", "strike_genre", "hr", "Action hr", "Comedy hr"]
		nice = results[end, cols]

		push!(nice_dfs, nice)
	end
end

# ╔═╡ 069786d0-7dd2-11eb-359e-d7783233628d


# ╔═╡ 291a54a0-7d71-11eb-0489-1bed3e39ca1c
nice_df = DataFrame(vcat(nice_dfs...))

# ╔═╡ 8157b140-7e0b-11eb-2fdd-052e2ef2e9ca
sort(nice_df, :strike_size)

# ╔═╡ af1db160-7d70-11eb-305d-6d0253d17454
plot(nice_df.strike_size, nice_df.hr, group=nice_df.strike_genre, kind="scatter")

# ╔═╡ 14601060-7dd8-11eb-3fdc-4db77ed819c2
?plot

# ╔═╡ Cell order:
# ╠═ed417ac0-7d71-11eb-3433-2bb5d1bde552
# ╠═a9045180-7bbd-11eb-334b-d9a2fc106895
# ╠═b81173a0-7d71-11eb-05de-252115b9905c
# ╠═22804ef0-7d72-11eb-13b0-832f251c034a
# ╠═c5e179d0-7dd0-11eb-18a1-95b44b195b59
# ╠═921ebc80-7e0b-11eb-0ee7-43707e956e9a
# ╠═13f3bfd0-7d71-11eb-3275-e5bde57f6cc9
# ╠═069786d0-7dd2-11eb-359e-d7783233628d
# ╠═291a54a0-7d71-11eb-0489-1bed3e39ca1c
# ╠═8157b140-7e0b-11eb-2fdd-052e2ef2e9ca
# ╠═af1db160-7d70-11eb-305d-6d0253d17454
# ╠═14601060-7dd8-11eb-3fdc-4db77ed819c2
