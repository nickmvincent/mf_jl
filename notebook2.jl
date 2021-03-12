### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b81173a0-7d71-11eb-05de-252115b9905c
using Gadfly, DataFrames, Printf, CSV

# ╔═╡ a9045180-7bbd-11eb-334b-d9a2fc106895
include("./mf_simple.jl")

# ╔═╡ c8f6e970-803c-11eb-1044-833a5c296b23
include("./data.jl")

# ╔═╡ ed417ac0-7d71-11eb-3433-2bb5d1bde552
md"""
# Interactive Data Leverage
## Simulating "Data Strikes", "Data Poisoning", and "Conscious Data Contribution" Against a Matrix Factorization Recommender System

In this notebook, we provide Julia code that illustrates simulated "data leverage campaigns" against a recommender system.


## What's data leverage?
In short, data leverage is the leverage exerted by a group of people who modify their "data-generating actions" to hurt a target company's "data-dependent technologies" or help a competitor's.

There are three distinct data levers:

* data strikes involve withholding or deleting data to harm a technology. We can simulate data strikes by comparing a machine learning model with full access to data with one that has does not have access to some portion of the data (because people deleted or withheld that data).

* data poisoning involves providing companies with deceptive data that may hurt a technology more than helping it

* conscious data contribution involves a group of people contributing data to help a company that aligns with their values, to help that company compete with incumbents.

To read much more about data leverage, check out this [paper](https://arxiv.org/abs/2012.09995).

We've also written papers that dive deeper into the specific of simulating [data strikes](http://nickmvincent.com/static/www2019_datastrike.pdf) and [conscious data contribution](http://nickmvincent.com/static/cdc_cscw.pdf)

## So what's in this document?
This document is mainly meant to (1) illustrate how data leverage simulations can work and (2) provide a degree of interactivity.

Currently, interacting with this notebook requires familiarity with the Julia programming language, but we plan to produce an Observable Notebook version loaded with pre-computed results, which requires only a web browser to interact with.

"""

# ╔═╡ f25307e0-8032-11eb-19b2-f7c7b57a7b00
md"""
## Configuration of the Recommender

A brief description of our configuration:
* `epochs` - how many times to look at the training data. More looks = we can get better performance, but it takes longer to train.
* `embedding_dim` - the "embedding dimension". How big will our model be? Bigger can get better performance, but it takes longer to train and uses up more space on our computer.
* η - the learning rate. Bigger means our model changes more quickly ("learns faster"), but setting this too large can cause problems. Typically set fairly low to be safe.
* λ - the "regularization". Bigger roughly means we prioritize a simpler model over a more complex model.
* `n_negatives` - each time we train with our "hits" (user-item pairs), we also want to train on some misses. In other words, we also include the fact that certain users *didn't* listen to certain albums. Having a larger n_negatives means we train on more misses.
* `stdev` - the standard deviation of the normal distribution that we pick our random starting embeddings from. Not a major concern, we'll just use the value suggested by Rendle and colleagues.

"""

# ╔═╡ c5e179d0-7dd0-11eb-18a1-95b44b195b59
epochs, embedding_dim, frac, learning_rate, regularization, n_negatives = (
	100, 16, 1.0, 0.002, 0.005, 8
)

# ╔═╡ aac56cd0-807b-11eb-07f2-dbe74e4276ee


# ╔═╡ b62409fe-807b-11eb-37e1-7fd2e79dbcaa


# ╔═╡ cdcb5d70-8053-11eb-1337-c7b88699aad1
md"""
Some notes

* we'll use just 20 epochs and set frac to 0.1 so our experiments don't take too long. This means we use only 10% of total data as the "maximum" data and show our mode the training data just 20 times. This means our maximum accuracy wil be lower than the "State of Art", but we can still show the general trends.
"""

# ╔═╡ 23e13020-8033-11eb-3052-7ba5a9019aad
md"""
# Data Lever Parameters


Let's we are organizing a data leverage campaign, and have some set of resources (funds, social capital, etc.) to allocate to organizing. We have to make choices about three "parameters" that describe our campaign

* size - how many people participate?
* genre - will our campaign target a specific genre of movies?
* type - will we engage in a data strike (delete data) or data poisoning (replace good data with "bad" data)? In this notebook, we consider "random replacement" data poisoning attacks in which participants remove their movie ratings (for a particular genre, if the campaign is targeted) and randomly pick an equal number of movies to rate.

"""

# ╔═╡ f4823870-81e9-11eb-210b-e5c941042875
#Threads.@threads for lever_size in [0, 0.05]
if length(ARGS) > 0
	lever_sizes = [parse(Float64, ARGS[1])]
else
	lever_sizes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	#lever_sizes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,]
	#lever_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
end

# ╔═╡ f307cb00-8279-11eb-3a90-3d5b5775d487
nice_dfs = []

# ╔═╡ 13f3bfd0-7d71-11eb-3275-e5bde57f6cc9

for lever_size in lever_sizes
	for lever_genre in ["All", "Comedy", "Action", "Drama"]
	for lever_type in ["strike", "poison"]
		name= "d=$embedding_dim,trn=$epochs-$learning_rate-$regularization-$n_negatives,frac=$frac,lever=$lever_size-$lever_genre-$lever_type"
		outname = "results/$name.csv"
		model_filename = "models/$name.jld"
		print(outname, "\n")
		#results = CSV.read(outname, DataFrame)
		if isfile(outname)
			results = CSV.read(outname, DataFrame)
		else
			results = main(
				epochs=epochs, lever_size=lever_size, frac=frac,
				lever_genre=lever_genre, lever_type=lever_type, outname=outname, 
				model_filename=model_filename
			)
		end
		cols = [
			"lever_size", "n_train", "lever_genre", "lever_type",
			"hr", "hr_Action", "hr_Comedy", "hr_Drama"
			#"hits_Action", "hits_Comedy", "hits_Drama"
		]
		results[:, "lever_type"] .= lever_type
		nice = results[end, cols]

		push!(nice_dfs, nice)
	end
	end
end

# ╔═╡ 291a54a0-7d71-11eb-0489-1bed3e39ca1c
nice_df = DataFrame(vcat(nice_dfs...));

# ╔═╡ 876beb20-8039-11eb-18dd-a543b3797128
size(nice_df)

# ╔═╡ 9d59ec40-8276-11eb-20d0-7feb17f82e6b
nice_df

# ╔═╡ 296cb3c0-8079-11eb-3b6f-11a1eaad14c5
max_train = maximum(nice_df.n_train)

# ╔═╡ 318fadf0-8079-11eb-18cf-3ddb1ece2e02
nice_df[:, "Observations Lost"] = max_train .- nice_df.n_train

# ╔═╡ c625e22e-8278-11eb-2c1d-21291a620125
cols_to_stack = ["hr", "hr_Action", "hr_Comedy", "hr_Drama"]
#cols_to_stack = ["hits_Action", "hits_Comedy", "hits_Drama"]

# ╔═╡ f064368e-803a-11eb-2caa-15eabf7b7f19
stacked = stack(nice_df, cols_to_stack)

# ╔═╡ 429b1530-8079-11eb-2245-59f35fcbc9c3
CSV.write("nice_df.csv", nice_df)

# ╔═╡ 714aaace-8277-11eb-0250-0b848eb41d8b
CSV.write("stacked.csv", stacked)

# ╔═╡ c55d6ca0-8279-11eb-0a0c-61297f33c369
stacked_emojis = copy(stacked)

# ╔═╡ 248e2b10-827a-11eb-34ea-45417691123f
stacked_emojis.variable = replace(
	stacked.variable, "hr_Drama" => "🎭", "hr_Comedy"=>"😂", "hr_Action"=>"🤜")

# ╔═╡ 03dba780-827a-11eb-289e-53cbc509bd14
stacked_emojis

# ╔═╡ 0c543bc0-827a-11eb-3e3a-fd04c395ee92
CSV.write("stacked_emojis.csv", stacked_emojis)

# ╔═╡ a17259de-8035-11eb-0ed7-2f65f80a7131
p1 = plot(
	stacked, x=:lever_size, y=:value, color=:lever_genre, xgroup=:variable, shape=:lever_type,
	Geom.subplot_grid(Geom.point)
)

# ╔═╡ d0c34a10-803a-11eb-36be-ff456de81a52
p2 = plot(
	stacked, x="Observations Lost", y=:value, color=:lever_genre, xgroup=:variable,
	shape=:lever_type,
	Geom.subplot_grid(Geom.point)
)

# ╔═╡ 15d9e580-8051-11eb-1f96-cb30074ec481


# ╔═╡ Cell order:
# ╠═ed417ac0-7d71-11eb-3433-2bb5d1bde552
# ╠═a9045180-7bbd-11eb-334b-d9a2fc106895
# ╠═c8f6e970-803c-11eb-1044-833a5c296b23
# ╠═b81173a0-7d71-11eb-05de-252115b9905c
# ╠═f25307e0-8032-11eb-19b2-f7c7b57a7b00
# ╠═c5e179d0-7dd0-11eb-18a1-95b44b195b59
# ╟─aac56cd0-807b-11eb-07f2-dbe74e4276ee
# ╟─b62409fe-807b-11eb-37e1-7fd2e79dbcaa
# ╠═cdcb5d70-8053-11eb-1337-c7b88699aad1
# ╠═23e13020-8033-11eb-3052-7ba5a9019aad
# ╠═f4823870-81e9-11eb-210b-e5c941042875
# ╠═f307cb00-8279-11eb-3a90-3d5b5775d487
# ╠═13f3bfd0-7d71-11eb-3275-e5bde57f6cc9
# ╠═291a54a0-7d71-11eb-0489-1bed3e39ca1c
# ╠═876beb20-8039-11eb-18dd-a543b3797128
# ╠═9d59ec40-8276-11eb-20d0-7feb17f82e6b
# ╠═296cb3c0-8079-11eb-3b6f-11a1eaad14c5
# ╠═318fadf0-8079-11eb-18cf-3ddb1ece2e02
# ╠═c625e22e-8278-11eb-2c1d-21291a620125
# ╠═f064368e-803a-11eb-2caa-15eabf7b7f19
# ╠═429b1530-8079-11eb-2245-59f35fcbc9c3
# ╠═714aaace-8277-11eb-0250-0b848eb41d8b
# ╠═c55d6ca0-8279-11eb-0a0c-61297f33c369
# ╠═248e2b10-827a-11eb-34ea-45417691123f
# ╠═03dba780-827a-11eb-289e-53cbc509bd14
# ╠═0c543bc0-827a-11eb-3e3a-fd04c395ee92
# ╠═a17259de-8035-11eb-0ed7-2f65f80a7131
# ╠═d0c34a10-803a-11eb-36be-ff456de81a52
# ╠═15d9e580-8051-11eb-1f96-cb30074ec481
