### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 4249a200-7dd5-11eb-0797-f915646bc3b7
using Markdown, Distributions, PlutoUI

# ╔═╡ 88fa2210-7e11-11eb-2a02-39213f33d8b9
using Printf

# ╔═╡ f06049de-7e14-11eb-2315-6f512f180152
include("mf_simple.jl")

# ╔═╡ 46576210-7dd5-11eb-023f-017df11d98f6
md"""
# Illustrated Recommender System

In this notebook, we'll walk through the process of training and using a recommender system that implements the simple "dot product matrix factorization" approach from 

Rendle, S., Krichene, W., Zhang, L., & Anderson, J. (2020, September). Neural collaborative filtering vs. matrix factorization revisited. In Fourteenth ACM Conference on Recommender Systems (pp. 240-248).

You can read the open access preprint here: https://arxiv.org/abs/2005.09683.

In this notebook, we'll focus on a very convenient "toy" example. However, if you follow along, you'll see allow the details needed to train and test this approach on a large dataset. As Rendle and colleagues showed, this approach can get great performance!

"""

# ╔═╡ c7871280-7ddb-11eb-2317-8fbf815bb00c
md"""
## Our Data: 
Let's imagine we have data about the musical preferences of four people, Al, Bo, Cam, and and Di. Specifically, we know that
* Al (a Drake fan) has listened to Take Care (Drake), Care Package (Drake), and "Scary Days 2" (Drake).
* Bo (a fan of The Weeknd) has listend to Trilogy (The Weeknd), Kiss Land (The Weeknd), and After Hours (The Weeknd).
* Cam has listened to Take Care and Care Package.
* Di has listened to Trilogy and Kiss Land.

We want to make a recommendation to Cam and Di. Obviously, we've made this problem a bit contrived and you've probably already computed some recommendations in your head: Cam, who has listened to two Drake albums already, will probably like the new Drake, and Di, who has listened to two albums from The Weeknd, will probably like the newer Weeknd.

But in this post, we'll walk through the precise mathematics of how a matrix factorization recommender system will come to this conclusion. And then you can imagine adding more and more such ratings (say, a million, or twenty million) and making recommendations for a ton of people at once.

Below, our characters and albums will become numbers, like so:

Al = 1, Bo = 2, Cam = 3, Di = 4.

Take Care = 1, Care Package = 2, Scary Days 2 = 3, Trilogy = 4, Kiss Land = 5, After Hours = 6.

"""

# ╔═╡ 84c47990-7e0f-11eb-22a7-33a465e12cad
md"""
The Training Data and Test Data, as a collection of numbers (left column = people, right column = albums).

In other words, this matrix is a represention of the above data (that Al--aka Person 1--listened to Take Care--aka item 1--, Care Package--aka item 2--- etc.)

"""

# ╔═╡ c4107110-7dd5-11eb-06d0-238f4562ac2d
pairs = [
	[1 1]
	[1 2]
	[1 3]
	[2 4]
	[2 5]
	[2 6]
	[3 1]
	[3 2]
	[4 4]
	[4 5]
]

# ╔═╡ f494c140-7e13-11eb-1fd4-55fc44f64643
md"""
We have "test pairs". We want our system to recommend Scary Days 2 (aka item 3) to Cam (aka Person 3) and After Hours (aka items 6) to Di (aka Person 4).

Specifically, each row here tells the PERSON, the ITEM, and a 1 to indicate this is a "hit" (a successful recommendation).
"""

# ╔═╡ af1f6b30-7dd5-11eb-215c-4f8a23e9968b
test = [
	[3 3 1]
	[4 6 1]
]

# ╔═╡ 8e318db0-7e14-11eb-15de-491ef294ccd3
md"""
We will also use the other albums as "test negatives". We want to see if our recommender can succesfully prioritize the "test hits" over the other options. In other words, if we produce a personalized score for *every album*, will the correct album come out on top.
"""

# ╔═╡ 957f7150-7dd7-11eb-397f-5d754ef7a983
test_negatives = [
	[4 5 6]
	[1 2 3]
]

# ╔═╡ d39efdb0-7e14-11eb-166f-e5b12ef8b93b
md"""
We'll want to keep track of the total number of user, items, "train observations" (user-item pairs), and "test observations" (also user-item pairs).
"""

# ╔═╡ ffeb82f0-7dd6-11eb-0c2c-a150a45120af
n_users, n_items = length(unique(pairs[:,1])), length(unique(pairs[:, 2]))

# ╔═╡ 238f3300-7dd7-11eb-0a82-b5aa375bb6f7
n_train, n_test = size(pairs, 1), size(test, 1)

# ╔═╡ 2ec70520-7e15-11eb-2c00-c1993728d87c
md"""
Our model requires choosing a bunch of configuration options
* `epochs` - how many times to look at the training data. More looks = we can get better performance, but it takes longer to train.
* `embedding_dim` - the "embedding dimension". How big will our model be? Bigger can get better performance, but it takes longer to train and uses up more space on our computer.
* η - the learning rate. Bigger means our model changes more quickly ("learns faster"), but setting this too large can cause problems. Typically set fairly low to be safe.
* λ - the "regularization". Bigger roughly means we prioritize a simpler model over a more complex model.
* `n_negatives` - each time we train with our "hits" (user-item pairs), we also want to train on some misses. In other words, we also include the fact that certain users *didn't* listen to certain albums. Having a larger n_negatives means we train on more misses.
* `stdev` - the standard deviation of the normal distribution that we pick our random starting embeddings from. Not a major concern, we'll just use the value suggested by Rendle and colleagues.
"""

# ╔═╡ 2f39749e-7dd6-11eb-06e9-05f3d8ce1cbb
epochs, embedding_dim, η, λ, n_negatives, stdev = (
	1000, 4, 0.002, 0.005, 1, 0.1
)

# ╔═╡ 470dcee0-7dd7-11eb-1ae4-07706263e9a0
config = Config(
        epochs, embedding_dim - 1,
        λ, n_negatives, 
        η, stdev
    )

# ╔═╡ 540386d0-7dd7-11eb-3d1b-a383f0235912
md"""
Our "model" has 5 components: a `global bias` (a single number), a set of `user biases` (represent how much each individual listens to music), `item biases` (represents how much each item gets listened to), `user embeddings`, and `item embeddings`. The embeddings are the real "meat" of the model: this is where capture the interaction between users and albums.

The `biases` all start at 0. The user embeddings start as random numbers, but we'll update them as we go through the "learning" process.

We can predict how likely a particular user will like a particular album in the following manner:

We add the global bias, the user's bias, the item's bias, and combine (using the "dot product") the "embedding" for that user and item to produce a score. By producing scores for each possible items, we can give each user a ranked list of "here are the top items for you".
"""

# ╔═╡ 5b60e710-7dd7-11eb-0f4f-456719976680
dist = Normal(0, config.stdev)

# ╔═╡ 5e9cd830-7dd7-11eb-26ec-a74cdd4fa3ed
u_emb, i_emb, u_b, i_b, b = rand(dist, n_users, config.embedding_dim), rand(dist, n_items, config.embedding_dim), zeros(n_users), zeros(n_items), 0;

# ╔═╡ 73ba9360-7dd7-11eb-1388-63c1b8ad7064
model = MfModel(u_emb, i_emb, u_b, i_b, b);

# ╔═╡ cdecd942-7e10-11eb-37cd-0de10a09d01f
md"""
First, let's see how our totall random model does. For each user, there's 4 possible options to recommend, so we'd expect to get it right about 1/4 of the time.
"""

# ╔═╡ ac781c20-7dd9-11eb-0099-4315647a7e14
k = 1

# ╔═╡ 10af7490-7e11-11eb-1b47-156e177197c2
md"""
Produce the ranked lists!
"""

# ╔═╡ f0459d50-7e11-11eb-324f-b3d2295ac7e7
function evaluate(model, test, test_negatives, do_print)
	hits = 0
	for i in 1:size(test, 1)
		triplet = test[i, :]
		negatives = test_negatives[i, :]

		items = negatives
		user, target, hit = triplet

		append!(items, target)

		d = Dict() # item to score dict
		users = fill(user, length(items))
		preds = predict(model, users, items)

		for i=1:length(items)
			item = items[i]
			d[item] = preds[i]
		end
		pop!(items)

		full_list = sort(collect(zip(values(d),keys(d))), rev=true)
		rank = 1
		hit = 0
		if do_print == true
			for tup in full_list
				item = tup[2]
				score = tup[1]
				print("At rank $rank, we have item $item with a score of $score ")
				if item == target
					print("(this is the target)")
					if rank == 1
						print("\nSUCCESS!")
						hit = 1
					end
				end
				print("\n")
				rank +=1

			end
		end
		hits += hit

	end
	return hits
end

# ╔═╡ 9438f220-7dd8-11eb-1081-d91fb8c003db
with_terminal() do
	evaluate(model, test, test_negatives, true)
end

# ╔═╡ ca17072e-7e12-11eb-006d-0be35c993c7c
md"""
Loss tells us roughly, how well we're doing. Lower is better.
"""

# ╔═╡ 5f975d70-7dda-11eb-2a5b-2fdbd930400b
with_terminal() do
	for i in 1:epochs
		loss = round(fit!(model, config, pairs), digits=3)
		print("'Loss': $loss ")
		hits = evaluate(model, test, test_negatives, false)
		if hits == 2
			evaluate(model, test, test_negatives, true)
		end
	end
	
end

# ╔═╡ 6e47ef50-7ddb-11eb-03fd-b5ee93de3183
model

# ╔═╡ 95dde330-7e12-11eb-0b90-752ed6062d41
md"""
Just to check, how exactly did we make our predictions?
"""

# ╔═╡ a5ba4050-7e12-11eb-0c4a-333c3599ad35


# ╔═╡ Cell order:
# ╠═4249a200-7dd5-11eb-0797-f915646bc3b7
# ╠═f06049de-7e14-11eb-2315-6f512f180152
# ╠═46576210-7dd5-11eb-023f-017df11d98f6
# ╠═c7871280-7ddb-11eb-2317-8fbf815bb00c
# ╠═84c47990-7e0f-11eb-22a7-33a465e12cad
# ╟─c4107110-7dd5-11eb-06d0-238f4562ac2d
# ╠═f494c140-7e13-11eb-1fd4-55fc44f64643
# ╠═af1f6b30-7dd5-11eb-215c-4f8a23e9968b
# ╠═8e318db0-7e14-11eb-15de-491ef294ccd3
# ╟─957f7150-7dd7-11eb-397f-5d754ef7a983
# ╟─d39efdb0-7e14-11eb-166f-e5b12ef8b93b
# ╠═ffeb82f0-7dd6-11eb-0c2c-a150a45120af
# ╠═238f3300-7dd7-11eb-0a82-b5aa375bb6f7
# ╠═2ec70520-7e15-11eb-2c00-c1993728d87c
# ╠═2f39749e-7dd6-11eb-06e9-05f3d8ce1cbb
# ╠═470dcee0-7dd7-11eb-1ae4-07706263e9a0
# ╠═540386d0-7dd7-11eb-3d1b-a383f0235912
# ╠═5b60e710-7dd7-11eb-0f4f-456719976680
# ╠═5e9cd830-7dd7-11eb-26ec-a74cdd4fa3ed
# ╠═73ba9360-7dd7-11eb-1388-63c1b8ad7064
# ╠═cdecd942-7e10-11eb-37cd-0de10a09d01f
# ╠═ac781c20-7dd9-11eb-0099-4315647a7e14
# ╟─10af7490-7e11-11eb-1b47-156e177197c2
# ╠═88fa2210-7e11-11eb-2a02-39213f33d8b9
# ╠═f0459d50-7e11-11eb-324f-b3d2295ac7e7
# ╠═9438f220-7dd8-11eb-1081-d91fb8c003db
# ╠═ca17072e-7e12-11eb-006d-0be35c993c7c
# ╠═5f975d70-7dda-11eb-2a5b-2fdbd930400b
# ╠═6e47ef50-7ddb-11eb-03fd-b5ee93de3183
# ╠═95dde330-7e12-11eb-0b90-752ed6062d41
# ╠═a5ba4050-7e12-11eb-0c4a-333c3599ad35
