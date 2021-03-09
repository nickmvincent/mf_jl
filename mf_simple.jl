#=
Adapted from:
https://github.com/google-research/google-research/blob/master/dot_vs_learned_similarity/mf_simple.py
=#

using Random, Distributions, LinearAlgebra, CSV, DataFrames, Printf

include("./data.jl") # load, prep, and split movielens data
include("./eval.jl") # evaluate recsys

#Random.seed!(123) # Setting the seed

function write_output(res, name)
    CSV.write("$name", res)
end

struct Config
    epochs::Int # number of epochs to train for
    embedding_dim::Int # embedding dim
    λ::Float64 # regularization
    n_negatives::Int # number of of negative samples per pos
    η::Float64 # learning rate
    stdev::Float64 # standard dev for init
end

mutable struct MfModel
    u_emb::Array{Float64} # user embeddings
    i_emb::Array{Float64} # item embeddings
    u_b::Array{Float64} # user biases
    i_b::Array{Float64} # item biases
    b::Float64 # global bias
end

"""
take a model, user index, and item index, and predict a score
before calling, check for -1 (unseen) values
"""
function predict_one(m::MfModel, user::Int, item::Int)

    return m.b + m.u_b[user] + m.i_b[item] + dot(
        m.u_emb[user, :], m.i_emb[item, :]
    )
end

"""
Make a prediction for each user x item.
    users - array of users. Must be same length as items.
    items - array of items.
"""
function predict(model::MfModel, users::Array{Int}, items::Array{Int})
    
    n_examples = length(users)
    preds = zeros(n_examples)
    for i = 1:n_examples
        user = users[i]
        item = items[i]
        if user == -1 || item == -1
            preds[i] = 0
        else
            preds[i] = predict_one(model, user, item)
        end
    end
    return preds
end

"""
convert positive pairs to binary (hit/no hit)
following the NCF paper
pairs is: positive pairs (user, item), and number of n_negatives (user didn't hit item) to randomly sample
triplets are user, item, hit/no hit
"""
function convert_implicit_triplets(n_items::Int, pairs::Array{Int}, n_negatives::Int)
   
    n_pairs = size(pairs)[1]
    matrix = zeros(n_pairs * (1+n_negatives), 3)
    index = 1

    for pos_index = 1:n_pairs
        user = pairs[pos_index, 1]
        item = pairs[pos_index, 2]

        matrix[index, :] = [user, item, 1]
        index += 1

        # as in Rendle et al., we just sample from ALL KNOWN items
        # this means we could include a negative twice, or include a 
        # negative that's actually a positive.
        for a = 1:n_negatives # a is unused
            random_item = rand(1:n_items)
            
            matrix[index, :] = [user, random_item, 0]
            index += 1
        end
    end
    return convert(Array{Int}, matrix)
end

"""
p - the prediction score
y - label (hit or no hit)

# using simplification from Rendle et al.'s code
# whether p >0 or not, this computes the same sigmoid and loss value from eq 8
"""
function compute_loss(p, y)
    
    if p > 0
        denom = 1.0 + exp(-p)
        σ = 1.0 / denom
        L = log(denom) + (1.0 - y) * p
    else
        exp_pred = exp(p)
        σ = exp_pred / (1.0 + exp_pred)
        L = -y*p + log(1.0 + exp_pred)
    end
    return σ, L
end

"""
fit the model with SGD
"""
function fit!(model::MfModel, config::Config, train_arr::Array{Int})
    n_items = size(model.i_emb)[1]
    # format the training data and shuffle (to avoid cycles)
    triplets = convert_implicit_triplets(n_items, train_arr, config.n_negatives)
    shuffled = triplets[shuffle(1:end), :]
    

    ΣL = 0.0 # summed loss

    λ = config.λ # for reference: regularization
    η = config.η # for reference: learning rate

    # SGD loop
    # TODO minibatch + parallelize?
    n_examples = size(shuffled)[1]
    for i = 1:n_examples
        user, item, y = shuffled[i, :]
        
        u_emb = copy(model.u_emb[user, :])
        i_emb = copy(model.i_emb[item, :])

        p = predict_one(model, user, item)
        σ, L = compute_loss(p, y)
        ∂L = σ - y

        # Update weights

        model.u_emb[user, :] -= η * (∂L * i_emb + λ * u_emb) #eq 11
        model.i_emb[item, :] -= η * (∂L * u_emb + λ * i_emb) #eq 12
        
        model.u_b[user] -= η * (∂L + λ * model.u_b[user]) #eq 9
        model.i_b[item] -= η * (∂L + λ * model.i_b[item]) #eq 10
        model.b -= η * (∂L + λ * model.b)

        # just keeping track of loss for debugging
        ΣL += L
    end

    mean_loss = ΣL / n_examples
    return mean_loss
end

"""
evalute the recommender model using "leave last out" (holdout LAST hit item + 100 random negatives)
based on this code/paper: https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
"""
function evaluate(model::MfModel, test_ratings::Array{Int}, test_negatives::Array{Int}, k::Int)
    hits = []
    ndcgs = []
    for i in 1:size(test_ratings)[1]
        triplet = test_ratings[i, :]
        negatives = test_negatives[i, :]
        hit, ndcg = eval_one_rating(model, triplet, negatives, k)
        append!(hits, hit)
        append!(ndcgs, ndcg)
    end
    return hits, ndcgs
end

function eval_one_rating(model::MfModel, triplet::Array{Int}, negatives::Array{Int}, k::Int)
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

    topk = sort(collect(zip(values(d),keys(d))), rev=true)[1:k]

    hr_at_k = get_hr(topk, target)
    ndcg_at_ak = get_ndcg(topk, target)
    return [hr_at_k, ndcg_at_ak]
end

# for REPL
# epochs = 20
# embedding_dim = 16
# stdev = 0.1
# frac = 1.0
# lever_size = 0
# lever_genre = ""
# learning_rate = 0.01
# regularization = 0.005
# n_negatives = 8

struct Lever
    size::Float64
    genre::Any
    type::Any
end

function main(;
    epochs=20, embedding_dim=16, stdev=0.1,
    frac=1.0, lever_size=0, lever_genre="All", lever_type="strike",
    learning_rate=0.002, regularization=0.005, n_negatives = 8, outname="out"
)
    #"ml-100k/u.data"
    filename = "ml-1m/ratings.dat"
    item_filename = "ml-1m/movies.dat"
    delim = "::"
    strat = "leave_out_last"
    lever = Lever(lever_size, lever_genre, lever_type)
    train, test_hits, test_negatives, n_users, n_items, items, all_genres = load(
        filename, lever, delim=delim, strat=strat, frac=frac, item_filename=item_filename
    )

    n_train = size(train)[1]
    n_test = size(test_hits)[1]
    print("n_users: $n_users, n_items: $n_items, n_train: $n_train, num_test: $n_test \n")

    # config for training the recsys
    λ = regularization
    
    η = learning_rate

    # embedding_dim - 1 is used because the user and item biases are one of the dimensions
    config = Config(
        epochs, embedding_dim - 1,
        λ, n_negatives, 
        η, stdev
    )
    print(config, "\n")

    # init the model. Embeddings are drawn from normal, biases start at zero.
    dist = Normal(0, config.stdev)
    u_emb = rand(dist, n_users, config.embedding_dim)
    i_emb = rand(dist, n_items, config.embedding_dim)
    u_b = zeros(n_users)
    i_b = zeros(n_items)
    b = 0.0
    model = MfModel(u_emb, i_emb, u_b, i_b, b)

    # convert data to Int arrays to make Julia compiler happer
    train_arr = convert(Array{Int},  train[:, ["user", "item"]])
    test_hits_arr = convert(Array{Int}, test_hits)
    test_neg_arr = convert(Array{Int}, test_negatives)
    
    k = 10

    rand_hits, rand_ndcgs = evaluate(model, test_hits_arr, test_neg_arr, k)
    rand_hr = mean(rand_hits)
    rand_ndcg = mean(rand_ndcgs)
    print("HR: $rand_hr | NDCG: $rand_ndcg\n")

    records = []
    for epoch=1:config.epochs
        record = Dict{Any,Any}("epoch"=>epoch, "n_train"=>n_train, "lever_size"=>lever_size, "lever_genre"=>lever_genre)
        record["epoch"] = epoch
        #print(train_arr[1:10, :], "\n|")
        # == Fit ==
        t_start_fit = time()
        mean_loss = fit!(model, config, train_arr)
        time_elapsed = time() - t_start_fit
        record["time_elapsed"] = time_elapsed
        # == Evaluate ==
        hits, ndcgs = evaluate(model, test_hits_arr, test_neg_arr, k)
        hr = mean(hits)
        ndcg = mean(ndcgs)
        record["hr"] = hr
        record["ndcg"] = ndcg
        record["loss"] = mean_loss

        round4 = x -> round(x, digits=4)
        s1, s2, s3, s4 = map(round4, [hr, ndcg, mean_loss, time_elapsed])
        print("$epoch:\t HR: $s1, NDCG: $s2, TrainL: $s3, Time: $s4\n")
        for genre in all_genres
            matches = items[items[:, genre], "item"]
            mask = [x in matches for x in test_hits.item]
            genre_hr = round(mean(hits[mask]), digits=2)
            genre_ndcg = round(mean(ndcgs[mask]), digits=2)
            record["hr_$genre"] = genre_hr
            record["hits_$genre"] = sum(hits[mask])
            record["ndcg_$genre"] = genre_ndcg
            print("$genre $genre_hr|")
        end 
        push!(records, record)
    end
    results = DataFrame(records[1])
    for record in records[2:end]
        push!(results, record)
    end
    write_output(results, outname)
    return results
end

# res = main(
#     epochs=2, learning_rate=0.01, regularization=0.005,
#     frac=0.05, n_negatives=8, lever_size=0.1, lever_genre="Comedy"
# )

