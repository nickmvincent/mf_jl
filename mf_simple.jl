#=
Adapted from:
From: https://github.com/google-research/google-research/blob/master/dot_vs_learned_similarity/mf_simple.py
"""
=#

using Random, Distributions, LinearAlgebra, CSV, DataFrames, Printf, Profile

include("./data.jl") # load, prep, and split movielens data
include("./eval.jl") # evaluate recsys

Random.seed!(123) # Setting the seed

struct Config
    epochs::Int # number of epochs to train for
    embedding_dim::Int # embedding dim
    λ::Float64 # regularization
    num_neg::Int # number of of negative samples per pos
    η::Float64 # learning rate
    stdev::Float64 # standard dev for init
end

mutable struct MfModel
    u_e::Array{Float64}
    i_e::Array{Float64}
    u_b::Array{Float64}
    i_b::Array{Float64}
    b::Float64
end

function predict_one(m::MfModel, user::Int, item::Int)
    """
    take a model, user index, and item index, and predict a score
    """
    return m.b + m.u_b[user] + m.i_b[item] + dot(
        m.u_e[user], m.i_e[item]
    )
end

# predict_one(model, 1, 2)

function predict(model::MfModel, users::Array{Int}, items::Array{Int})
    """
    Make a prediction for each user x item.
        users - an array of users. Must be same lenght as items.
    """
    n_examples = length(users)
    preds = zeros(n_examples)
    for i = 1:n_examples
        user = users[i]
        item = items[i]
        preds[i] = predict_one(model, user, item)
    end
    return preds
end


# 
function convert_implicit_triplets(model::MfModel, pos_pairs::Array{Int}, num_neg::Int)
    """
    convert positive pairs to binary (hit/no hit)
    following the NCF paper, 
    """
    # positive pairs (user, item), and number of num_neg (user didn't hit item) to randomly sample
    # triplets are user, item, hit/no hit
    num_items = size(model.i_e)[1]
    num_pos = size(pos_pairs)[1]
    matrix = zeros(num_pos * (1+num_neg), 3)
    index = 1

    for pos_index = 1:num_pos
        user = pos_pairs[pos_index, 1]
        item = pos_pairs[pos_index, 2]

        #append!(matrix, [user, item, 1])
        matrix[index, :] = [user, item, 1]
        index += 1

        for a = 1:num_neg
            random_item = rand(1:num_items)
            
            matrix[index, :] = [user, random_item, 0]
            index += 1
        end
    end
    return convert(Array{Int}, matrix)
end


function fit!(model::MfModel, config::Config, pos_pairs::Array{Int})
    triplets = convert_implicit_triplets(model, pos_pairs, config.num_neg)
    triplets = triplets[shuffle(1:end), :]

    n_examples = size(triplets)[1]

    print("fitting with $n_examples examples\n")

    ΣL = 0.0 # summed loss
    λ = config.λ # regularization
    η = config.η # learning rate

    # Iterate over all examples and perform one SGD step.
    for i = 1:n_examples
        user, item, y = triplets[i, :]
        
        u_e = model.u_e[user, :]
        i_e = model.i_e[item, :]
        u_b = model.u_b[user]
        i_b = model.i_b[item]
        b = model.b

        score = predict_one(model, user, item)

        if score > 0
            # sigmoid
            denom = 1.0 + exp(-score)
            σ = 1.0 / denom
            L = log(denom) + (1.0 - y) * score
        else
            exp_pred = exp(score)
            σ = exp_pred / (1.0 + exp_pred)
            L = log(1.0 + exp_pred) - y * score
        end
        #loss = -1 * (hit * log(pred) + (1-hit)*log(1-pred))

        ∂L = σ - y
        
        model.u_b[user] -= η * (∂L + (λ * u_b)) #eq 9
        model.i_b[item] -= η * (∂L + (λ * i_b)) #eq 10

        model.u_e[user, :] -= η * ((∂L * i_e) + (λ * u_e)) #eq 1 
        model.i_e[item, :] -= η * ((∂L * u_e) + (λ * i_e)) #eq 12
        
        model.b -= η * (∂L + (λ * b))

        ΣL += L
    end
    
    mean_loss = ΣL / n_examples
    return mean_loss
end



#test_ratings = [[1,1,1], [1,2,0], [2,1,0], [2,2,1]]
#test_negatives =[[3], [3], [3], [3]]
#evaluate(model, test_ratings, test_negatives, 1)



function main(; epochs=10, strike_size=0, strike_genre="")
    #"ml-100k/u.data"
    filename = "ml-1m/ratings.dat"
    item_filename = "ml-1m/movies.dat"
    delim = "::"
    strat = "leave_out_last"
    frac = 0.20
    train, test_hits, test_negatives, num_users, num_items, items, all_genres = load(
        filename, delim=delim, strat=strat, frac=frac, item_filename=item_filename,
        strike_size=strike_size, strike_genre=strike_genre
        
    )

    n_train = size(train)[1]

    # config for training the recsys
    embedding_dim = 8
    λ = 0.005
    num_neg = 4
    η = 0.01#0.002
    stdev = 0.1

    config = Config(
        epochs, embedding_dim,
        λ, num_neg, 
        η, stdev
    )

    # init the model. Embeddings are drawn from normal, biases start at zero.
    dist = Normal(0, config.stdev)
    u_e = rand(dist, num_users, config.embedding_dim)
    i_e = rand(dist, num_items, config.embedding_dim)
    u_b = zeros(num_users)
    i_b = zeros(num_items)
    b = 0.0
    model = MfModel(u_e, i_e, u_b, i_b, b)

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
        record = Dict{Any,Any}("epoch"=>epoch, "n_train"=>n_train, "strike_size"=>strike_size, "strike_genre"=>strike_genre)
        record["epoch"] = epoch
        print("=== epoch $epoch\n")
        mean_loss = fit!(model, config, train_arr)
        hits, ndcgs = evaluate(model, test_hits_arr, test_neg_arr, k)
        hr = mean(hits)
        ndcg = mean(ndcgs)
        record["hr"] = hr
        record["ndcg"] = ndcg
        record["loss"] = mean_loss
        print("mean train loss: $mean_loss | HR: $hr | NDCG: $ndcg\n")
        for genre in all_genres
            matches = items[items[:, genre], "item"]
            mask = [x in matches for x in test_hits.item]
            genre_hr = round(mean(hits[mask]), digits=2)
            genre_ndcg = round(mean(ndcgs[mask]), digits=2)
            record["$genre hr"] = genre_hr
            record["$genre ndcg"] = genre_ndcg
            print("$genre $genre_hr|")
        end 
        print("\n")
        push!(records, record)
    end
    results = DataFrame(records[1])
    for record in records[2:end]
        push!(results, record)
    end
    cols = ["strike_size", "n_train", "strike_genre", "hr", "Action hr", "Comedy hr"]
    return results, results[end, cols]
end

all = []
nice_dfs = []
for size in [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    res, nice = main(epochs=10, strike_size=size)
    res_comedy, nice_comedy = main(epochs=10, strike_size=size, strike_genre="Comedy")
    push!(all, res)
    push!(all, res_comedy)

    push!(nice_dfs, nice)
    push!(nice_dfs, nice_comedy)
end
nice_df = DataFrame(vcat(nice_dfs...))
# res1, nice1 = main(epochs=5)
# res2, nice2 = main(epochs=5, strike_size=0.5)
# res3, nice3 =  main(epochs=5, strike_size=0.5, strike_genre="Comedy")
# nice = DataFrame(vcat(nice1,nice2,nice3))
# results = main()
# results

#Profile.print()