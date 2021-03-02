#=
From: https://github.com/google-research/google-research/blob/master/dot_vs_learned_similarity/mf_simple.py


Details:
 - Model: Matrix factorization with biases:
     y(u,i) = b + v_{u,1}+v_{i,1}+\sum_{f=2}^d v_{u,f}*v_{i,f}
 - Loss: logistic loss
 - Optimization algorithm: stochastic gradient descent
 - Negatives sampling: Random negatives are added during training
 - Optimization objective (similar to NCF paper)
     argmin_V \sum_{(u,i) \in S} [
          ln(1+exp(-y(u,i)))
        + #neg/|I| * \sum_{j \in I} ln(1+exp(y(u,j)))
        + reg * ||V||_2^2 ]
"""
=#

using Random, Distributions, LinearAlgebra, CSV, DataFrames, Printf

include("./data.jl")

Random.seed!(123) # Setting the seed

#train, test, test_negatives, num_items = load("ml-100k/u.data", delim="\t")


struct experiment_config
    epochs::Int64 # number of epochs to train for
    embedding_dim::Int64 # embedding dim
    λ::Float64 # regularization
    num_neg::Int64 # number of of negative samples per pos
    η::Float64 # learning rate
    st_dev::Float64 # standard dev for init
end

mutable struct mf_model
    u_e::Array{Float64}
    i_e::Array{Float64}
    u_b::Array{Float64}
    i_b::Array{Float64}
    b::Float64
end

function predict_one(m::mf_model, user::Int64, item::Int64)
    """
    take a model instance, user index, and item index, and predict a score
    """
    return m.b + m.u_b[user] + m.i_b[item] + dot(
        m.u_e[user], m.i_e[item]
    )
end

# predict_one(model, 1, 2)

function predict(model, users, items)
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


# convert positive pairs to binary (hit/no hit)
function convert_implicit_triplets(model, pos_pairs, num_neg)
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
    return convert(Array{Int64}, matrix)
end


function fit!(model, config, pos_pairs)
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

# evaluate is in a different repo, here:
# https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
function evaluate(model, test_ratings, test_negatives, k)
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

#test_ratings = [[1,1,1], [1,2,0], [2,1,0], [2,2,1]]
#test_negatives =[[3], [3], [3], [3]]
#evaluate(model, test_ratings, test_negatives, 1)

function eval_one_rating(model, triplet, negatives, k)
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

    ranked_k = sort(collect(zip(values(d),keys(d))), rev=true)[1:k]

    hr_at_k = get_hit_ratio(ranked_k, target)
    ndcg_at_ak = get_ndcg(ranked_k, target)
    return [hr_at_k, ndcg_at_ak]
end

function get_hit_ratio(ranked, target)
    for tuple in ranked
        item = tuple[2]
        if item == target
            return 1
        end
    end
    return 0
end

function get_ndcg(ranked, target)
    for i=1:length(ranked)
        tuple = ranked[i]
        item = tuple[2]
        if item == target
            return log(2) / log(i+2)
        end
    end
    return 0
end

function main()
    train, test, test_negatives, num_items = load("ml-1m/ratings.dat", delim="::", strat="leave_out_last")
    #train, test, test_negatives, num_users, num_items = load("ml-100k/u.data", delim="\t", strat="leave_out_last")
    
    epochs = 100 #256
    embedding_dim = 16
    λ = 0.005
    num_neg = 8
    η = 0.002
    st_dev = 0.1

    config = experiment_config(
        epochs, embedding_dim,
        λ, num_neg, 
        η, st_dev
    )
    num_users = length(unique(train.user))
    dist = Normal(0, config.st_dev)
    # emb and bias
    u_e = rand(dist, num_users, config.embedding_dim)
    i_e = rand(dist, num_items, config.embedding_dim)
    u_b = zeros(num_users)
    i_b = zeros(num_items)
    b = 0.0

    model = mf_model(u_e, i_e, u_b, i_b, b)

    test_hits_arr = convert(Array{Int64}, test)
    test_neg_arr = convert(Array{Int64}, test_negatives)
    k = 10

    rand_hits, rand_ndcgs = evaluate(model, test_hits_arr, test_neg_arr, k)
    rand_hr = mean(rand_hits)
    rand_ndcg = mean(rand_ndcgs)
    print("HR: $rand_hr | NDCG: $rand_ndcg\n")

    for i=1:config.epochs
        print("=== epoch $i\n")
        mean_loss = fit!(model, config, train[train.hit .== true, ["user", "item"]])
        hits, ndcgs = evaluate(model, test_hits_arr, test_neg_arr, k)
        hr = mean(hits)
        ndcg = mean(ndcgs)
        print("mean train loss: $mean_loss | HR: $hr | NDCG: $ndcg\n")
    end
end

main()