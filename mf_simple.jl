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

using Random, Distributions, LinearAlgebra, CSV, DataFrames
include("./data.jl")

Random.seed!(123) # Setting the seed

train = load("ml-100k/u1.base")
test = load("ml-100k/u1.test")

# init a random user embeddings and item embeddings. normal var w/ mean 0 and stdev by param
# vector that is # user x embedding dimensions
mutable struct mf_model
    u_e
    i_e
    u_b
    i_b
    b
end


struct experiment_config
    epochs # number of epochs to train for
    embedding_dim # embedding dim
    reg # regularization
    num_neg # number of of negative samples per pos
    lr # learning rate
    st_dev # standard dev for init
end

epochs = 10
embedding_dim = 16
regularization = 0.005
num_neg = 8
lr = 0.002
st_dev = 0.01

config = experiment_config(
    epochs, embedding_dim,
    regularization, num_neg, 
    lr, st_dev
)

d = Normal(0, config.st_dev)

num_users = length(unique(train.user))
num_items = length(unique(train.item))

#num_user = 10
# user embedding
u_e = rand(d, num_users, config.embedding_dim)
# item embedding
i_e = rand(d, num_items, config.embedding_dim)

# user bias
u_b = zeros(num_users)
# item bias
i_b = zeros(num_items)
# global bias
b = 0

model = mf_model(u_e, i_e, u_b, i_b, b)

function predict_one(m, u_index, i_index)
# bias, user bias, item bias, user embedding, item embedding
    return m.b + m.u_b[u_index] + m.i_b[i_index] + dot(
        m.u_e[u_index], m.i_e[i_index]
    )
end

predict_one(model, 1, 2)

function predict(model, users, items)
    num_examples = length(pairs[1])
    predictions = zeros(num_examples)
    for i = 1:num_examples
        u_index = users[i]
        i_index = items[i]
        predictions[i] = predict_one(model, u_index, i_index)
    end
    return predictions
end


# convert positiv pairs to binary (hit/no hit)
function convert_implicit_triplets(model, pos_pairs, num_neg)
    # positive pairs (user, item), and number of negatives (user didn't hit item) to randomly sample
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
            j = rand(1:num_items)
            
            matrix[index, :] = [user, j, 0]
            #append!(matrix, [user, j, 0])
            index += 1
        end
    end
    return convert(Array{Int64}, matrix)
end


function fit(model, config, pos_pairs)
    triplets = convert_implicit_triplets(model, pos_pairs, config.num_neg)
    triplets = triplets[shuffle(1:end), :]

    # Iterate over all examples and perform one SGD step.
    num_examples = length(triplets)

    summed_loss = 0.0

    for i = 1:num_examples
        user, item, hit = triplets[i, :]
        u_e = model.u_e[user, :]
        i_e = model.i_e[item, :]
        u_b = model.u_b[user]
        i_b = model.i_b[item]
        pred = predict_one(model, user, item)

        if pred > 0
            # sigmoid
            one_plus_exp_neg_pred = 1 / (1 +  exp(-pred))
            sigmoid = 1.0 / one_plus_exp_neg_pred

            # if rating = 0, add the pred. if rating = 1, don't.
            loss = log(one_plus_exp_neg_pred) + ((1 - hit) * pred)
        else
            exp_pred = exp(pred)
            sigmoid = exp_pred / (1.0 + exp_pred)
            loss = log(1.0 + exp_pred) - hit * pred
        end

        grad = hit - sigmoid
        
        model.u_e[user, :] += lr * (grad * i_e - config.reg  * u_e)
        model.i_e[item, :] += lr * (grad * u_e - config.reg  * i_e)

        model.u_b[user] += lr * (grad - config.reg  * u_b)
        model.i_b[item] += lr * (grad - config.reg  * i_b)
        model.b += lr * (grad - config.reg  * model.b)

        summed_loss += loss
    end
    
    mean_loss = summed_loss / num_examples
end

# evaluate is in a different repo, here:
# https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
function evaluate(model, test_ratings, test_negatives, k)
    hits = []
    ndcgs = []
    for i in 1:length(test_ratings)
        triplet = test_ratings[i]
        negatives = test_negatives[i]
        hit, ndcg = eval_one_rating(model, triplet, negatives, k)
        print(hit, ndcg)
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
    user, target, rating = triplet
    append!(items, target)
    
    d = Dict() # item to score dict
    users = fill(user, length(items))
    preds = predict(model, users, items)

    for i=1:length(items)
        item = items[i]
        d[item] = predictions[i]
    end
    pop!(items)

    ranked_k = sort(collect(zip(values(d),keys(d))))[1:k]

    hr_at_k = get_hit_ratio(ranked_k, target)
    ndcg_at_ak = get_ndcg(ranked_k, target)
    return [hr_at_k, ndcg_at_ak]
end

function get_hit_ratio(ranked, target)
    for item in ranked
        if item == target
            return 1
        end
    end
    return 0
end

function get_ndcg(ranked, target)
    for i=1:length(ranked)
        item = ranked[i]
        if item == target
            return log(2) / log(i+2)
        end
    end
    return 0
end


fit(model, config, train[train.hit .== true, ["user", "item"]])