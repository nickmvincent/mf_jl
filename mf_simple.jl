#=
Adapted from:
https://github.com/google-research/google-research/blob/master/dot_vs_learned_similarity/mf_simple.py
=#

using Random, Distributions, LinearAlgebra, CSV, DataFrames, Printf, JLD

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
    u_emb::Matrix{Float64} # user embeddings
    i_emb::Matrix{Float64} # item embeddings
    u_b::Vector{Float64} # user biases
    i_b::Vector{Float64} # item biases
    b::Float64 # global bias
    epoch::Int #keep track of how many epochs the model has been trained
end

"""
take a model, user index, and item index, and predict a score
before calling, check for -1 (unseen) value s
"""
function predict_one(m::MfModel, user::Int, item::Int)
    return m.b + m.u_b[user] + m.i_b[item] + dot(
        m.u_emb[:, user], m.i_emb[:,item]
    )
end

"""
Make a prediction for each user x item.
    users - Matrix of users. Must be same length as items.
    items - Matrix of items.
"""
function predict(model::MfModel, users::Vector{Int}, items::Vector{Int})
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
function convert_implicit_triplets(n_items::Int, pairs_in_cols::Matrix{Int}, n_negatives::Int)
   
    n_pairs = size(pairs_in_cols, 2)
    # 3 (user-item-hit) x n_pairs * total_per_user. e.g. for n_negatives = 8, size is 3 x (n_users*9)
    # we'll transpose at the end. We're doing this cause Julia use column-major
    triplets_in_cols = zeros(Int, 3, n_pairs * (1+n_negatives))
    col_index = 1

    for pair_col = 1:n_pairs
        user, item = pairs_in_cols[:, pair_col]

        triplets_in_cols[:, col_index] = [user, item, 1]
        col_index += 1

        # as in Rendle et al., we just sample from ALL KNOWN items
        # this means we could include a negative twice, or include a 
        # negative that's actually a positive.
        for a = 1:n_negatives # a is unused
            random_item = rand(1:n_items)
            
            triplets_in_cols[:, col_index] = [user, random_item, 0]
            col_index += 1
        end
    end
    return triplets_in_cols
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
function fit!(model::MfModel, config::Config, train::Matrix{Int})
    n_items = size(model.i_emb, 2)
    # format the training data and shuffle (to avoid cycles)
    triplets = convert_implicit_triplets(n_items, train, config.n_negatives)
    shuffled_transposed = triplets[:, shuffle(1:end)]

    ΣL = 0.0 # summed loss. Not actually used, just printed for debugging.

    λ = config.λ # for reference: regularization
    η = config.η # for reference: learning rate

    # SGD loop
    # (MAYBE) TODO minibatch and/or parallelize?
    n_examples = size(shuffled_transposed, 2)
    for i = 1:n_examples
        user, item, y = shuffled_transposed[:, i]
        
        u_emb = copy(model.u_emb[:, user])
        i_emb = copy(model.i_emb[:, item])

        p = predict_one(model, user, item)
        σ, L = compute_loss(p, y)
        ∂L = σ - y

        # Update weights

        model.u_emb[:, user] -= η * (∂L * i_emb + λ * u_emb) #eq 11
        model.i_emb[:, item] -= η * (∂L * u_emb + λ * i_emb) #eq 12
        
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
function evaluate(model::MfModel, test_hits::Matrix{Int}, test_negatives::Matrix{Int}, k::Int)
    hits = []
    ndcgs = []
    for col in 1:size(test_hits, 2)
        triplet_col = test_hits[:, col]
        negatives_col = test_negatives[:, col]
        hit, ndcg = eval_one_rating(model, triplet_col, negatives_col, k)
        append!(hits, hit)
        append!(ndcgs, ndcg)
    end
    return hits, ndcgs
end

function eval_one_rating(model::MfModel, triplet::Vector{Int}, negatives::Vector{Int}, k::Int)
    items = copy(negatives)
    user, target, hit = triplet

    append!(items, target)
    
    users = fill(user, length(items))
    preds = predict(model, users, items)

    d = Dict() # item to score dict
    for i=1:length(items)
        item = items[i]
        d[item] = preds[i]
    end

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
# lever_genre = "All"
# lever_type = "strike"
# learning_rate = 0.01
# regularization = 0.005
# n_negatives = 8

struct Lever
    size::Float64
    genre::Any
    genre2::Any
    type::Any
end

function main(;
    epochs=20, embedding_dim=16, stdev=0.1,
    frac=1.0, lever_size=0, lever_genre="All", lever_type="strike",
    learning_rate=0.002, regularization=0.005, n_negatives = 8, outname="out",
    k = 10, model_filename="", load_model=false, dataset="ml-1m"
)
    #"ml-100k/u.data"
    if dataset == "ml-1m"
        filename = "ml-1m/ratings.dat"
        item_filename = "ml-1m/movies.dat"
        delim = "::"
    elseif dataset == "ml-1m"
        filename = "ml-25m/ratings.csv"
        item_filename = "ml-25m/movies.csv"
        delim = ","
    end
    
    strat = "leave_out_last"
    genre2 = false
    lever = Lever(lever_size, lever_genre, genre2, lever_type)
    train_df, test_hits_df, test_negatives, n_users, n_items, items, all_genres = load_custom(
        filename, lever, delim=delim, strat=strat, frac=frac, item_filename=item_filename
    )

    # transpose everything so we can access by column
    # + convert data to Int Matrixs to make Julia compiler happer
    my_convert = x -> copy(transpose(convert(Matrix{Int}, x)))

    train = my_convert(train_df[:, ["user", "item"]])
    test_hits = my_convert(test_hits_df)
    test_negatives = my_convert(test_negatives)

    n_train = size(train, 2)
    n_test = size(test_hits, 2)
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
    if load_model && isfile(model_filename)
        d = load(model_filename)
        model = d["model"]
    else
        dist = Normal(0, config.stdev)
        u_emb = rand(dist, config.embedding_dim, n_users)
        i_emb = rand(dist, config.embedding_dim, n_items)
        u_b = zeros(n_users)
        i_b = zeros(n_items)
        bias = 0.0
        epoch = 1
        model = MfModel(u_emb, i_emb, u_b, i_b, bias, epoch)
    end
    
    rand_hits, rand_ndcgs = evaluate(model, test_hits, test_negatives, k)
    rand_hr = mean(rand_hits)
    rand_ndcg = mean(rand_ndcgs)
    print("HR: $rand_hr | NDCG: $rand_ndcg\n")

    records = []
    for epoch=model.epoch:config.epochs
        model.epoch = epoch
        if epoch == 1 || epoch % 10 == 0 || epoch == config.epochs
            save(model_filename, "model", model)
        end
        record = Dict{Any,Any}(
            "epoch"=>epoch, "n_train"=>n_train, 
            "lever_size"=>lever_size, "lever_genre"=>lever_genre,
            "lever_type"=>lever_type
        )
        record["epoch"] = epoch
        #print(train[1:10, :], "\n|")
        # == Fit ==
        t_start_fit = time()
        mean_loss = fit!(model, config, train)
        time_elapsed = time() - t_start_fit
        record["time_elapsed"] = time_elapsed
        # == Evaluate ==
        hits, ndcgs = evaluate(model, test_hits, test_negatives, k)
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
            mask = [x in matches for x in test_hits_df.item]
            genre_hr = round(mean(hits[mask]), digits=2)
            genre_ndcg = round(mean(ndcgs[mask]), digits=2)
            record["hr_$genre"] = genre_hr
            record["hits_$genre"] = sum(hits[mask])
            record["ndcg_$genre"] = genre_ndcg
            #print("$genre $genre_hr|")
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