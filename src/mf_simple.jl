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

struct Lever
    size::Float64
    genre::Any
    genre2::Any
    type::Any
end

struct TrainingConfig
    epochs::Int # number of epochs to train for
    embedding_dim::Int # embedding dim
    λ::Float64 # regularization
    n_negatives::Int # number of of negative samples per pos
    η::Float64 # learning rate
    stdev::Float64 # standard dev for init

end

struct DataLoadingConfig
    n_test_negatives::Int
    cutoff::Any
    dataset::Any
    frac::Float64
end

mutable struct MfModel
    u_emb::Matrix{Float64} # user embeddings
    i_emb::Matrix{Float64} # item embeddings
    u_b::Vector{Float64} # user biases
    i_b::Vector{Float64} # item biases
    b::Float64 # global bias
    epoch::Int #keep track of how many epochs the model has been trained
    outpath::String # where to write files?
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

function predict_unseen(m::MfModel)
    return m.b
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
            preds[i] = predict_unseen(model)
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
function fit!(model::MfModel, config::TrainingConfig, train::Matrix{Int})
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
function evaluate(
    model::MfModel, hit_triplets::Matrix{Int},
    negatives::Dict{}, k::Int;
    print_users=[]
)

    hits = []
    ndcgs = []
    topk_records = []
    for col in 1:size(hit_triplets, 2)
        triplet_col = hit_triplets[:, col]
        user = triplet_col[1]
        # if triplet_col[1] == -1
        #     triplet_col[1] = shuffle(1:size(negatives,2))[1]
        # end
        # get the random negative items to rank alongside the "hit"
        negatives_col = negatives[user]
        # drop the zeros (no item). Only needed if negatives is an array. As of 4/21, it's a Dict.
        #negatives_col = [x for x in negatives_col if x != 0]

        save_topk = false
        if user in print_users
            save_topk = true
        end
        hit, ndcg, topk = eval_one_rating(model, triplet_col, negatives_col, k, save_topk=save_topk)
        append!(hits, hit)
        append!(ndcgs, ndcg)
        if user in print_users
            record = Dict()
            record["user"] = user
            record["item"] = triplet_col[2]
            record["topk"] = [x[2] for x in topk]
            push!(topk_records, record)
        end
        
    end
    return hits, ndcgs, topk_records
end

"""
Evaluate one "hit". 
We create a list of user-specific "no-hits" (negatives) and one hit. We rank this list and see if the hit appears in top k.

Returns hit rate at k, ndcg at k, and 
"""
function eval_one_rating(model::MfModel, triplet::Vector{Int}, negatives::Vector{Int}, k::Int; save_topk=false)
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

    if k > length(values(d))
        k = length(values(d))
    end
    topk = sort(collect(zip(values(d),keys(d))), rev=true)[1:k]
    if save_topk
        dest = "$(model.outpath)/topk/$user.jld"
        save(dest, "topk", topk)
    end

    hr_at_k = get_hr(topk, target)
    ndcg_at_ak = get_ndcg(topk, target)
    return [hr_at_k, ndcg_at_ak, topk]
end

function main(
    data_config::DataLoadingConfig,
    trn_config::TrainingConfig,
    outpath::String;
    lever::Lever,
    k::Int = 10, load_model=false, modelname=""
)
    if data_config.dataset == "ml-1m"
        filename = "ml-1m/ratings.dat"
        item_filename = "ml-1m/movies.dat"
        delim = "::"
        datarow = 1
    elseif data_config.dataset == "ml-25m"
        filename = "ml-25m/ratings.csv"
        item_filename = "ml-25m/movies.csv"
        delim = ","
        datarow = 2 
    end
        
    train_df, test_df, unseen_negatives, subj_df, n_users, n_items, items, all_genres = load_custom(
        data_config,
        filename, lever, delim=delim,
        item_filename=item_filename, datarow=datarow
    )

    # transpose everything so we can access by column
    # + convert data to Int Matrixs to make Julia compiler happer
    my_convert = x -> copy(transpose(convert(Matrix{Int}, x)))

    train = my_convert(train_df[:, ["user", "item"]])
    test = my_convert(test_df)
    #unseen_negatives = my_convert(unseen_negatives)
    subj = my_convert(subj_df)

    n_train = size(train, 2)
    n_test = size(test, 2)
    n_subj = size(subj, 2)
    print("n_users: $n_users, n_items: $n_items, n_train: $n_train, n_test: $n_test, n_subj: $n_subj \n")

    # init the model. Embeddings are drawn from normal, biases start at zero.
    if load_model && isfile(modelname)
        print(modelname, "\n")
        d = load(modelname)
        model = d["model"]
        print("loading at epoch ", model.epoch, "\n")
    else
        dist = Normal(0, trn_config.stdev)
        u_emb = rand(dist, trn_config.embedding_dim, n_users)
        i_emb = rand(dist, trn_config.embedding_dim, n_items)
        u_b = zeros(n_users)
        i_b = zeros(n_items)
        bias = 0.0
        model = MfModel(u_emb, i_emb, u_b, i_b, bias, 1, outpath)
    end
    
    init_hits, init_ndcgs, init_topk = evaluate(model, test, unseen_negatives, k)
    rand_hr = mean(init_hits)
    rand_ndcg = mean(init_ndcgs)
    print("Starting eval: HR: $rand_hr | NDCG: $rand_ndcg\n")

    records = []
    topk_records = []
    for epoch=model.epoch:trn_config.epochs
        model.epoch = epoch
        # print on 1, 10, 20, ... last.
        if epoch == 1 || epoch % 10 == 0 || epoch == trn_config.epochs
            cur_modelname = "$outpath/$epoch.jld"
            cur_resultsname = "$outpath/$epoch.csv"
            save(cur_modelname, "model", model)
        end
        record = Dict{Any,Any}(
            "epoch"=>epoch, "n_train"=>n_train, 
            "lever_size"=>lever.size, "lever_genre"=>lever.genre,
            "lever_type"=>lever.type
        )
        record["epoch"] = epoch

        # == Fit ==
        t_start_fit = time()
        mean_loss = fit!(model, trn_config, train)
        record["fit_time_elapsed"] = time() - t_start_fit
        # == Evaluate ==
        t_start_eval = time()
        rand_users = shuffle(1:n_users)[1:3]
        hits, ndcgs, topk = evaluate(model, test, unseen_negatives, k, print_users=rand_users)
        hr = mean(hits)
        ndcg = mean(ndcgs)
        if epoch == trn_config.epochs # all done
            append!(topk_records, topk)
        end
        record["eval_time_elapsed"] =  time() - t_start_eval
        record["hr"] = hr
        record["ndcg"] = ndcg
        record["loss"] = mean_loss

        subj_hits, subj_ndcgs, subj_topk = evaluate(model, subj, unseen_negatives, k)
        subj_hr = mean(subj_hits)
        subj_ndcg = mean(subj_ndcgs)
        record["subj_hr"] = subj_hr
        record["subj_ndcg"] = subj_ndcg

        # == Print and Save to Dict ==
        round5 = x -> round(x, digits=5)
        s1, s2, s3, s4, s5, s6 = map(round5, [hr, ndcg, subj_hr, mean_loss, record["fit_time_elapsed"], record["eval_time_elapsed"]])
        print("$epoch:\t HR: $s1, NDCG: $s2, SUBJ HR: $s3, TrainL: $s4, Times: $s5, $s6\n")
        
        # == Genre Specific Scores ==
        for genre in all_genres
            matches = items[items[:, genre], "item"]
            mask = [x in matches for x in test_df.item]
            genre_hr = round(mean(hits[mask]), digits=2)
            genre_ndcg = round(mean(ndcgs[mask]), digits=2)
            record["hr_$genre"] = genre_hr
            record["hits_$genre"] = sum(hits[mask])
            record["ndcg_$genre"] = genre_ndcg
        end

        push!(records, record)
        
    end

    results = DataFrame(records[1])
    for record in records[2:end]
        push!(results, record)
    end
    epoch = model.epoch
    final_outname = "$outpath/$epoch-final.csv"
    write_output(results, final_outname)

    print(topk_records, "\n")
    example_results = DataFrame(topk_records[1])
    for record in topk_records[2:end]
        append!(example_results, DataFrame(record))
    end
    print(example_results, "\n")

    worksheet_outname = "$outpath/worksheet.csv"
    write_output(example_results, worksheet_outname)

    return results
end
