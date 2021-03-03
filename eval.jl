

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

    topk = sort(collect(zip(values(d),keys(d))), rev=true)[1:k]

    hr_at_k = get_hr(topk, target)
    ndcg_at_ak = get_ndcg(topk, target)
    return [hr_at_k, ndcg_at_ak]
end

function get_hr(topk, target)
    for tuple in topk
        item = tuple[2]
        if item == target
            return 1
        end
    end
    return 0
end

function get_ndcg(topk, target)
    for i=1:length(topk)
        tuple = topk[i]
        item = tuple[2]
        if item == target
            return log(2) / log(i+1)
        end
    end
    return 0
end

# evaluate is in a different repo, here:
# https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/evaluate.py
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