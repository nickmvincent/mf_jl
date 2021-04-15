
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

