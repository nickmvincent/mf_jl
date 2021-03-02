using CSV, DataFrames

function load(filename; delim="\t", strat="leave_out_last")
    df = CSV.read(filename, DataFrame, delim=delim, header=["orig_user", "orig_item", "rating", "utc"])
    
    # can implement a hit threshold here.
    df[:, "hit"] = df.rating .>= 0

    # renumber users and items so they are continuous indices
    count = 1
    item_d = Dict()
    for item in df.orig_item
        if !haskey(item_d, item)
            item_d[item] = count
            count += 1
        end
    end

    count = 1
    user_d = Dict()
    for user in df.orig_user
        if !haskey(user_d, user)
            user_d[user] = count
            count += 1
        end
    end

    df[:, "user"] = map(x -> user_d[x], df.orig_user)
    df[:, "item"] = map(x -> item_d[x], df.orig_item)
    num_items = length(unique(df.item))
    num_users = length(unique(df.user))

    cols = ["user", "item", "hit"]
    
    if strat == "random_holdout"
        shuffled = df[shuffle(1:end), :]
        num_train = Int(length(df.user) * 0.8)
        train = df[1:num_train, cols]
        test = df[num_train:end, cols]
        return train, test

    elseif strat == "leave_out_last"
        df = sort(df, [:user, :utc])
        num_negatives_per_user = 100
        df[:, "is_test"] = fill(false, length(df.user))
        test_negatives = zeros(num_users, num_negatives_per_user)
        all_items = unique(df.item)
        for group in groupby(df, :user)
            group[end, "is_test"] = true
            seen_items = group[1:end-1, "item"]
            unseen_items = setdiff(all_items, seen_items)
            test_negatives[group[1, "user"], :] = shuffle(unseen_items)[1:num_negatives_per_user]
        end

        train = df[df.is_test .== false, cols]
        test_hits = df[df.is_test .== true, cols]
        return train, test_hits, test_negatives, num_users, num_items
            
    end
    
end