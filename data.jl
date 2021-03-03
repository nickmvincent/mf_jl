using CSV, DataFrames, Printf

function load(
    filename; delim="\t", strat="leave_out_last", frac=1.0, item_filename="",
    strike_size=0, strike_genre=""
)
    df = CSV.read(filename, DataFrame, delim=delim, header=["orig_user", "orig_item", "rating", "utc"])

    if frac != 1.0
        n_rows = size(df)[1]
        new_n_rows = Int(floor(n_rows * frac))
        df = df[shuffle(1:end)[1:new_n_rows], :]
    end

    if item_filename != ""
        item_df = CSV.read(item_filename, DataFrame, delim=delim, header=["orig_item", "name", "genres"])
        genre_arrs = [split(x, '|') for x in item_df.genres]
        all_genres = []
        for genre_arr in genre_arrs
            append!(all_genres, genre_arr)
        end
        all_genres = convert(Array{String}, unique(all_genres))
        for genre in all_genres
            vals = [genre in x for x in genre_arrs]
            item_df[:, genre] = vals
        end

    else
        item_df = []
        all_genres = []
    end

    
    
    # can implement a hit threshold here.
    df[:, "hit"] = df.rating .>= 0

    #do strike here
    if strike_size > 0
        if strike_genre != ""
            elig_observations = item_df[item_df[:, strike_genre], "orig_item"]
            n_obs = size(elig_observations)[1]
            print(strike_genre, n_obs, "\n")
            n_drop = Int(floor(strike_size * n_obs))

            print("$strike_genre, $n_obs, $n_drop\n")

            drop = shuffle(elig_observations)[1:n_drop]

            mask = [!(x in drop) for x in df.orig_item]
            df = df[mask, :]
        else
            n_orig_users = length(unique(df.orig_user))
            elig_users = 1:n_orig_users
            n_drop = Int(floor(strike_size * n_orig_users))
            strike_users = shuffle(elig_users)[1:n_drop]
            mask = [!(x in strike_users) for x in df.orig_user]
            df = df[mask, :]

        end


    end

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
    function get(x)
        if haskey(item_d, x)
            return item_d[x]
        else
            return -1
        end
    end
    item_df[:, "item"] = map(get, item_df.orig_item)




    num_items = length(unique(df.item))
    num_users = length(unique(df.user))

    cols = ["user", "item", "hit"]
    
    if strat == "random_holdout"
        shuffled = df[shuffle(1:end), :]
        num_train = Int(length(df.user) * 0.8)
        train = df[1:num_train, cols]
        test = df[num_train:end, cols]

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
    end

    return train, test_hits, test_negatives, num_users, num_items, item_df, all_genres
    
end