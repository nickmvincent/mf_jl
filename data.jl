using CSV, DataFrames, Printf, Random

function get_item_df(item_filename, delim)
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
    item_df, all_genres
end

function load_custom(
    filename, lever; delim="\t", strat="leave_out_last", frac=1.0, item_filename=""
)
    df = CSV.read(filename, DataFrame, delim=delim, header=["orig_user", "orig_item", "rating", "utc"])
    # can implement a hit threshold here.
    # currently we copy NCF and use ALL Ratings as hit
    df[:, "hit"] = df.rating .>= 0

    #==
    Sort by user and time to get the "last" rating for each user.
    This will be used for testing.
    ==#
    df = sort(df, [:orig_user, :utc])
    num_negatives_per_user = 100
    df[:, "is_test"] = fill(false, size(df)[1])
    hidden_negatives = zeros(length(unique(df.orig_user)), num_negatives_per_user)
    all_items = unique(df.orig_item)
    for group in groupby(df, :orig_user)
        group[end, "is_test"] = true
        seen_items = group[1:end-1, "orig_item"]
        unseen_items = setdiff(all_items, seen_items)
        hidden_negatives[group[1, "orig_user"], :] = shuffle(unseen_items)[1:num_negatives_per_user]
    end

    hidden_hits = df[df.is_test .== true, :]
    df = df[df.is_test .== false, :]

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
    # handle the fact that some items might appear ONLY in hidden
    
    function get_item(x)
        if haskey(item_d, x)
            return item_d[x]
        else
            return -1
        end
    end

    function get_user(x)
        if haskey(user_d, x)
            return user_d[x]
        else
            return -1
        end
    end

    df[:, "user"] = map(get_user, df.orig_user)
    df[:, "item"] = map(get_item, df.orig_item)

    hidden_hits[:, "user"] = map(get_user, hidden_hits.orig_user)
    hidden_hits[:, "item"] = map(get_item, hidden_hits.orig_item)
    hidden_negatives = map(get_item, hidden_negatives)
    
    n_items = length(unique(df.item))
    n_users = length(unique(df.user))


    # Here, we can toss some data to train faster, e.g. for testing

    if frac != 1.0
        n_rows = size(df)[1]
        new_n_rows = Int(floor(n_rows * frac))
        df = df[shuffle(1:end)[1:new_n_rows], :]
    end

    
    item_df, all_genres = get_item_df(item_filename, delim)
    item_df[:, "item"] = map(get_item, item_df.orig_item)

    # do strike here
    if lever.size > 0
        if lever.genre != "All" # GENRE SPECIFIC
            elig_observations = item_df[item_df[:, lever.genre], "orig_item"]
            # use the ORIG ITEM id. We'll check against the orig_item column below in `mask =`` 
            n_obs = size(elig_observations, 1)
            #print(lever.genre, n_obs, "\n")
            n_drop = Int(floor(lever.size * n_obs))

            #print("$lever.genre, $n_obs, $n_drop\n")

            obs_at_stake = shuffle(elig_observations)[1:n_drop]
            if lever.type == "strike"
                mask = [!(x in obs_at_stake) for x in df.orig_item]
                df = df[mask, :]
            elseif lever.type == "poison"
                mask = [x in obs_at_stake for x in df.orig_item]
                if lever.genre2 != false
                    elig_switches = unique(item_df[item_df[:, lever.genre], "item"]) # as the NEW item id
                else
                    elig_switches = unique(df.item)
                end
                df[mask, :item] = rand(elig_switches, sum(mask))
            end

        else # NOT GENRE SPECIFIC
            n_orig_users = length(unique(df.orig_user))
            elig_users = 1:n_orig_users
            n_lever_users = Int(floor(lever.size * n_orig_users))
            lever_users = shuffle(elig_users)[1:n_lever_users]
            mask = [(x in lever_users) for x in df.orig_user]

            if lever.genre != "All"
                matching_observations = item_df[item_df[:, lever.genre], "orig_item"]
                mask = [x in matching_observations for x in df.orig_item] .& mask
            end

            if lever.type == "strike"
                df = df[.!mask, :]
            elseif lever.type == "poison"
                elig_switches = unique(df.item)
                df[mask, :item] = rand(elig_switches, sum(mask))
            end
        end
    end
    
    # if strat == "random_holdout"
    #     shuffled = df[shuffle(1:end), :]
    #     num_train = Int(length(df.user) * 0.8)
    #     train = df[1:num_train, cols]
    #     test = df[num_train:end, cols]

    cols = ["user", "item", "hit"]

    return df[:, cols], hidden_hits[:, cols], hidden_negatives, n_users, n_items, item_df, all_genres
    
end