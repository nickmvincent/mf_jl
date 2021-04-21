using CSV, DataFrames, Printf, Random, Dates


"""
Loads a DataFrame with metadata about individual items. 
One row per item.
"""
function get_item_df(item_filename, delim, datarow=1)
    item_df = CSV.read(item_filename, DataFrame, datarow=datarow, delim=delim, header=["orig_item", "name", "genres"])
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
    return item_df, all_genres
end


"""
Sort by user and time to get the "last" rating for each user.
This will be used for testing.

Modifies df and return hidden negatives and hidden hits

This replicates evaluation fron Neural Collaborative Filtering etc.
"""
function leave_out_test(df::DataFrame, n_test_negatives::Any=100, cutoff=false)
    sort!(df, [:orig_user, :utc])
    df[:, "is_test"] = fill(false, size(df, 1))
    n_users = length(unique(df.orig_user))

    # if n_test_negatives == 0
    #     hidden_negatives = zeros(n_users, length(unique(df.orig_item)))
    # else
    #     hidden_negatives = zeros(n_users, n_test_negatives)
    # end
    negatives = Dict()
    all_items = unique(df.orig_item)

    for group in groupby(df, :orig_user)
        user_id = group[1, "orig_user"]
        if cutoff == false
            group[end, "is_test"] = true
            seen_items = group[1:end-1, "orig_item"]
        else
            cutoff_utc = datetime2unix(DateTime(cutoff))
            after_cutoff_mask = group.utc .> cutoff_utc
            group[after_cutoff_mask, "is_test"] .= true
            seen_items = group[.!after_cutoff_mask, "orig_item"]
        end
        
        unseen_items = setdiff(all_items, seen_items)
        if n_test_negatives == 0
            #hidden_negatives[,1:length(unseen_items)] = shuffle(unseen_items)[1:end]
            negatives[user_id] = shuffle(unseen_items)[1:end]
        else
            negatives[user_id] = shuffle(unseen_items)[1:n_test_negatives]
        end
    end
    
    # set negatives for unseen users
    negatives[-1] = all_items

    hidden_df = df[df.is_test .== true, :]
    return df[df.is_test .== false, :], hidden_df, negatives
end

function get_renumbering_dict(df, col)
    count = 1
    d = Dict()
    for x in df[:, col]
        if !haskey(d, x)
            d[x] = count
            count += 1
        end
    end

    return d
end


function renumber!(train_df, item_df, test_df, negatives)
    # renumber users and items so they are continuous indices
    user_d = get_renumbering_dict(train_df, :orig_user)
    item_d = get_renumbering_dict(train_df, :orig_item)

    # if an item or user does not appear in the TRAIN set (`train_df`)
    # then it will be assigned value "-1"
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

    train_df[:, "user"] = map(get_user, train_df.orig_user)
    train_df[:, "item"] = map(get_item, train_df.orig_item)

    test_df[:, "user"] = map(get_user, test_df.orig_user)
    test_df[:, "item"] = map(get_item, test_df.orig_item)

    item_df[:, "item"] = map(get_item, item_df.orig_item)

    renumbered_negatives = Dict()
    for (user, items) in negatives
        new_user_id = get_user(user)
        new_item_ids = [get_item(item) for item in items]
        if new_user_id != -1
            renumbered_negatives[new_user_id] = new_item_ids
        end
    end
    return renumbered_negatives
end

function toss_data(df, frac)
    n_rows = size(df)[1]
    new_n_rows = Int(floor(n_rows * frac))
    return df[shuffle(1:end)[1:new_n_rows], :]
end

function sample_frac_users(df, col, frac)
    unique_vals = unique(df[:, col])
    n = length(unique_vals)
    elig = 1:size(unique_vals, 1)
    n_in_sample = Int(floor(frac * n))
    sample = shuffle(elig)[1:n_in_sample]
    return sample
end

function targeted_lever(df, lever, lever_users, item_df)
    # figure out which items might be "up for grabs" (to delete or poison)
    elig_items = item_df[item_df[:, lever.genre], "item"]

    matching_obs_mask = [(x in elig_items) for x in df.item] .& [x in lever_users for x in df.user]
    if lever.type == "strike"
        # do the dropping!
        df = df[.!matching_obs_mask, :]

    elseif lever.type == "poison"
        if lever.genre2 != false
            # must be an item from genre2
            items_in_genre2 = item_df[:, lever.genre2]
            elig_switches = unique(item_df[items_in_genre2, "item"])
        else
            # can be any item
            elig_switches = unique(df.item)
        end
        # randomly pick n_affected switched and apply them using "mask"
        df[matching_obs_mask, :item] = rand(elig_switches, sum(matching_obs_mask))
    end
    return df
end

function general_lever(df, lever, lever_users)
    mask = [(x in lever_users) for x in df.user]

    if lever.type == "strike"
        return df[.!mask, :]

    elseif lever.type == "poison"
        elig_switches = unique(df.item)
        df[mask, :item] = rand(elig_switches, sum(mask))
        # return test df as is
        return df
    end
end

"""
hits will depend on the size of the test set (what's the cutoff)
negatives will be one row per user.
"""
function create_test_without_participants(hits, negatives, lever_users)
    not_participant_mask = [!(x in lever_users) for x in hits.user]
    #indices = [i for i in 1:size(negatives, 1) if !(i in lever_users)]
    non_participant_hits = hits[not_participant_mask, :]
    #non_participant_negatives = negatives[indices, :]
    return non_participant_hits
end
    
"""
Custom data loader.

Returns many
 train_df - train as DataFrame
 test_df - test as DataFrame
 negatives per user
"""
function load_custom(
    config, filename, lever; delim::String="\t",
    item_filename::String="", datarow::Int=1,
)
    # handle headers in file.
    df = CSV.read(filename, DataFrame, delim=delim, datarow=datarow, header=["orig_user", "orig_item", "rating", "utc"])
    
    # Here, we can toss some data to make everything faster, e.g. for writing test scripts.
    if config.frac != 1.0
        df = toss_data(df, config.frac)
    end

    # currently we copy NCF paper and use ALL Ratings as hit
    df[:, "hit"] = df.rating .>= 0

    # leave-out-some data splitting (either one test per user, or use a data cutoff.)
    train_df, test_df, unseen_negatives = leave_out_test(df, config.n_test_negatives, config.cutoff)

    # get item info (e.g. movie genres)
    item_df, all_genres = get_item_df(item_filename, delim, datarow)

    # renumber user and items so they can act as matrix indices
    # modifies df, item_df, hidden_htis. Return negatives (matrix)
    unseen_negatives = renumber!(train_df, item_df, test_df, unseen_negatives)
        
    n_items = length(unique(train_df.item))
    n_users = length(unique(train_df.user))

    # participants = people engaging in data leverage. e.g. they will delete their data or poison it.
    participants = sample_frac_users(train_df, :user, lever.size)
    if lever.size > 0
        if lever.genre != "All"
            df = targeted_lever(train_df, lever, participants, item_df)
        else
            df = general_lever(train_df, lever, participants)
        end 
    end

    # Also create a test set WITHOUT leverage participants.
    subj_df = create_test_without_participants(test_df, unseen_negatives, participants)

    # columns we will keep from the dataframes.
    cols = ["user", "item", "hit"]

    return train_df[:, cols], test_df[:, cols], unseen_negatives,
        subj_df[:, cols], n_users, n_items, item_df, all_genres
    
end
