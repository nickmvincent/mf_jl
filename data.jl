using CSV, DataFrames

function load(filename)
    df = CSV.read(filename, DataFrame, delim='\t', header=["user", "item", "rating", "utc"])
    df[:, "hit"] = df.rating .>= 3
    return df[:, ["user", "item", "hit"]]
end