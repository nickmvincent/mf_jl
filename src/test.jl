include("./mf_simple.jl")

data_config = DataLoadingConfig(
	10, "2000-12-29T23:42:56.4",
	"ml-1m", 1.0
)
lever = Lever(0, "All", false, "strike")
epochs = 2
trn_config = TrainingConfig(
	epochs, 15, # because bias is one of the dimensions
	0.005, #reg
    8, 
	0.002, #lr
    0.1
)

outpath = "results/test"
modelname = "$outpath/$epochs.jld"
mkpath(outpath)

res = main(
    data_config, trn_config, outpath,
    lever=lever,
    load_model=false,
    modelname=modelname
)

# try loading the prev model
res = main(
    data_config, trn_config, outpath,
    lever=lever,
    load_model=false,
    modelname=modelname,
    load_model=true
)

# try without a strike
lever2 = Lever(0.1, "All", false, "strike")


# try with poison
lever = Lever(0.1, "All", false, "poison")

