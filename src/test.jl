include("./mf_simple.jl")

load_config = DataLoadingConfig(
	0, "2000-12-29T23:42:56.4",
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

lever_size = 0.1

outpath = "results/test"
modelname = "$outpath/$epochs.jld"
mkpath(outpath)

res = main(
    load_config, trn_config, outpath,
    lever=lever,
    load_model=false,
    modelname=modelname
)

res = main(
    load_config, trn_config, outpath,
    lever=lever,
    load_model=false,
    modelname=modelname,
    load_model=true
)