using VLDataScienceMachineLearningPackage
using Statistics
using JLD2
using LinearAlgebra
using Plots
using Distances
using NNlib
using Distributions
using PrettyTables
using DataFrames
using StatsBase
using IJulia
using Random

include(joinpath(@__DIR__, "src", "BagOfWords.jl"))
include(joinpath(@__DIR__, "src", "TFIDF.jl"))
include(joinpath(@__DIR__, "src", "PMI.jl"))
include(joinpath(@__DIR__, "src", "CBOW.jl"))
