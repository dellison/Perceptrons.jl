using Perceptrons, Test

@testset "Perceptrons.jl" begin
    using Perceptrons: update!, predict, score, scores
    include("test_binary_perceptrons.jl")
    include("test_multiclass_perceptrons.jl")
    include("test_averaged_perceptrons.jl")
end
