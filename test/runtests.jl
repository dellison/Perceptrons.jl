using Test
using Perceptrons

@testset "Perceptrons.jl" begin
    for test in ("binary_perceptrons",
                 "perceptrons")
        include("test_$test.jl")
    end
end
