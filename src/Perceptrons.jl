module Perceptrons

Base.argmax(f::Function, xs) = first(sort(collect(xs), by = f, rev = true))

"""
"""
abstract type AbstractPerceptron{T} end

include("perceptron.jl")
include("averaged_perceptron.jl")

end # module
