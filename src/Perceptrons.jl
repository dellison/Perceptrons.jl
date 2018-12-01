module Perceptrons

Base.argmax(f::Function, xs) = first(sort(collect(xs), by = f, rev = true))

"""
"""
abstract type AbstractPerceptron{T} end


include("binary_perceptron.jl")
include("perceptron.jl")


end # module
