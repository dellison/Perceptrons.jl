module Perceptrons

export Perceptron, MulticlassPerceptron

using LinearAlgebra, SparseArrays

Base.argmax(f::Function, xs) = first(sort(collect(xs), by = f, rev = true))

LinearAlgebra.dot(dict::Dict, x) = sum(get(dict, feature, 0) for feature in x)

"""
    AbstractPerceptron

"""
abstract type AbstractPerceptron{T} end

"""
    fit!(p, data, r=1)

todo
"""
function fit!(p::AbstractPerceptron, data, r=1)
    for (x, y) in data
        fit_one!(p, x, y, r)
    end
end

include("binary.jl")
include("multiclass.jl")

end # module
