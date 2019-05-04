"""
    Perceptron

todo
"""
mutable struct Perceptron{T,N} <: AbstractPerceptron{T}
    w::T
    b::N
end

Perceptron(nfeats::Int) = Perceptron(zeros(Int,nfeats), 0)
Perceptron(w::AbstractVector) = Perceptron(w, zero(eltype(w)))

predict(p::Perceptron, x) = score(p, x) > 0
score(p::Perceptron,   x) = dot(p.w, x) + p.b

function fit_one!(p::Perceptron, x, y::Bool, r=1)
    ŷ = predict(p, x)
    if ŷ != y
        update!(p, x, y, r)
    end
    ŷ
end

function update!(p::Perceptron, x, y::Bool, r=1)
    !y && (r *=-1)
    p.b += r
    p.w .+= (x * r)
end

const SparsePerceptron{T} = Perceptron{SparseVector,T}
