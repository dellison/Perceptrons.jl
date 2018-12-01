"""
"""
struct BinaryPerceptron{T} <: AbstractPerceptron{T}
    weights::T
end

BinaryPerceptron{T}() where {T<:Dict{<:Any,<:Number}} = BinaryPerceptron(T())
BinaryPerceptron(::Type{Dict}) = BinaryPerceptron{Dict{Any,Number}}()
BinaryPerceptron(T::Type{<:Dict}) = BinaryPerceptron{T}()
BinaryPerceptron(n::Int) = BinaryPerceptron{Vector{Int}}(zeros(n))

function fit_one!(p::BinaryPerceptron, y, ϕ, α=1)
    ŷ = predict(p, ϕ)
    (y != ŷ) && update!(p, y, ϕ, α)
    return ŷ
end

predict(p::BinaryPerceptron, ϕ) = score(p, ϕ) >= 0
score(p::BinaryPerceptron, ϕ) = sum(weight(p, x) for x in ϕ)

weight(p::BinaryPerceptron{<:Dict}, x) = get(p.weights, x, 0)
weight(p::BinaryPerceptron{<:Vector}, x) = p.weights[x]

function update!(p::BinaryPerceptron, y::Bool, ϕ, α=1)
    @assert 0 <= α <= 1 
    !y && (α *= -1)
    for x in ϕ
        p.weights[x] = weight(p, x) + α
    end
end
