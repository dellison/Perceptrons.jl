mutable struct LazyWeight{T}
    timestamp::Int
    weight::T
    summed_weight::T
end

LazyWeight(T::Type{<:Number}, t = 0) = LazyWeight{T}(t, zero(T), zero(T))
LazyWeight(t::Int = 0) = LazyWeight(Int, t)

import Base: +
+(w::LazyWeight, x) = w.weight + x
+(x, w::LazyWeight) = x + w.weight
+(w1::LazyWeight, w2::LazyWeight) = w1.weight + w2.weight

Base.isless(w::LazyWeight, x) = isless(w.weight, x)
Base.isless(x, w::LazyWeight) = isless(x, w.weight)
Base.isless(w1::LazyWeight, w2::LazyWeight) = isless(w1.weight, w2.weight)

weight(w::LazyWeight) = w.weight

function freshen!(w::LazyWeight, t)
    w.summed_weight += (t - w.timestamp) * w.weight
    w.timestamp = t
end

function update!(w::LazyWeight, value, t)
    freshen!(w, t)
    w.weight += value
end

function average!(w::LazyWeight, t)
    freshen!(w)
    w.weight = w.summed_weight / t
end


mutable struct BinaryAveragedPerceptron{T} <: AbstractPerceptron{T}
    weights::T
    time::Int
end

BinaryAveragedPerceptron{T}() where {T<:Dict{<:Any,<:LazyWeight}} = BinaryAveragedPerceptron(T(),1)
BinaryAveragedPerceptron() = BinaryAveragedPerceptron(Dict)
BinaryAveragedPerceptron(::Type{Dict}) = BinaryAveragedPerceptron{Dict{Any,LazyWeight}}()
BinaryAveragedPerceptron(T::Type{<:Dict}) = BinaryAveragedPerceptron{T}()
BinaryAveragedPerceptron(n::Int) = BinaryAveragedPerceptron{Vector{LazyWeight}}([LazyWeight() for _ in 1:n],1)

function fit_one!(p::BinaryAveragedPerceptron, y, ϕ, α = 1)
    ŷ = predict(p, ϕ)
    (y != ŷ) && update!(p, y, ϕ, α)
    p.time += 1
    return ŷ
end

predict(p::BinaryAveragedPerceptron, ϕ) = score(p, ϕ) >= 0

score(p::BinaryAveragedPerceptron, ϕ) = sum(weight(weight(p, x)) for x in ϕ)

function update!(p::BinaryAveragedPerceptron, y::Bool, ϕ, α = 1)
    @assert 0 <= α <= 1
    !y && (α *= -1)
    for x in ϕ
        update!(weight(p, x), α, p.time)
    end
end

weight(p::BinaryAveragedPerceptron{<:Dict}, x) =
    get!(() -> LazyWeight(p.time), p.weights, x)
weight(p::BinaryAveragedPerceptron{<:Vector}, x) = p.weights[x]
