mutable struct LazyWeight{T}
    timestamp::Int
    weight::T
    summed_weight::T
end

LazyWeight(T::Type{<:Number}, t = 0) = LazyWeight{T}(t, zero(T), zero(T))
LazyWeight(t::Number = 0) = LazyWeight(Number, t)

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
    freshen!(w, t)
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

function finalize!(p::BinaryAveragedPerceptron{<:Dict})
    for (feat, weight) in p.weights
        average!(weight, p.time)
    end
end

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

function train!(p::BinaryAveragedPerceptron, Xs, Ys; epochs=20, alpha=1)
    for epoch = 1:epochs, (x, y) in shuffle(zip(Xs, Ys))
        fit_one!(p, y, x, alpha)
    end
    finalize!(p)
    return p
end

function train_binary_avg_perceptron(Xs, Ys; epochs=20, alpha=1)
    train!(BinaryAveragePerceptron(), Xs, Ys; epochs=epochs, alpha=alpha)
end


mutable struct AveragedPerceptron{T} <: AbstractPerceptron{T}
    weights::T
    classes
    time::Int
end

AveragedPerceptron{T}(classes) where T = AveragedPerceptron(T(),classes,1)
AveragedPerceptron(::Type{Dict}, classes) = AveragedPerceptron{Dict{Any,Dict{Any,LazyWeight}}}(classes)
AveragedPerceptron(T::Type{<:Dict}, classes) = AveragedPerceptron{T}(classes)
AveragedPerceptron(n::Int) = AveragedPerceptron{Vector{LazyWeight}}([LazyWeight() for _ in 1:n],1)

function finalize!(p::AveragedPerceptron{<:Dict})
    for (feat, weights) in p.weights, (class, weight) in weights
        average!(weight, p.time)
    end
end

function fit_one!(p::AveragedPerceptron, y, ϕ, α=1)
    ŷ = predict(p, ϕ)
    p.time += 1
    y != ŷ && update!(p, ŷ, y, ϕ, α)
    return ŷ
end

function update!(p::AveragedPerceptron, ŷ, y, ϕ, α = 1)
    for x in ϕ
        update!(weight(p, x, y), α, p.time)
        update!(weight(p, x, ŷ), -α, p.time)
    end
end

predict(p::AveragedPerceptron, ϕ) = argmax(y -> score(p, y, ϕ), p.classes)

score(p::AveragedPerceptron, y, ϕ) = sum(weight(weight(p, x, y)) for x in ϕ)

weight(p::AveragedPerceptron{<:Dict}, x, y) =
    get!(() -> LazyWeight(p.time), weights(p, x), y)
weight(p::AveragedPerceptron{<:AbstractMatrix}, x, y) = p.weights[x, y]

weights(p::AveragedPerceptron{<:AbstractMatrix}, x) = p.weights[:, x]
weights(p::AveragedPerceptron{<:Dict}, x) =
    get!(() -> Dict(c => LazyWeight(p.time) for c in p.classes), p.weights, x)


function train!(p::AveragedPerceptron, Xs, Ys; epochs=20, alpha=1)
    data = collect(zip(Xs, Ys))
    for epoch = 1:epochs, (x, y) in shuffle(data)
        fit_one!(p, y, x, alpha)
    end
    finalize!(p)
    return p
end

function train_avg_perceptron(Xs, Ys; epochs=20, alpha=1)
    p = AveragedPerceptron(unique(Ys))
    train!(p, Xs, Ys, epochs=epochs, alpha=alpha)
    return p
end
