"""
"""
struct Perceptron{T} <: AbstractPerceptron{T}
    classes
    weights::T
end

Perceptron(::Type{Dict}, classes) = Perceptron(classes, Dict{Any, Dict{Any, Number}}())
Perceptron(nclasses::Int, nfeats::Int) = Perceptron(1:nclasses, zeros(nclasses, nfeats))

weight(p::Perceptron{<:Dict}, x, y) = get(weights(p, x), y, 0)
weight(p::Perceptron{<:AbstractMatrix}, x, y) = p.weights[x, y]

score(p::Perceptron, y, ϕ) = sum(weight(p, x, y) for x in ϕ)

predict(p::Perceptron, ϕ) = argmax(y -> score(p, y, ϕ), p.classes)

function scores(p::Perceptron{<:Dict}, ϕ)
    dict = Dict(c => 0 for c in p.classes)
    for ϕi in ϕ, (c, w) in weights(p, ϕi)
        dict[c] += w
    end
    return dict
end

weights(p::Perceptron{<:AbstractMatrix}, x) = p.weights[:, x]
weights(p::Perceptron{<:Dict}, x) =
    get!(() -> Dict(c => 0 for c in p.classes), p.weights, x)

function update!(p::Perceptron{<:Dict}, ŷ, y, ϕ, α=1)
    for x in ϕ
        ws = weights(p, x)
        ws[y] += α
        ws[ŷ] -= α
    end
end

function fit_one!(p::Perceptron, y, ϕ, α=1)
    ŷ = predict(p, ϕ)
    y != ŷ && update!(p, ŷ, y, ϕ, α)
    return ŷ
end
