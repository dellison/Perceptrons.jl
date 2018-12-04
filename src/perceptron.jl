struct BinaryPerceptron{T} <: AbstractPerceptron{T}
    weights::T
end

BinaryPerceptron{T}() where {T<:Dict{<:Any,<:Number}} = BinaryPerceptron(T())
BinaryPerceptron(::Type{Dict}) = BinaryPerceptron{Dict{Any,Number}}()
BinaryPerceptron(T::Type{<:Dict}) = BinaryPerceptron{T}()
BinaryPerceptron(n::Int) = BinaryPerceptron{Vector{Int}}(zeros(n))

function fit_one!(p::BinaryPerceptron, ϕ, y, α=1)
    ŷ = predict(p, ϕ)
    (y != ŷ) && update!(p, ϕ, y, α)
    return ŷ
end

predict(p::BinaryPerceptron, ϕ) = score(p, ϕ) >= 0
score(p::BinaryPerceptron, ϕ) = sum(weight(p, x) for x in ϕ)

weight(p::BinaryPerceptron{<:Dict}, x) = get(p.weights, x, 0)
weight(p::BinaryPerceptron{<:Vector}, x) = p.weights[x]

function update!(p::BinaryPerceptron, ϕ, y::Bool, α=1)
    @assert 0 <= α <= 1 
    !y && (α *= -1)
    for x in ϕ
        p.weights[x] = weight(p, x) + α
    end
end



struct Perceptron{T} <: AbstractPerceptron{T}
    classes
    weights::T
end

Perceptron(::Type{Dict}, classes) = Perceptron(classes, Dict{Any,Dict{Any,Number}}())
Perceptron(nclasses::Int, nfeats::Int) = Perceptron(1:nclasses, zeros(nclasses, nfeats))

weight(p::Perceptron{<:Dict}, x, y) = get(weights(p, x), y, 0)
weight(p::Perceptron{<:AbstractMatrix}, x, y) = p.weights[x, y]

score(p::Perceptron, ϕ, y) = sum(weight(p, x, y) for x in ϕ)

predict(p::Perceptron, ϕ) = argmax(y -> score(p, ϕ, y), p.classes)

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

function update!(p::Perceptron{<:Dict}, ϕ, ŷ, y, α=1)
    for x in ϕ
        ws = weights(p, x)
        ws[y] += α
        ws[ŷ] -= α
    end
end

function fit_one!(p::Perceptron, ϕ, y, α=1)
    ŷ = predict(p, ϕ)
    y != ŷ && update!(p, ϕ, ŷ, y, α)
    return ŷ
end
