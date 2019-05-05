"""
    MulticlassPerceptron

todo
"""
mutable struct MulticlassPerceptron{T,N} <: AbstractPerceptron{T}
    w::T
    b::Vector{N}
end

MulticlassPerceptron(nfeats::Int, nclasses::Int) =
    MulticlassPerceptron(zeros(nfeats, nclasses), zeros(nclasses))

function MulticlassPerceptron(T, nfeats::Int, nclasses::Int)
    w = fill(zero(T), nfeats, nclasses)
    b = fill(zero(T), nclasses)
    MulticlassPerceptron(w, b)
end

scores(p::MulticlassPerceptron, x) =
    vec(mapslices(w -> dot(w, x), p.w, dims=1)) + p.b

predict(p, x) = argmax(scores(p, x))

function fit_one!(p::MulticlassPerceptron, x, y, r=1)
    ŷ = predict(p, x)
    ŷ != y && update!(p, x, ŷ, y, r)
    ŷ
end

function update!(p::MulticlassPerceptron, x, ŷ, y, r=1)
    p.b[ŷ] -= r
    p.b[y] += r
    update = x * r
    p.w[:, y] .+= update
    p.w[:, ŷ] .-= update
    p
end

"""
    SparseMulticlassPerceptron

todo
"""
const SparseMulticlassPerceptron{T} =
    MulticlassPerceptron{SparseMatrixCSC{AveragedWeight{T},Int}}

function SparseMulticlassPerceptron{T}(nfeatures::Int,nclasses::Int) where T
    w = spzeros(T,nfeatures,nclasses)
    b = [zero(T) for _=1:nclasses]
    MulticlassPerceptron(w,b)
end
SparseMulticlassPerceptron(nfeatures::Int,nclasses::Int) =
    SparseMulticlassAveragedPerceptron{Int}(nfeatures, nclasses)

"""
    MulticlassAveragedPerceptron

todo
"""
mutable struct MulticlassAveragedPerceptron{W,T} <: AbstractPerceptron{T}
    t::Int
    p::MulticlassPerceptron{W,AveragedWeight{T}}
end

function MulticlassAveragedPerceptron(in::Int, out::Int)
    w = fill(AveragedWeight(), in, out)
    b = fill(AveragedWeight(), out)
    p = MulticlassPerceptron(w, b)
    MulticlassAveragedPerceptron(0, p)
end

"""
    averaged(p)

Average the perceptron's weights, returning a `MulticlassPerceptron`.
"""
function averaged(p::MulticlassAveragedPerceptron)
    t, w, b = p.t, p.p.w, p.p.b
    average(w) = averaged(w, t)
    MulticlassPerceptron(average.(w), average.(b))
end

"""
    average!(p)

Average the perceptron's weights in-place.
"""
function average!(p::MulticlassAveragedPerceptron)
    avg! = w -> average!(w, p.t)
    avg!.(p.p.w)
    avg!.(p.p.b)
    p
end

scores(p::MulticlassAveragedPerceptron, x) = scores(p.p, x)
predict(p::MulticlassAveragedPerceptron, x) = predict(p.p, x)

function fit_one!(p::MulticlassAveragedPerceptron, x, y, r=1)
    p.t += 1
    fit_one!(p.p, x, y, r)
end

const SparseMulticlassAveragedPerceptron{T} = MulticlassAveragedPerceptron{SparseMatrixCSC{AveragedWeight{T},Int}}

function SparseMulticlassAveragedPerceptron{T}(nfeatures::Int,nclasses::Int) where T
    w = spzeros(AveragedWeight{T},nfeatures,nclasses)
    b = [AveragedWeight(T) for _=1:nclasses]
    MulticlassAveragedPerceptron(0,MulticlassPerceptron(w,b))
end
SparseMulticlassAveragedPerceptron(nfeatures::Int,nclasses::Int) =
    SparseMulticlassAveragedPerceptron{Int}(nfeatures, nclasses)

