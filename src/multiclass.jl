"""
"""
mutable struct MulticlassPerceptron{T,B} <: AbstractPerceptron{T}
    w::T
    b::B
end

MulticlassPerceptron(nfeats::Int, nclasses::Int) =
    MulticlassPerceptron(zeros(nfeats, nclasses), zeros(nclasses))

scores(p::MulticlassPerceptron, x) = vec(mapslices(w -> dot(w, x), p.w, dims=1))

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
