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

"""
    SparsePerceptron

todo
"""
const SparsePerceptron{T} = Perceptron{SparseVector,T}

"""
    AveragedPerceptron

todo
"""
mutable struct AveragedPerceptron{W,T} <: AbstractPerceptron{T}
    t::Int
    p::Perceptron{W,AveragedWeight{T}}
end

function AveragedPerceptron(nfeats::Int)
    w = [AveragedWeight() for _=1:nfeats]
    b = AveragedWeight()
    AveragedPerceptron(0, Perceptron(w, b))
end

score(p::AveragedPerceptron, x) = _w(score(p.p, x))
predict(p::AveragedPerceptron, x) = predict(p.p, x)

function fit_one!(p::AveragedPerceptron, x, y, r=1)
    p.t += 1
    fit_one!(p.p, x, y, r)
end

update!(p::AveragedPerceptron, x, y::Bool, r=1) = update(p.p, x, y, r)

function average!(p::AveragedPerceptron)
    for w in param, param in (p.w, p.b)
        average!(w, p.t)
    end
end

"""
    SparseAveragedPerceptron

todo
"""
const SparseAveragedPerceptron{T} = AveragedPerceptron{SparseVector,T}
