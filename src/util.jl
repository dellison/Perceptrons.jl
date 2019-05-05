import Base.==, Base.+, Base.*, Base.-, Base./
import Base.isless, Base.round, Base.zero
import Base.iterate, Base.show

"""
    AveragedWeight

Weight in an averaged perceptron.

It keeps track of its sum over time as it's updated, and can be
efficiently averaged at the end of training.
"""
mutable struct AveragedWeight{T}
    weight::T
    t::Int
    summed::T
end
    
AveragedWeight(T::Type{<:Number}, t=0) = AveragedWeight{T}(t,zero(T),zero(T))
AveragedWeight(t=0) = AveragedWeight(typeof(t),t)

_w(x) = x
_w(w::AveragedWeight) = w.weight

"""
    tick!(w, t)

Make weight current as of timestamp `t`.
"""
function tick!(w::AveragedWeight, t)
    w.summed += (t - w.t) * w.weight
    w.t = t
end

"""
    update!(w::AveragedWeight, t, r=1)


"""
function update!(w::AveragedWeight, t, r=1)
    tick!(w, t)
    w.weight += r
end

# averaged weight over its lifetime, at time t.
averaged(w::AveragedWeight, t) = (w.summed + (t - w.t) * w.weight) / t

# average weight in-place over its lifetime at time t
function average!(w::AveragedWeight, t)
    tick!(w, t)
    avg = w.summed / t
    try
        w.weight = avg
    catch err
        if err isa InexactError
            w.weight = round(typeof(w.weight), avg)
        else
            rethrow(err)
        end
    end
    return w
end

Base.isless(w::AveragedWeight, x) = isless(w.weight, x)
Base.isless(x, w::AveragedWeight) = isless(x, w.weight)
Base.isless(w1::AveragedWeight, w2::AveragedWeight) = isless(w1.weight, w2.weight)

==(w::AveragedWeight, x) = _w(w) == _w(x)

+(w::AveragedWeight, x) = AveragedWeight(_w(w) + _w(x), w.t, w.summed + _w(x))
+(x, w::AveragedWeight) = AveragedWeight(_w(w) + _w(x), w.t, w.summed + _w(x))
-(w::AveragedWeight, x) = AveragedWeight(_w(w) - _w(x), w.t, w.summed - _w(x))
-(x, w::AveragedWeight) = AveragedWeight(_w(w) - _w(x), w.t, w.summed - _w(x))
*(w::AveragedWeight, x) = AveragedWeight(_w(w) * _w(x), w.t, w.summed * _w(x))
*(x, w::AveragedWeight) = AveragedWeight(_w(w) * _w(x), w.t, w.summed * _w(x))
/(w::AveragedWeight, x) = AveragedWeight(_w(w) / _w(x), w.t, w.summed / _w(x))
/(x, w::AveragedWeight) = AveragedWeight(_w(w) / _w(x), w.t, w.summed / _w(x))

Base.round(w::AveragedWeight) = AveragedWeight(round(w.weight), w.t, w.summed)
Base.zero(T::AveragedWeight) = AveragedWeight(T)
Base.zero(T::Type{<:AveragedWeight}) = AveragedWeight(T)

Base.iterate(w::AveragedWeight, state...) = iterate(w.weight, state...)

Base.show(io::IO, w::AveragedWeight) = print(io, w.weight)
