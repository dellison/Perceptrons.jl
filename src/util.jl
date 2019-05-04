mutable struct AveragedWeight{T}
    weight::T
    t::Int
    summed::T
end
    
AveragedWeight(T::Type{<:Number}, t=0) = AveragedWeight{T}(t,zero(T),zero(T))
AveragedWeight(t=0) = AveragedWeight(typeof(t),t)

import Base.==, Base.+, Base.*, Base.-, Base./, Base.isless, Base.zero
import Base.iterate, Base.show

Base.isless(w::AveragedWeight, x) = isless(w.weight, x)
Base.isless(x, w::AveragedWeight) = isless(x, w.weight)
Base.isless(w1::AveragedWeight, w2::AveragedWeight) = isless(w1.weight, w2.weight)

==(w::AveragedWeight, x) = _w(w) == _w(x)

+(w::AveragedWeight, x) = AveragedWeight(_w(w) + _w(x), w.t, w.summed)
+(x, w::AveragedWeight) = AveragedWeight(_w(w) + _w(x), w.t, w.summed)
-(w::AveragedWeight, x) = AveragedWeight(_w(w) - _w(x), w.t, w.summed)
-(x, w::AveragedWeight) = AveragedWeight(_w(w) - _w(x), w.t, w.summed)
*(w::AveragedWeight, x) = AveragedWeight(_w(w) * _w(x), w.t, w.summed)
*(x, w::AveragedWeight) = AveragedWeight(_w(w) * _w(x), w.t, w.summed)
/(w::AveragedWeight, x) = AveragedWeight(_w(w) / _w(x), w.t, w.summed)
/(x, w::AveragedWeight) = AveragedWeight(_w(w) / _w(x), w.t, w.summed)

Base.zero(T::AveragedWeight) = AveragedWeight(T)
Base.zero(T::Type{<:AveragedWeight}) = AveragedWeight(T)

Base.iterate(w::AveragedWeight, state...) = iterate(w.weight, state...)

Base.show(io::IO, w::AveragedWeight) = print(io, w.weight)

_w(x) = x
_w(w::AveragedWeight) = w.weight

function tick!(w::AveragedWeight, t)
    w.summed += t - w.t * w.weight
    w.t = t
end

function update!(w::AveragedWeight, t, r=1)
    tick!(w, t)
    w.weight += r
end

function average!(w::AveragedWeight, t)
    tick!(w, t)
    w.weight = w.summed / t
end
