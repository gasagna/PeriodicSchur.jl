module PeriodicSchur

using Base.Iterators
using LinearAlgebra

export ProductOfMatrices,
       hesstriu!

"""
The product of matrices M1, M2, ..., MK as `MK*....*M2*M1`
"""
struct ProductOfMatrices{T, M<:DenseMatrix{T}, N, V<:NTuple{N, M}}
    Ms::V
end

for fun in (:(Base.length), :(Base.getindex))
    @eval ($fun)(Π::ProductOfMatrices, args...) = ($fun(Π.Ms, args...))
end

"""
    prod(Π::ProductOfMatrices)

Evaluate the product `MK ⋅ ... ⋅ M2 ⋅ M1`.
"""
Base.prod(Π::ProductOfMatrices) = *(reverse(Π.Ms)...)

"""
Transform a sequence of matrices into an Hessemberg-Triangular form
"""
function hesstriu!(Π::ProductOfMatrices{T, M}) where {T, M<:DenseMatrix{T}}
    # number of matrices
    K = length(Π)

    # check all matrices have equal size and are square
    N = size(Π[1], 1)
    for k in 1:K
        size(Π[k]) == (N, N) ||
            throw(ArgumentError("all matrices must be square and with equal size"))
    end

    # preallocate reflector and work arrays
    v    = Vector{T}(undef, N)
    work = Vector{T}(undef, N)

    # transform the sequence of matrices M1, M2, ..., MK
    # to a Hessenberg-Triangular form, where M1 is reduces to a
    # Hessemberg form and all other matrices to upper triangular form
    for n in 1:N-1
        # e.g. K = 4: k in 2, 3, 4, 1
        for k in take(drop(cycle(1:K), 1), K)
            # construct reflector
            istart = k == 1 ? n+1 : n
            v .= 0
            @inbounds @simd for i ∈ istart:N
                v[i] = Π[k][i, n]
            end
            τ = LinearAlgebra.LAPACK.larfg!(view(v, istart:N))

            # apply reflector to the left of of Π[k]
            LinearAlgebra.LAPACK.larf!('L', v, τ, Π[k])

            # apply reflector to the right of Π[k+1]
            LinearAlgebra.LAPACK.larf!('R', v, τ, Π[mod1(k+1, K)])
        end
    end
    return Π
end

end