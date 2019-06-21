using PeriodicSchur
using Test
using LinearAlgebra
using Random

Random.seed!(0)

@testset "hesstriu" begin
    n = 5
    K = 10
    Ms = ntuple(i->rand(n, n), K)
    Π = ProductOfMatrices(Ms)

    # test prod function does the right thing
    @test prod(Π) ≈ *(reverse(Ms)...)

    # store initial norm of the product
    Ms_norm = norm(prod(Π), 2)

    # reduce to hessemberg-triangular
    hesstriu!(Π)

    # check the induced matrix norm of the lower triangular parts is small 
    @test norm(tril(Π[1], -2)) < n*eps(1.0)
    for k = 2:K
        @test norm(tril(Π[k], -1)) < n*eps(1.0)
    end

    # we have applied similarity transformations
    @test norm(prod(Π), 2) ≈ Ms_norm
end




# # For the sequence of matrices M1, M2, M3, ... MK
# # evaluate the product Mk⋅...⋅M2⋅M1 
# function matprod(\::AbstractVector{<:Matrix})
#     K = length(\)
#     K == 1 && return \[1]
#     out =    copy(\[1])
#     tmp = similar(\[1])
#     for k in 2:K
#         mul!(tmp, \[i], out)
#         out .= tmp
#     end
#     return out
# end

# @testset "" begin
#     for n in (2, 10, 20, 100, 200)
#         for K in (1, 10, 100)
#             \ = [rand(n, n) for _ in K]
            
