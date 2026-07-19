# Test suite for GFT.jl
#
# Includes golden-value tests generated from the independently verified
# Python implementation (gft.py, NumPy/SciPy, double precision), so a
# passing run constitutes a cross-language verification of the solver.
#
# Run:  julia runtests.jl

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using GFT
using Test, LinearAlgebra, Random

@testset "GFT.jl" begin

    # ---------------------------------------------------------------- golden
    @testset "golden values from Python implementation" begin
        # Case 1: Toeplitz rho = 0.9, n = 5
        z1 = [1.153290016506162, 0.5558856623939625, 0.35800387580539283,
              0.27329448930908706, 1.0109967961569906, 0.5069766634781976,
              0.358003875805393, 1.0109967961569903, 0.5558856623939642,
              1.1532900165061641]
        x1 = [-1.0295579541972146, -1.511633306658801, -1.5605423055745682,
              -1.5116333066588024, -1.0295579541972182]
        r = inv_gft(z1; tol = 1e-15)
        @test r.converged
        @test maximum(abs, r.x - x1) < 1e-11
        # forward map round-trip
        C = 0.9 .^ abs.((1:5) .- (1:5)')
        @test maximum(abs, gft(C) - z1) < 1e-12
        @test maximum(abs, r.C - C) < 1e-11

        # Case 2: equicorrelation rho = 0.99, n = 4 (repeated eigenvalues)
        z2 = [1.4959840701717646, 1.4959840701718055, 1.4959840701718055,
              1.4959840701717937, 1.4959840701717941, 1.495984070171792]
        x2 = [-3.1091861158162764, -3.1091861158162533, -3.10918611581629,
              -3.109186115816293]
        r = inv_gft(z2; tol = 1e-15)
        @test r.converged
        @test maximum(abs, r.x - x2) < 1e-10

        # Case 3: fixed z, n = 3, with the reconstructed C
        z3 = [0.5, -1.25, 2.0]
        x3 = [-0.4468779844556956, -1.1593398688810534, -1.7913253190785425]
        C3 = [1.0 -0.19502680396870378 -0.6143366326910232;
              -0.19502680396870392 1.000000000000004 0.871779280890441;
              -0.6143366326910233 0.8717792808904407 1.0000000000000056]
        r = inv_gft(z3; tol = 1e-15)
        @test maximum(abs, r.x - x3) < 1e-11
        @test maximum(abs, r.C - C3) < 1e-11
    end

    # ------------------------------------------------------------ round trips
    @testset "round trips, all solvers" begin
        rng = MersenneTwister(1)
        for n in (10, 50)
            X = randn(rng, n, 2n); S = X * X'
            Dh = Diagonal(1 ./ sqrt.(diag(S)))
            C = Symmetric(Dh * S * Dh)
            z = gft(Matrix(C))
            for solver in (inv_gft, inv_gft_fp, inv_gft_broyden,
                           inv_gft_newton)
                r = solver(z)
                @test r.converged
                @test maximum(abs, diag(r.C) .- 1) < 1e-12
                @test maximum(abs, r.C - C) < 1e-10
            end
        end
    end

    # ----------------------------------------------------------- hard cases
    @testset "extreme z (log-domain robustness)" begin
        rng = MersenneTwister(2)
        z = 8.0 * randn(rng, 40 * 39 ÷ 2)      # spectra spanning ~200 log-units
        r = inv_gft(z)
        @test r.converged
        @test maximum(abs, diag(r.C) .- 1) < 1e-12
        rfp = inv_gft_fp(z; maxit = 20000)
        @test rfp.converged
        @test maximum(abs, r.x - rfp.x) < 1e-8
    end

    @testset "near-singular one-factor" begin
        rng = MersenneTwister(3)
        n = 100
        b = 0.995 .+ 0.0049 * rand(rng, n)
        C = b * b' + Diagonal(1 .- b .^ 2)
        z = gft(C)
        r = inv_gft(z)
        @test r.converged && r.eighs < 30
        @test maximum(abs, r.C - C) < 1e-9
    end

    # ------------------------------------------- derivative and Hessian checks
    @testset "gradient, Hessian-vector products, exact Hessian" begin
        rng = MersenneTwister(4)
        n = 12
        A0 = randn(rng, n, n); A0 = (A0 + A0') / 2
        A0[diagind(A0)] .= 0
        x = randn(rng, n)
        A = A0 + Diagonal(x)
        F = eigen(Symmetric(A)); lam, Q = F.values, F.vectors
        f(x) = sum(exp, eigen(Symmetric(A0 + Diagonal(x))).values) - sum(x)
        g = exp.(GFT.logdiagexp(lam, Q)) .- 1
        eps = 1e-6
        for i in 1:n
            e = zeros(n); e[i] = eps
            @test abs((f(x + e) - f(x - e)) / 2eps - g[i]) < 1e-6
        end
        L = GFT.loewner_exp(lam)
        H = GFT.hessian(lam, Q, L)
        v = randn(rng, n)
        @test maximum(abs, GFT.hess_vec(lam, Q, L, v) - H * v) < 1e-10
        # Proposition 1: H*1 = d, lmin(E) I <= H <= D
        E = GFT.expA(lam, Q); d = diag(E)
        @test maximum(abs, H * ones(n) - d) < 1e-8 * maximum(d)
        @test eigmin(Symmetric(Diagonal(d) - H)) > -1e-8 * maximum(d)
        @test eigmin(Symmetric(H - exp(minimum(lam)) * I)) > -1e-10
    end

    @testset "repeated eigenvalues in divided differences" begin
        lam = [1.0, 1.0, 2.0, 2.0 + 1e-15]
        L = GFT.loewner_exp(lam)
        @test L[1, 2] ≈ exp(1.0)
        @test L[3, 4] ≈ exp(2.0) rtol = 1e-12
        @test issymmetric(round.(L, digits = 10))
    end

end
println("All tests passed.")
