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

    # ------------------------------------------- revised algorithm (S5)
    @testset "shift identities and normalization" begin
        rng = MersenneTwister(5)
        n = 8
        z = randn(rng, n * 7 ÷ 2); A0 = unvecl(z); x = randn(rng, n)
        F = eigen(Symmetric(A0 + Diagonal(x)))
        ell = GFT.logdiagexp(F.values, F.vectors)
        c = 0.7
        F2 = eigen(Symmetric(A0 + Diagonal(x .+ c)))
        ell2 = GFT.logdiagexp(F2.values, F2.vectors)
        @test maximum(abs, ell2 - ell .- c) < 1e-12     # ell(x + c1) = ell + c1
        # exact normalization: tr exp(A) = n afterwards, and f decreases
        # by n (e^s - 1 - s)
        lam = copy(F.values); xn = copy(x); elln = copy(ell)
        f0 = sum(exp, lam) - sum(x)
        s = GFT._normalize!(xn, lam, elln)
        @test abs(sum(exp, lam) - n) < 1e-10
        @test abs((f0 - (sum(exp, lam) - sum(xn))) - n * (exp(s) - 1 - s)) < 1e-9
        @test maximum(abs, elln - (ell .- s)) < 1e-12
        # J 1 = 1: hess_vec(1) == diag(exp A) (Theorem 2(i))
        L = GFT.loewner_exp(F.values)
        @test maximum(abs, GFT.hess_vec(F.values, F.vectors, L, ones(n)) - exp.(ell)) < 1e-10
    end

    @testset "revised GFT-FP+N: variants agree, one-step recovery along 1" begin
        rng = MersenneTwister(6)
        z = 4.0 * randn(rng, 50 * 49 ÷ 2)
        r = inv_gft(z)                                   # revised default
        rA = inv_gft(z; residual = :gradient, forcing = :sqrt,
                     normalize = false, phase = true, adaptive = false)
        rE = inv_gft(z; exact_hess = true)
        for rr in (r, rA, rE)
            @test rr.converged
            @test maximum(abs, diag(rr.C) .- 1) < 1e-12
        end
        @test maximum(abs, r.x - rA.x) < 1e-9
        @test maximum(abs, r.x - rE.x) < 1e-9
        @test r.eighs <= rA.eighs                        # the point of the revision
        # from x* + c1 the log-residual Newton step returns in one step
        r2 = inv_gft(z; x0 = r.x .+ 3.0, normalize = false)
        @test r2.converged && r2.eighs <= 3
    end

    @testset "3x3 non-descent example handled" begin
        # Hδ = -Dℓ ascends f here (g'δ = 9.46 > 0); the descent test must
        # substitute a fixed-point step and the solve must still converge
        z = [4.5, 11.5, -12.0]                           # A12, A13, A23
        x0 = [-3.9, -16.1, -25.5]
        r = inv_gft(z; x0 = x0, normalize = false)
        @test r.converged
        @test maximum(abs, diag(r.C) .- 1) < 1e-12
    end

    @testset "quadrature preconditioner: certificate and solves" begin
        rng = MersenneTwister(8)
        n = 12
        z = 1.5 * randn(rng, n * 11 ÷ 2); x = randn(rng, n)
        F = eigen(Symmetric(unvecl(z) + Diagonal(x)))
        lam, Q = F.values, F.vectors
        c = maximum(lam); w = lam .- c
        Hs = GFT.hessian(lam, Q, GFT.loewner_exp(w))     # e^{-c} H
        spread = maximum(w) - minimum(w)
        for r in 1:3
            CF, rr = GFT.quadrature_precond(w, Q; rfixed = r)
            @test rr == r && CF !== nothing
            M = Matrix(CF.L) * Matrix(CF.L)'
            ev = eigvals(Symmetric(Hs), Symmetric(M))          # generalized
            @test minimum(ev) > 1 - 1e-8                      # M_r <= H
            @test maximum(ev) <= exp(GFT.logphi(r, spread)) * (1 + 1e-8)
        end
        # thresholds of the paper's Table S4
        @test abs(exp(GFT.logphi(2, 11.6)) - 2) < 0.01
        @test abs(exp(GFT.logphi(3, 22.4)) - 2) < 0.01
        # the solvers agree
        b = 0.8 .+ 0.195 * rand(rng, 100)
        z1 = gft(b * b' + Diagonal(1 .- b .^ 2))
        r0 = inv_gft(z1; tol = 1e-13)
        rq = inv_gft(z1; tol = 1e-13, preconditioner = :quadrature)
        ra = inv_gft(z1; tol = 1e-13, preconditioner = :auto)
        @test r0.converged && rq.converged && ra.converged
        @test maximum(abs, rq.x - r0.x) < 1e-9
        @test maximum(abs, ra.x - r0.x) < 1e-9
        @test rq.hvs < r0.hvs                                   # fewer products
        @test ra.hvs == rq.hvs                                  # rule: two nodes certify -> M_r
        # well-conditioned input: one node certifies, rule keeps the diagonal
        X = randn(rng, 30, 90); S = X * X'; Dh = Diagonal(1 ./ sqrt.(diag(S)))
        z2 = gft(Dh * S * Dh)
        rd = inv_gft(z2; tol = 1e-13); ra2 = inv_gft(z2; tol = 1e-13, preconditioner = :auto)
        @test ra2.hvs == rd.hvs && ra2.eighs == rd.eighs
        _, _, _, st = GFT._inv_gft(z1; tol = 1e-13, preconditioner = :quadrature)
        @test st.builds > 0 && st.cholfail == 0
    end

    @testset "guarded Anderson and tangent predictor" begin
        rng = MersenneTwister(7)
        z = 4.0 * randn(rng, 30 * 29 ÷ 2)
        x = inv_gft(z; tol = 1e-14).x
        a = inv_gft_anderson(z; tol = 1e-14, m = 5)
        g = inv_gft_anderson(z; tol = 1e-14, m = 5, guarded = true)
        @test a.converged && g.converged
        @test maximum(abs, a.x - x) < 1e-9
        @test maximum(abs, g.x - x) < 1e-9
        # predictor: error O(h^2)
        n = 30
        b = 0.85 .+ 0.149 * rand(rng, n)
        z0 = gft(b * b' + Diagonal(1 .- b .^ 2))
        r0, lam, Q = GFT._inv_gft(z0; tol = 1e-14)
        dz = randn(rng, length(z0)); dz ./= norm(dz)
        errs = Float64[]
        for h in (0.05, 0.1, 0.2)
            z1 = z0 .+ h .* dz
            xhat, _ = gft_predict(z0, r0.x, lam, Q, z1; rtol = 1e-8)
            push!(errs, maximum(abs, xhat - inv_gft(z1; tol = 1e-14).x))
        end
        @test errs[2] / errs[1] > 2.5 && errs[3] / errs[2] > 2.5   # ~4x per doubling
        # sequential inversion with and without the predictor
        zs = [z0 .+ 0.02 .* k .* dz for k in 0:5]
        out, nb = inv_gft_path(zs; predictor = true)
        out0, nb0 = inv_gft_path(zs; predictor = false)
        @test all(r.converged for r in out) && all(r.converged for r in out0)
        @test nb == 5 && nb0 == 0
        @test sum(r.eighs for r in out) <= sum(r.eighs for r in out0)
    end

end
println("All tests passed.")
