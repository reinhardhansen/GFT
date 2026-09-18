# Regression sweep for the inv_gft_newton rounding-floor stall fixed in
# 1.0.1.  Before the fix this reports a few percent of failures at n = 10
# with eighs in the thousands; after the fix it should report 0 failures
# and eighs in the single digits.
#
# Run:  julia stress_newton.jl

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using GFT, LinearAlgebra, Random, Printf

const N = 200

@printf("inv_gft_newton stress sweep, %d draws per n\n\n", N)

for n in (10, 50)
    bad = 0
    mx = 0
    worst = 0.0
    rng = MersenneTwister(20260822)
    t0 = time()
    for _ in 1:N
        X = randn(rng, n, 2n)
        S = X * X'
        Dh = 1 ./ sqrt.(diag(S))
        C = S .* (Dh * Dh')
        C = (C + C') / 2
        z = gft(C)
        r = inv_gft_newton(z)
        if !r.converged
            bad += 1
            worst = max(worst, r.err)
        end
        mx = max(mx, r.eighs)
    end
    @printf("n = %2d   not converged %3d/%d   max eighs %6d   %s%.1fs\n",
            n, bad, N, mx,
            bad > 0 ? @sprintf("worst stalled err %.2e   ", worst) : "",
            time() - t0)
end

println("\nall four solvers on one draw per n:")
for n in (10, 50)
    rng = MersenneTwister(1)
    X = randn(rng, n, 2n)
    S = X * X'
    Dh = 1 ./ sqrt.(diag(S))
    C = S .* (Dh * Dh')
    C = (C + C') / 2
    z = gft(C)
    for (nm, slv) in (("inv_gft", inv_gft), ("inv_gft_fp", inv_gft_fp),
                      ("inv_gft_broyden", inv_gft_broyden),
                      ("inv_gft_newton", inv_gft_newton))
        r = slv(z)
        @printf("  n = %2d  %-16s converged %-5s err %.2e  eighs %5d  max|C-C0| %.2e\n",
                n, nm, r.converged, r.err, r.eighs, maximum(abs, r.C - C))
    end
end
