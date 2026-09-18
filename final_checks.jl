# Two final checks requested in review, on the SAME serialized draws as
# the overnight run (the RNG stream is replayed so the z-design inputs
# are identical to those behind Table 1):
#
#   1. Tolerance sensitivity on the z ~ N(0,16I) design at 1e-6 and
#      1e-13, same 1000 inputs at both tolerances.
#   2. Broyden with equal globalization: the published one-step
#      initialization versus the same log-domain fixed-point phase as
#      GFT-FP+N (fixed-point steps until max ell <= log 2), on the
#      same 1000 draws of each extreme design.
#
# Run:  julia -t 1 final_checks.jl        (about 10-15 minutes)
# Output: results/final.csv and console summary.

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using GFT
using LinearAlgebra, Random, Statistics, Printf

BLAS.set_num_threads(1)
const TOL = 1e-13

# --- replay the overnight RNG stream to recover the table's z draws
rng = MersenneTwister(18900217)
for _ in 1:1000; randn(rng, 100, 200); end       # wishart_n100
for _ in 1:1000; rand(rng, 100); end             # factor_n100
d = 50 * 49 ÷ 2
zs2 = [2.0 * randn(rng, d) for _ in 1:1000]      # z_sd2 draws (Table 1)
zs4 = [4.0 * randn(rng, d) for _ in 1:1000]      # z_sd4 draws (Table 1)

# warmup
let C = 0.9 .^ abs.((1:10) .- (1:10)')
    z = gft(C)
    inv_gft(z); inv_gft_broyden(z); inv_gft_broyden(z; globalized = true)
    inv_gft_fp(z); inv_gft_newton(z)
    inv_gft_anderson(z); inv_gft_lbfgs(z); inv_gft_lbfgs(z; globalized = true)
end

# run solver on every input, timing each solve (wall seconds)
function timed(sv, zs)
    out = Vector{Any}(undef, length(zs)); ts = zeros(length(zs))
    for (i, z) in enumerate(zs)
        t0 = time_ns(); out[i] = sv(z); ts[i] = (time_ns() - t0) / 1e9
    end
    return out, ts
end

function summarize(tag, results, ts)
    ok = [r.converged for r in results]
    es = [r.eighs for r in results[ok]]
    fails = count(!, ok)
    med = isempty(es) ? NaN : median(es)
    tmed = isempty(es) ? NaN : median(ts[ok])
    @printf("%-28s median eighs=%6.1f  median time=%7.1fms fails=%d/%d\n",
            tag, med, 1e3 * tmed, fails, length(results))
    return (med, fails, length(results), tmed)
end

open("results/final.csv", "w") do io
    println(io, "experiment,method,eighs_med,fails,total,time_med")

    println("== tolerance sensitivity, z~N(0,16I), same 1000 inputs ==")
    for tol in (1e-6, 1e-13)
        for (m, sv) in (("fp", z -> inv_gft_fp(z; tol = tol, maxit = 5000)),
                        ("broyden", z -> inv_gft_broyden(z; tol = tol)),
                        ("newton", z -> inv_gft_newton(z; tol = tol, warm = 1, safeguard = false)),
                        ("fpn", z -> inv_gft(z; tol = tol)))
            res, ts = timed(sv, zs4)
            med, fails, tot, tmed = summarize("tol=$tol $m", res, ts)
            println(io, "tol_$tol,$m,$med,$fails,$tot,$tmed")
        end
    end

    println("== Broyden, same initial fixed-point phase, same 1000 inputs ==")
    for (tag, zs) in (("sd2", zs2), ("sd4", zs4))
        for (m, sv) in (("published", z -> inv_gft_broyden(z; tol = TOL)),
                        ("globalized", z -> inv_gft_broyden(z; tol = TOL,
                                                            globalized = true)),
                        ("fpn", z -> inv_gft(z; tol = TOL)))
            res, ts = timed(sv, zs)
            med, fails, tot, tmed = summarize("$tag broyden-$m", res, ts)
            println(io, "broyden_$tag,$m,$med,$fails,$tot,$tmed")
        end
    end

    # --- full Newton given ALL of GFT-FP+N's safeguards: identical
    # algorithm, but the Newton system is solved with the explicit
    # O(n^4) Hessian instead of matrix-free CG. Isolates Hessian
    # formation as the only design difference.
    println("== full Newton with GFT-FP+N's safeguards, same 1000 inputs ==")
    let z = gft(0.9 .^ abs.((1:10) .- (1:10)'))   # sanity: agree with CG path
        r1 = inv_gft(z; tol = 1e-14)
        r2 = inv_gft(z; tol = 1e-14, exact_hess = true)
        @assert maximum(abs, r1.x - r2.x) < 1e-10
    end
    for (tag, zs) in (("sd2", zs2), ("sd4", zs4))
        res, ts = timed(z -> inv_gft(z; tol = TOL, exact_hess = true), zs)
        med, fails, tot, tmed = summarize("$tag newton-safeguarded", res, ts)
        println(io, "newtonsafe_$tag,exact_hess,$med,$fails,$tot,$tmed")
    end
    # --- standard Jacobian-free tools as comparators: Anderson
    # acceleration of the fixed point (memory 5) and L-BFGS on f
    # (memory 10), the latter from x = 0 and after GFT-FP+N's initial
    # fixed-point phase. Same 1000 inputs per design.
    println("== Anderson(5) and L-BFGS(10), same 1000 inputs ==")
    let z = gft(0.9 .^ abs.((1:10) .- (1:10)'))   # sanity vs GFT-FP+N
        x = inv_gft(z; tol = 1e-14).x
        @assert maximum(abs, inv_gft_anderson(z; tol = 1e-14).x - x) < 1e-9
        @assert maximum(abs, inv_gft_lbfgs(z; tol = 1e-14).x - x) < 1e-9
    end
    for (tag, zs) in (("sd2", zs2), ("sd4", zs4))
        for (m, sv) in (("anderson5", z -> inv_gft_anderson(z; tol = TOL, m = 5)),
                        ("lbfgs10", z -> inv_gft_lbfgs(z; tol = TOL, m = 10)),
                        ("lbfgs10glob", z -> inv_gft_lbfgs(z; tol = TOL, m = 10,
                                                            globalized = true)))
            res, ts = timed(sv, zs)
            med, fails, tot, tmed = summarize("$tag $m", res, ts)
            println(io, "tools_$tag,$m,$med,$fails,$tot,$tmed")
        end
    end
end
println("done: results/final.csv")
