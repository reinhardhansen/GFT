# Quadrature preconditioner (Section S6 of the paper): the diagonal
# preconditioner D, the certified quadrature model M_r at every Newton
# step, and the selection rule (M_r when two to eight nodes certify
# kappa <= 2, the diagonal when one node suffices), compared on
#
#   1. the designs of Table 1 (same serialized draws, RNG stream replayed;
#      500 draws at n = 800), and larger designs: one-factor with loadings
#      Un(0.8, 0.995) at n = 200 and 400, Toeplitz at n = 800, Wishart at
#      n = 400 (Table S5);
#   2. sequential inversion with the tangent predictor at n = 100, 400
#      and 800 (Table S6).
#
# Run:  julia -t 1 precond.jl [blocks]     blocks: table  warm  rule  (default: table warm)
#       "rule" reruns only the selection rule on the designs below n = 800
#       (results/precond_rule.csv); at n = 800 the rule selects quadrature.
#       (about 2-3 hours, dominated by n = 800)
# Output: results/precond.csv, results/precond_warm.csv

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using GFT
using LinearAlgebra, Random, Statistics, Printf, Dates

BLAS.set_num_threads(1)
const TOL = 1e-13
const BLOCKS = isempty(ARGS) ? ["table", "warm"] : ARGS
const CONFSEL = "rule" in BLOCKS && !("table" in BLOCKS) ? [3] : [1, 2, 3]

corr_toeplitz(n, rho) = rho .^ abs.((1:n) .- (1:n)')
function corr_wishart(rng, n; dof = 2n)
    X = randn(rng, n, dof); S = X * X'
    Dh = Diagonal(1 ./ sqrt.(diag(S)))
    return Dh * S * Dh
end
function corr_factor(rng, n, lo, hi)
    b = lo .+ (hi - lo) .* rand(rng, n)
    return b * b' + Diagonal(1 .- b .^ 2)
end

const CONFIGS = [("diagonal",   (preconditioner = :diagonal,)),
                 ("quadrature", (preconditioner = :quadrature, kappa = 2.0)),
                 ("rule",       (preconditioner = :auto, kappa = 2.0, rmin = 2, rmax = 8, nmin = 0))]

# warmup
let z = gft(corr_toeplitz(10, 0.5))
    for (_, kw) in CONFIGS; inv_gft(z; kw...); end
    inv_gft_path([z, z]; preconditioner = :quadrature)
end

q(v, p) = isempty(v) ? NaN : quantile(v, p)

function run_design(io, design, zs; reps_time = 1)
    for (label, kw) in CONFIGS[CONFSEL]
        es = Int[]; hv = Int[]; ts = Float64[]; nd = Float64[]; bd = Int[]
        tset = Float64[]; fail = 0; cf = 0
        for z in zs
            best = Inf
            local r, st
            for _ in 1:reps_time
                t0 = time_ns()
                r, _, _, st = GFT._inv_gft(z; tol = TOL, kw...)
                best = min(best, (time_ns() - t0) / 1e9)
            end
            r.converged || (fail += 1; continue)
            push!(es, r.eighs); push!(hv, r.hvs); push!(ts, best)
            push!(bd, st.builds); push!(tset, st.tsetup); cf += st.cholfail
            st.builds > 0 && push!(nd, st.nodes / st.builds)
        end
        @printf(io, "%s,%s,%d,%.1f,%.1f,%.1f,%.1f,%.6f,%.6f,%d,%d\n", design, label,
                length(zs), q(es, .5), q(hv, .5), q(nd, .5), q(bd, .5), q(ts, .5),
                q(tset, .5), fail, cf)
        flush(io)
        @printf("%-24s %-10s E=%5.1f P=%6.1f nodes=%4.1f builds=%4.1f %9.2fms (setup %6.2f) fails=%d cholfail=%d\n",
                design, label, q(es, .5), q(hv, .5), q(nd, .5), q(bd, .5),
                1e3 * q(ts, .5), 1e3 * q(tset, .5), fail, cf); flush(stdout)
    end
end

# ------------------------------------------------------------ 1. table
"table" in BLOCKS && open("results/precond.csv", "w") do io
    println(io, "design,method,draws,eighs_med,hv_med,nodes_med,builds_med,time_med,setup_med,fails,cholfail")
    println("[$(now())] Table 1 designs (replayed draws)"); flush(stdout)
    rng = MersenneTwister(18900217)
    zs_wish = [gft(corr_wishart(rng, 100)) for _ in 1:1000]
    zs_fac  = [gft(corr_factor(rng, 100, 0.85, 0.999)) for _ in 1:1000]
    d = 50 * 49 ÷ 2
    zs2 = [2.0 * randn(rng, d) for _ in 1:1000]
    zs4 = [4.0 * randn(rng, d) for _ in 1:1000]
    for (name, n, rho) in (("toeplitz_rho0.5_n100", 100, 0.5), ("toeplitz_rho0.9_n100", 100, 0.9),
                           ("toeplitz_rho0.99_n100", 100, 0.99), ("toeplitz_rho0.99_n300", 300, 0.99))
        run_design(io, name, [gft(corr_toeplitz(n, rho))]; reps_time = 5)
    end
    run_design(io, "wishart_n100", zs_wish)
    run_design(io, "factor_n100", zs_fac)
    run_design(io, "z_sd2.0_n50", zs2)
    run_design(io, "z_sd4.0_n50", zs4)
    println("[$(now())] larger designs (seed 20260920)"); flush(stdout)
    rng2 = MersenneTwister(20260920)
    run_design(io, "factor08_n200", [gft(corr_factor(rng2, 200, 0.8, 0.995)) for _ in 1:1000])
    run_design(io, "factor08_n400", [gft(corr_factor(rng2, 400, 0.8, 0.995)) for _ in 1:200])
    run_design(io, "wishart_n400", [gft(corr_wishart(rng2, 400)) for _ in 1:100])
    for rho in (0.5, 0.9, 0.99)
        run_design(io, "toeplitz_rho$(rho)_n800", [gft(corr_toeplitz(800, rho))]; reps_time = 3)
    end
    println("[$(now())] n = 800 (replayed draws)"); flush(stdout)
    zs800 = [gft(corr_factor(rng, 800, 0.8, 0.995)) for _ in 1:500]
    run_design(io, "factor_n800", zs800)
end

# ------------------------------------------------------------ 1b. rule only
"rule" in BLOCKS && !("table" in BLOCKS) && open("results/precond_rule.csv", "w") do io
    println(io, "design,method,draws,eighs_med,hv_med,nodes_med,builds_med,time_med,setup_med,fails,cholfail")
    println("[$(now())] selection rule, designs below n = 800"); flush(stdout)
    rng = MersenneTwister(18900217)
    zs_wish = [gft(corr_wishart(rng, 100)) for _ in 1:1000]
    zs_fac  = [gft(corr_factor(rng, 100, 0.85, 0.999)) for _ in 1:1000]
    d = 50 * 49 ÷ 2
    zs2 = [2.0 * randn(rng, d) for _ in 1:1000]
    zs4 = [4.0 * randn(rng, d) for _ in 1:1000]
    for (name, n, rho) in (("toeplitz_rho0.5_n100", 100, 0.5), ("toeplitz_rho0.9_n100", 100, 0.9),
                           ("toeplitz_rho0.99_n100", 100, 0.99), ("toeplitz_rho0.99_n300", 300, 0.99))
        run_design(io, name, [gft(corr_toeplitz(n, rho))]; reps_time = 5)
    end
    run_design(io, "wishart_n100", zs_wish)
    run_design(io, "factor_n100", zs_fac)
    run_design(io, "z_sd2.0_n50", zs2)
    run_design(io, "z_sd4.0_n50", zs4)
    rng2 = MersenneTwister(20260920)
    run_design(io, "factor08_n200", [gft(corr_factor(rng2, 200, 0.8, 0.995)) for _ in 1:1000])
    run_design(io, "factor08_n400", [gft(corr_factor(rng2, 400, 0.8, 0.995)) for _ in 1:200])
    run_design(io, "wishart_n400", [gft(corr_wishart(rng2, 400)) for _ in 1:100])
    for rho in (0.5, 0.9, 0.99)
        run_design(io, "toeplitz_rho$(rho)_n800", [gft(corr_toeplitz(800, rho))]; reps_time = 3)
    end
end

# ------------------------------------------------------------ 2. warm
"warm" in BLOCKS && open("results/precond_warm.csv", "w") do io
    println(io, "n,method,seq,eighs_per_step,hv_per_step,builds_per_step,time_per_step_ms")
    println("[$(now())] sequential inversion with the predictor"); flush(stdout)
    for (n, nseq, T) in ((100, 10, 50), (400, 4, 25), (800, 2, 15))
        rng_w = MersenneTwister(18900217 + n)
        for seq_id in 1:nseq
            z0 = gft(corr_factor(rng_w, n, 0.85, 0.999))
            seq = [copy(z0)]
            for _ in 2:T
                push!(seq, seq[end] + 0.002 * sqrt(4950 / length(z0)) * randn(rng_w, length(z0)))
            end                                      # aggregate step ~0.14 as at n = 100
            for (label, kw) in CONFIGS[1:2]
                t0 = time_ns()
                out, _ = inv_gft_path(seq; predictor = true, rtol = 1e-3, tol = TOL, kw...)
                tt = (time_ns() - t0) / 1e9
                all(r.converged for r in out) || @warn "non-convergence" n label seq_id
                te = sum(r.eighs for r in out); th = sum(r.hvs for r in out)
                @printf(io, "%d,%s,%d,%.3f,%.2f,%.2f,%.4f\n", n, label, seq_id,
                        te / T, th / T, NaN, 1e3 * tt / T)
            end
            flush(io)
        end
        println("  n = $n done"); flush(stdout)
    end
end
println("[$(now())] done. Results in ./results/")
