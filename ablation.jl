# Supplementary experiments behind Section S5 of the paper, on the SAME
# serialized draws as overnight.jl (the RNG stream is replayed):
#
#   1. Ablation of the Newton step (Table S1): residual (gradient or log),
#      forcing exponent, normalization, initial fixed-point phase, and
#      adaptive initial step, varied one at a time under otherwise
#      identical rules, on the z-designs, the structured designs, and
#      the first 50 draws of the n = 800 design.
#   2. Anderson acceleration with memory 5 to 50, plain and guarded
#      (Table S2), on the z-designs.
#   3. Sequential inversion with and without the tangent predictor at
#      three step sizes (Table S3).
#
# Run:  julia -t 1 ablation.jl [blocks]   (about 40 minutes)
#       blocks: any of  ablation  anderson  warm  (default: all three)
# Output: results/ablation.csv, results/anderson_sweep.csv,
#         results/warm_pred.csv

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using GFT
using LinearAlgebra, Random, Statistics, Printf, Dates

BLAS.set_num_threads(1)
const TOL = 1e-13
const BLOCKS = isempty(ARGS) ? ["ablation", "anderson", "warm"] : ARGS

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

# --- replay the overnight RNG stream (same draws as Table 1)
println("[$(now())] generating inputs"); flush(stdout)
rng = MersenneTwister(18900217)
zs_wish = [gft(corr_wishart(rng, 100)) for _ in 1:1000]
zs_fac  = [gft(corr_factor(rng, 100, 0.85, 0.999)) for _ in 1:1000]
d = 50 * 49 ÷ 2
zs2 = [2.0 * randn(rng, d) for _ in 1:1000]
zs4 = [4.0 * randn(rng, d) for _ in 1:1000]
zs800 = [gft(corr_factor(rng, 800, 0.8, 0.995)) for _ in 1:50]   # first 50 of 500
zs_toep = [("toeplitz_rho0.5_n100", gft(corr_toeplitz(100, 0.5))),
           ("toeplitz_rho0.9_n100", gft(corr_toeplitz(100, 0.9))),
           ("toeplitz_rho0.99_n100", gft(corr_toeplitz(100, 0.99))),
           ("toeplitz_rho0.99_n300", gft(corr_toeplitz(300, 0.99)))]

# --- the variants (labels as in the paper's Table S1)
V(; residual, forcing, normalize, phase, adaptive) =
    (residual = residual, forcing = forcing, normalize = normalize,
     phase = phase, adaptive = adaptive)
const VARIANTS = [
    ("A",     V(residual = :gradient, forcing = :sqrt, normalize = false, phase = true,  adaptive = false)),
    ("B",     V(residual = :gradient, forcing = :quad, normalize = false, phase = true,  adaptive = false)),
    ("C",     V(residual = :log,      forcing = :sqrt, normalize = false, phase = true,  adaptive = false)),
    ("D",     V(residual = :log,      forcing = :quad, normalize = false, phase = true,  adaptive = false)),
    ("AN",    V(residual = :gradient, forcing = :sqrt, normalize = true,  phase = true,  adaptive = false)),
    ("BN",    V(residual = :gradient, forcing = :quad, normalize = true,  phase = true,  adaptive = false)),
    ("CN",    V(residual = :log,      forcing = :sqrt, normalize = true,  phase = true,  adaptive = false)),
    ("DN",    V(residual = :log,      forcing = :quad, normalize = true,  phase = true,  adaptive = false)),
    ("AN0",   V(residual = :gradient, forcing = :sqrt, normalize = true,  phase = false, adaptive = false)),
    ("BN0",   V(residual = :gradient, forcing = :quad, normalize = true,  phase = false, adaptive = false)),
    ("CN0",   V(residual = :log,      forcing = :sqrt, normalize = true,  phase = false, adaptive = false)),
    ("DN0",   V(residual = :log,      forcing = :quad, normalize = true,  phase = false, adaptive = false)),
    ("CN0t",  V(residual = :log,      forcing = :sqrt, normalize = true,  phase = false, adaptive = true)),
    ("DN0t",  V(residual = :log,      forcing = :quad, normalize = true,  phase = false, adaptive = true)),
]
const KEY = ["A", "B", "C", "D", "DN", "DN0", "DN0t"]   # structured designs

# warmup
let z = gft(corr_toeplitz(10, 0.5))
    for (_, kw) in VARIANTS; inv_gft(z; kw...); end
    inv_gft_anderson(z); inv_gft_anderson(z; guarded = true)
    inv_gft_path([z, z]); inv_gft_path([z, z]; predictor = false)
end

q(v, p) = isempty(v) ? NaN : quantile(v, p)

function run_variant(io, design, label, kw, zs; reps_time = 1)
    es = Int[]; hv = Int[]; ts = Float64[]; fail = 0
    for z in zs
        best = Inf
        local r
        for _ in 1:reps_time
            t0 = time_ns()
            r = inv_gft(z; tol = TOL, kw...)
            best = min(best, (time_ns() - t0) / 1e9)
        end
        r.converged || (fail += 1; continue)
        push!(es, r.eighs); push!(hv, r.hvs); push!(ts, best)
    end
    @printf(io, "%s,%s,%d,%.1f,%.1f,%d,%.1f,%.6f,%d\n", design, label, length(zs),
            q(es, .5), q(es, .9), isempty(es) ? -1 : maximum(es), q(hv, .5), q(ts, .5), fail)
    flush(io)
    @printf("%-24s %-6s E=%5.1f [q90 %5.1f, max %3d] P=%5.1f  %7.2fms fails=%d\n",
            design, label, q(es, .5), q(es, .9), isempty(es) ? -1 : maximum(es),
            q(hv, .5), 1e3 * q(ts, .5), fail); flush(stdout)
end

# ------------------------------------------------------------ 1. ablation
"ablation" in BLOCKS && open("results/ablation.csv", "w") do io
    println("[$(now())] ablation"); flush(stdout)
    println(io, "design,variant,draws,eighs_med,eighs_q90,eighs_max,hv_med,time_med,fails")
    for (label, kw) in VARIANTS
        run_variant(io, "z_sd2.0_n50", label, kw, zs2)
        run_variant(io, "z_sd4.0_n50", label, kw, zs4)
    end
    for (label, kw) in VARIANTS
        label in KEY || continue
        for (name, z) in zs_toep
            run_variant(io, name, label, kw, [z]; reps_time = 5)
        end
        run_variant(io, "wishart_n100", label, kw, zs_wish)
        run_variant(io, "factor_n100", label, kw, zs_fac)
    end
    for (label, kw) in VARIANTS
        label in ("A", "B", "D", "DN0", "DN0t") || continue
        run_variant(io, "factor_n800", label, kw, zs800)
    end
end

# ------------------------------------------------------ 2. Anderson sweep
"anderson" in BLOCKS && open("results/anderson_sweep.csv", "w") do io
    println("[$(now())] Anderson memory sweep"); flush(stdout)
    println(io, "design,method,draws,eighs_med,eighs_q90,eighs_max,time_med,fails")
    for (design, zs) in (("z_sd2.0_n50", zs2), ("z_sd4.0_n50", zs4))
        for m in (5, 10, 20, 50), guarded in (false, true)
            es = Int[]; ts = Float64[]; fail = 0
            for z in zs
                t0 = time_ns()
                r = inv_gft_anderson(z; tol = TOL, m = m, guarded = guarded)
                t = (time_ns() - t0) / 1e9
                r.converged || (fail += 1; continue)
                push!(es, r.eighs); push!(ts, t)
            end
            label = "anderson$(m)" * (guarded ? "g" : "")
            @printf(io, "%s,%s,%d,%.1f,%.1f,%d,%.6f,%d\n", design, label, length(zs),
                    q(es, .5), q(es, .9), isempty(es) ? -1 : maximum(es), q(ts, .5), fail)
            flush(io)
            @printf("%-12s %-12s E=%6.1f [q90 %6.1f, max %4d] %7.2fms fails=%d\n",
                    design, label, q(es, .5), q(es, .9), isempty(es) ? -1 : maximum(es),
                    1e3 * q(ts, .5), fail); flush(stdout)
        end
    end
end

# ------------------------------------------ 3. warm starts and predictor
const VA = VARIANTS[1][2]                       # current algorithm (A)
"warm" in BLOCKS && open("results/warm_pred.csv", "w") do io
    println("[$(now())] sequential inversion with the tangent predictor"); flush(stdout)
    println(io, "sd,method,seq,eighs_per_step,eighs_warm_only,hv_per_step,bdz_per_step,time_per_step_ms")
    for sd in (0.0005, 0.002, 0.008)
        rng_w = MersenneTwister(18900217 + round(Int, 1e6 * sd))
        for seq_id in 1:10
            n, T = 100, 50
            z0 = gft(corr_factor(rng_w, n, 0.85, 0.999))
            seq = [copy(z0)]
            for _ in 2:T
                push!(seq, seq[end] + sd * randn(rng_w, length(z0)))
            end
            for (method, kw, pred, rtol) in (("current_warm", VA, false, 1e-3),
                                             ("current_pred1e-8", VA, true, 1e-8),
                                             ("current_pred1e-3", VA, true, 1e-3),
                                             ("revised_warm", NamedTuple(), false, 1e-3),
                                             ("revised_pred1e-8", NamedTuple(), true, 1e-8),
                                             ("revised_pred1e-3", NamedTuple(), true, 1e-3))
                t0 = time_ns()
                out, nb = inv_gft_path(seq; predictor = pred, rtol = rtol, tol = TOL, kw...)
                tt = (time_ns() - t0) / 1e9
                all(r.converged for r in out) || @warn "non-convergence" sd method seq_id
                te = sum(r.eighs for r in out); tw = sum(r.eighs for r in out[2:end])
                th = sum(r.hvs for r in out)
                @printf(io, "%g,%s,%d,%.3f,%.3f,%.2f,%.2f,%.4f\n", sd, method, seq_id,
                        te / T, tw / (T - 1), th / T, nb / T, 1e3 * tt / T)
            end
            flush(io)
        end
        println("  sd = $sd done")
    end
end
println("[$(now())] done. Results in ./results/")
