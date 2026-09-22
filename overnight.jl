# Full benchmark protocol for the paper: writes CSV files that the table
# and figure are built from. Expected runtime: roughly 4-6 hours
# single-threaded (dominated by the 500 draws at n=800).
#
# Run:  julia -t 1 overnight.jl
#
# Outputs (in ./results/):
#   table.csv     - one row per (design, method): median/quartile eighs,
#                   hv, time, failures, and 1-lmin(H*) per design
#   rate.csv      - RATE_PTS random designs: lmin(H*), fixed-point iterations
#   hist.csv      - convergence histories for the two figure panels
#   warm.csv      - warm-start experiment, 10 sequences x 50 steps
#   tol.csv       - sigma=4 design at tolerances 1e-6 and 1e-13
#   log.txt       - versioninfo, BLAS config, timestamps

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using GFT
using LinearAlgebra, Random, Statistics, Printf, Dates

BLAS.set_num_threads(1)
mkpath("results")

const TOL = 1e-13
const SEED = 18900217
const REPS = 1000         # draws per random design
const REPS_BIG = 500      # draws at n = 800
const RATE_PTS = 1000     # designs for the rate-validation figure

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

function lmin_Hstar(z, x)
    F = eigen(Symmetric(unvecl(z) + Diagonal(x)))
    return eigmin(Symmetric(GFT.hessian(F.values, F.vectors)))
end

const METHODS = Dict(
    "fp"      => z -> inv_gft_fp(z; tol = TOL, maxit = 5000),
    "broyden" => z -> inv_gft_broyden(z; tol = TOL),
    "newton"  => z -> inv_gft_newton(z; tol = TOL, warm = 1, safeguard = false),
    "newtonsafe" => z -> inv_gft(z; tol = TOL, exact_hess = true),
    "fpn"     => z -> inv_gft(z; tol = TOL),
)

q(v, p) = isempty(v) ? NaN : quantile(v, p)

function run_case(io, tag, zs, methods; reps_time = 1)
    println("[$(now())] $tag ($(length(zs)) draws)"); flush(stdout)
    n = round(Int, (1 + sqrt(1 + 8 * length(zs[1]))) / 2)
    lm = n <= 300 ?
         median([lmin_Hstar(z, inv_gft(z; tol = TOL).x) for z in zs]) : NaN
    for m in methods
        ts = Float64[]; es = Int[]; hv = Int[]; fail = 0
        for z in zs
            best = Inf
            local r
            for _ in 1:reps_time
                t0 = time_ns()
                r = METHODS[m](z)
                best = min(best, (time_ns() - t0) / 1e9)
            end
            r.converged || (fail += 1; continue)
            push!(ts, best); push!(es, r.eighs); push!(hv, r.hvs)
        end
        @printf(io, "%s,%s,%d,%.1f,%.1f,%.1f,%.1f,%.6f,%.6f,%.6f,%d,%d,%.6f\n",
                tag, m, length(zs), q(es, .5), q(es, .25), q(es, .75),
                q(hv, .5), q(ts, .5), q(ts, .25), q(ts, .75),
                fail, length(zs), 1 - lm)
        flush(io)
    end
end

function main()
    open("results/log.txt", "w") do lg
        println(lg, "started: ", now())
        println(lg, "julia: ", VERSION)
        println(lg, "blas: ", BLAS.get_config())
        println(lg, "threads: ", BLAS.get_num_threads())
        println(lg, "REPS=$REPS REPS_BIG=$REPS_BIG RATE_PTS=$RATE_PTS SEED=$SEED")
    end
    # JIT warmup
    let z = gft(corr_toeplitz(10, 0.5))
        for m in values(METHODS); m(z); end
        inv_gft_path([z, z]); inv_gft_fp(z; x0 = zeros(10)); inv_gft_broyden(z; x0 = zeros(10), warm = 0)
    end

    rng = MersenneTwister(SEED)
    io = open("results/table.csv", "w")
    println(io, "design,method,draws,eighs_med,eighs_q25,eighs_q75,hv_med," *
                "time_med,time_q25,time_q75,fails,total,one_minus_lmin")

    for n in (100, 300), rho in (0.5, 0.9, 0.99)
        meths = n <= 100 ? ["fp", "broyden", "newton", "newtonsafe", "fpn"] :
                           ["fp", "broyden", "fpn"]
        run_case(io, "toeplitz_rho$(rho)_n$(n)",
                 [gft(corr_toeplitz(n, rho))], meths; reps_time = 5)
    end
    run_case(io, "wishart_n100",
             [gft(corr_wishart(rng, 100)) for _ in 1:REPS],
             ["fp", "broyden", "newton", "newtonsafe", "fpn"])
    run_case(io, "factor_n100",
             [gft(corr_factor(rng, 100, 0.85, 0.999)) for _ in 1:REPS],
             ["fp", "broyden", "newton", "newtonsafe", "fpn"])
    d = 50 * 49 ÷ 2
    zs_sd4 = Vector{Vector{Float64}}()          # kept for the tolerance block
    for s in (2.0, 4.0)
        zs = [s * randn(rng, d) for _ in 1:REPS]
        s == 4.0 && (zs_sd4 = zs)
        run_case(io, "z_sd$(s)_n50", zs,
                 ["fp", "broyden", "newton", "newtonsafe", "fpn"])
    end
    run_case(io, "factor_n800",
             [gft(corr_factor(rng, 800, 0.8, 0.995)) for _ in 1:REPS_BIG],
             ["fp", "fpn"])
    close(io)

    # ---- rate-validation points (Figure, panel c)
    println("[$(now())] rate validation ($RATE_PTS designs)")
    open("results/rate.csv", "w") do f
        println(f, "lmin,iters")
        n = 50; dd = n * 49 ÷ 2
        done = 0
        while done < RATE_PTS
            k = done % 3
            z = k == 0 ? gft(corr_wishart(rng, n)) :
                k == 1 ? gft(corr_factor(rng, n, 0.3 + 0.65 * rand(rng), 0.999)) :
                         (0.5 + 2.0 * rand(rng)) * randn(rng, dd)
            r = inv_gft_fp(z; tol = TOL, maxit = 20000)
            r.converged || continue
            println(f, lmin_Hstar(z, r.x), ",", r.iters)
            done += 1
        end
    end

    # ---- convergence histories (Figure, panels a and b)
    println("[$(now())] histories")
    open("results/hist.csv", "w") do f
        println(f, "panel,method,iter,err")
        panels = [("a", gft(corr_factor(rng, 100, 0.85, 0.999))),
                  ("b", 4.0 * randn(rng, d))]
        for (p, z) in panels, (m, sv) in
                (("fp", z -> inv_gft_fp(z; maxit = 5000)),
                 ("broyden", z -> inv_gft_broyden(z)),
                 ("fpn", z -> inv_gft(z)))
            r = sv(z)
            for (i, e) in enumerate(r.hist)
                println(f, p, ",", m, ",", i, ",", min(e, 1e6))
            end
            println(f, p, ",", m, "_converged,0,", r.converged ? 1 : 0)
        end
    end

    # ---- warm starts: 10 sequences x 50 steps
    println("[$(now())] warm starts")
    open("results/warm.csv", "w") do f
        println(f, "seq,method,eighs_per_step,time_per_step_ms")
        for seq_id in 1:10
            n, T, step = 100, 50, 0.002
            z0 = gft(corr_factor(rng, n, 0.85, 0.999))
            seq = [copy(z0)]
            for _ in 2:T
                push!(seq, seq[end] + step * randn(rng, length(z0)))
            end
            for m in ("fp", "broyden", "fpn", "fpn_pred")
                x0 = nothing; te = 0; tt = 0.0
                if m == "fpn_pred"                 # warm start + tangent predictor
                    t0 = time_ns()
                    out, _ = inv_gft_path(seq; predictor = true, rtol = 1e-3, tol = TOL)
                    tt = (time_ns() - t0) / 1e9
                    te = sum(r.eighs for r in out)
                else
                    for zt in seq
                        t0 = time_ns()
                        r = m == "fp" ?
                                inv_gft_fp(zt; x0 = x0, tol = TOL, maxit = 5000) :
                            m == "broyden" ?
                                inv_gft_broyden(zt; x0 = x0, tol = TOL,
                                                warm = x0 === nothing ? 1 : 0) :
                                inv_gft(zt; x0 = x0, tol = TOL)
                        tt += (time_ns() - t0) / 1e9
                        te += r.eighs; x0 = r.x
                    end
                end
                println(f, seq_id, ",", m, ",", te / T, ",", 1e3 * tt / T)
            end
        end
    end

    # ---- tolerance sensitivity, sigma = 4 design, on the SAME inputs
    # as the z_sd4 rows of table.csv (no fresh draws)
    println("[$(now())] tolerance sensitivity")
    open("results/tol.csv", "w") do f
        println(f, "tol,method,eighs_med,fails,total")
        zs = zs_sd4
        for tol in (1e-6, 1e-13)
            for (m, sv) in (("fp", z -> inv_gft_fp(z; tol = tol, maxit = 5000)),
                            ("broyden", z -> inv_gft_broyden(z; tol = tol)),
                            ("newton", z -> inv_gft_newton(z; tol = tol, warm = 1, safeguard = false)),
                            ("fpn", z -> inv_gft(z; tol = tol)))
                es = Int[]; fail = 0
                for z in zs
                    r = sv(z)
                    r.converged ? push!(es, r.eighs) : (fail += 1)
                end
                println(f, tol, ",", m, ",", q(es, .5), ",", fail, ",", length(zs))
            end
        end
    end

    open("results/log.txt", "a") do lg
        println(lg, "finished: ", now())
    end
    println("[$(now())] done. Results in ./results/")
end

main()
