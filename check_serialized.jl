# Cross-language agreement on serialized random inputs (supplement S4).
# Reads the z vectors and NumPy iteration counts in ./serialized/ and
# reruns all four methods in Julia on the identical inputs.
#
# Run:  julia -t 1 check_serialized.jl        (about 2-4 minutes)

import Pkg; Pkg.activate(@__DIR__; io = devnull)
using GFT
using LinearAlgebra, Statistics, Printf

BLAS.set_num_threads(1)
const TOL = 1e-13

readz(path) = [parse.(Float64, split(l, ',')) for l in readlines(path)]

meths = Dict(
    "fp"      => z -> inv_gft_fp(z; tol = TOL, maxit = 5000),
    "broyden" => z -> inv_gft_broyden(z; tol = TOL),
    "newton"  => z -> inv_gft_newton(z; tol = TOL, warm = 1, safeguard = false),
    "fpn"     => z -> inv_gft(z; tol = TOL),
)

# python counts: (design, draw, method) -> (eighs, converged)
py = Dict{Tuple{String,Int,String},Tuple{Int,Int}}()
for l in readlines(joinpath(@__DIR__, "serialized", "py_counts.csv"))[2:end]
    d, j, m, e, c = split(l, ',')
    py[(String(d), parse(Int, j), String(m))] = (parse(Int, e), parse(Int, c))
end
length(py) == 300 || error("py_counts.csv incomplete ($(length(py)) of 300 " *
                           "entries): wait for the file to sync and rerun")

# warmup
let z = gft(0.9 .^ abs.((1:10) .- (1:10)'))
    for f in values(meths); f(z); end
end

println("method    exact-match  |count diff|>0 (max)   conv-status mismatches")
for m in ("fp", "broyden", "newton", "fpn")
    nmatch = 0; ndiff = 0; maxdiff = 0; nconv = 0; tot = 0
    for d in ("wishart_n100", "factor_n100", "z_sd4_n50")
        zs = readz(joinpath(@__DIR__, "serialized", "zs_$d.csv"))
        for (j, z) in enumerate(zs)
            r = meths[m](z)
            pe, pc = py[(d, j, m)]
            tot += 1
            if (r.converged ? 1 : 0) != pc
                nconv += 1
            elseif r.eighs == pe
                nmatch += 1
            else
                ndiff += 1
                maxdiff = max(maxdiff, abs(r.eighs - pe))
            end
        end
    end
    @printf("%-8s  %3d/%d        %d (max %d)              %d\n",
            m, nmatch, tot, ndiff, maxdiff, nconv)
end
