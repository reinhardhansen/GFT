"""
    GFT

Generalized Fisher transformation of correlation matrices and its inverse.

The inverse is computed from the variational characterization

    x*(z) = argmin_x  tr(exp(A[x;z])) - sum(x),

where `A[x;z]` is symmetric with off-diagonal elements `z` and diagonal `x`
(Archakov and Hansen, 2021; this paper). Solvers:

  * `inv_gft`         -- GFT-FP+N: fixed-point phase in the log domain,
                         then matrix-free inexact Newton via preconditioned
                         conjugate gradients (recommended)
  * `inv_gft_fp`      -- Archakov-Hansen fixed point (log domain)
  * `inv_gft_broyden` -- Broyden's method as in Chen, Fei and Yu (2025)
  * `inv_gft_newton`  -- full Newton with exact O(n^4) Hessian

All solvers count eigendecompositions (`eighs`), the dominant O(n^3)
kernel, and Hessian-vector products (`hvs`) where applicable.

Only standard libraries are used (LinearAlgebra). Requires Julia >= 1.6.
"""
module GFT

using LinearAlgebra

export gft, inv_gft, inv_gft_fp, inv_gft_broyden, inv_gft_newton,
       inv_gft_anderson, inv_gft_lbfgs, inv_gft_path, gft_predict,
       vecl, unvecl, InvResult

# ------------------------------------------------------------------ types

struct InvResult
    x::Vector{Float64}
    C::Matrix{Float64}
    iters::Int
    eighs::Int
    hvs::Int
    err::Float64
    converged::Bool
    hist::Vector{Float64}
end

# ------------------------------------------------------------------ utilities

"Stack the below-diagonal elements of `M` column by column."
function vecl(M::AbstractMatrix)
    n = size(M, 1)
    z = Vector{Float64}(undef, n * (n - 1) ÷ 2)
    k = 0
    @inbounds for j in 1:n-1, i in j+1:n
        z[k += 1] = M[i, j]
    end
    return z
end

"Symmetric matrix with zero diagonal and below-diagonal elements `z`."
function unvecl(z::AbstractVector)
    d = length(z)
    n = round(Int, (1 + sqrt(1 + 8d)) / 2)
    @assert n * (n - 1) ÷ 2 == d "length(z) must be n(n-1)/2"
    M = zeros(n, n)
    k = 0
    @inbounds for j in 1:n-1, i in j+1:n
        M[i, j] = z[k += 1]
        M[j, i] = M[i, j]
    end
    return M
end

"Generalized Fisher transformation: gamma = vecl(log C)."
function gft(C::AbstractMatrix)
    lam, Q = eigen(Symmetric(Matrix(float.(C))))
    @assert minimum(lam) > 0 "C must be positive definite"
    return vecl(Q * Diagonal(log.(lam)) * Q')
end

"exp(A) from the eigendecomposition, symmetrized."
function expA(lam::Vector{Float64}, Q::Matrix{Float64})
    E = Q * Diagonal(exp.(lam)) * Q'
    return (E + E') / 2
end

"""
log(diag(exp(A))) via row-wise weighted log-sum-exp: no overflow for any
spectrum, and no underflow when a row of `Q` is (nearly) orthogonal to the
top eigenspace (a global shift `m = maximum(lam)` returns `-Inf` for such
rows when the spectrum is very spread).
"""
function logdiagexp(lam::Vector{Float64}, Q::Matrix{Float64})
    a = lam' .+ 2.0 .* log.(abs.(Q))          # -Inf where Q[i,k] == 0
    m = vec(maximum(a, dims = 2))
    return m .+ log.(vec(sum(exp.(a .- m), dims = 2)))
end

"""
Divided differences of exp: `L[k,l] = (e^{lam_k}-e^{lam_l})/(lam_k-lam_l)`,
with the continuous value `e^{max(lam_k,lam_l)}` whenever `lam_k == lam_l`
(including repeated eigenvalues with `k != l`). Evaluated in the symmetric,
overflow-safe form `e^{max} * (-expm1(-r))/r` with `r = |lam_k - lam_l|`,
which avoids the `0 * Inf` hazard of one-sided formulas for widely spread
spectra.
"""
function loewner_exp(lam::Vector{Float64})
    n = length(lam)
    L = Matrix{Float64}(undef, n, n)
    @inbounds for l in 1:n, k in 1:n
        r = abs(lam[k] - lam[l])
        em = exp(max(lam[k], lam[l]))
        L[k, l] = r == 0.0 ? em : em * (-expm1(-r)) / r
    end
    return L
end

"""
Hessian-vector product (Daleckii-Krein), two matrix multiplications:
`H(x) v = diag( Q (L .* (Q' Diagonal(v) Q)) Q' )`.
"""
function hess_vec(lam::Vector{Float64}, Q::Matrix{Float64},
                  L::Matrix{Float64}, v::Vector{Float64})
    M = Q' * (v .* Q)                          # Q' Diagonal(v) Q
    return vec(sum((Q * (L .* M)) .* Q, dims = 2))
end

"""
Exact Hessian `H[i,j] = sum_{k,l} Q[i,k]Q[j,k] L[k,l] Q[i,l]Q[j,l]`,
formed in O(n^4) time and O(n^2) memory. Used by `inv_gft_newton` and
`inv_gft_broyden`; GFT-FP+N does not need it.
"""
function hessian(lam::Vector{Float64}, Q::Matrix{Float64},
                 L::Matrix{Float64} = loewner_exp(lam))
    n = length(lam)
    H = Matrix{Float64}(undef, n, n)
    R = similar(Q)
    RL = similar(Q)
    @inbounds for j in 1:n
        @views R .= Q .* Q[j:j, :]             # R[i,k] = Q[i,k] Q[j,k]
        mul!(RL, R, L)
        RL .*= R
        sum!(reshape(view(H, :, j), :, 1), RL)
    end
    return (H + H') / 2
end

_prep(z, x0) = begin
    A0 = unvecl(z)
    n = size(A0, 1)
    x = x0 === nothing ? zeros(n) : copy(float.(x0))
    (A0, n, x)
end

_eigx(A0, x) = eigen(Symmetric(A0 + Diagonal(x)))

# ------------------------------------------------------------------ solvers

"""
    inv_gft_fp(z; x0=nothing, tol=1e-13, maxit=5000)

Archakov-Hansen fixed point `x <- x - log diag(exp(A[x]))`, evaluated in
the log domain throughout.
"""
function inv_gft_fp(z::AbstractVector; x0 = nothing, tol = 1e-13,
                    maxit = 5000)
    A0, n, x = _prep(z, x0)
    hist = Float64[]
    lam, Q = zeros(n), zeros(n, n)
    err = Inf
    for k in 0:maxit-1
        F = _eigx(A0, x); lam, Q = F.values, F.vectors
        ell = logdiagexp(lam, Q)
        err = maximum(abs, expm1.(ell))
        push!(hist, err)
        if err < tol
            return InvResult(x, expA(lam, Q), k, k + 1, 0, err, true, hist)
        end
        x .-= ell
    end
    return InvResult(x, expA(lam, Q), maxit, maxit, 0, err, false, hist)
end

"""
    inv_gft_broyden(z; x0=nothing, tol=1e-13, maxit=500, warm=1,
                    globalized=false)

Broyden's method as in Chen, Fei and Yu (2025): residual
`F(x) = log diag(exp(A[x]))`, exact Jacobian computed once (after `warm`
fixed-point steps), then rank-one updates with Sherman-Morrison updates
of the inverse. One eigendecomposition per iteration. With
`globalized = true`, the one-step initialization is replaced by the same
log-domain fixed-point phase as GFT-FP+N: fixed-point steps until
`max_i ell_i <= log 2`, and only then is the Jacobian formed.
"""
function inv_gft_broyden(z::AbstractVector; x0 = nothing, tol = 1e-13,
                         maxit = 500, warm = 1, globalized = false)
    A0, n, x = _prep(z, x0)
    hist = Float64[]
    eighs = 0
    F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
    if globalized
        while maximum(logdiagexp(lam, Q)) > log(2.0)
            x .-= logdiagexp(lam, Q)
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
        end
    else
        for _ in 1:warm
            x .-= logdiagexp(lam, Q)
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
        end
    end
    ell = logdiagexp(lam, Q)
    dE = exp.(ell)
    J = hessian(lam, Q) ./ dE                  # Jacobian of log-diag: D^{-1} H
    Binv = try
        inv(J)
    catch
        return InvResult(x, expA(lam, Q), 0, eighs, 0, Inf, false, hist)
    end
    err = Inf
    for k in 0:maxit-1
        err = maximum(abs, expm1.(ell))
        push!(hist, err)
        if err < tol
            return InvResult(x, expA(lam, Q), k, eighs, 0, err, true, hist)
        end
        s = -(Binv * ell)
        x .+= s
        if !all(isfinite, x)                   # diverged: fail gracefully
            return InvResult(x, expA(lam, Q), k + 1, eighs, 0, Inf, false,
                             hist)
        end
        F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
        ell_new = logdiagexp(lam, Q)
        y = ell_new - ell
        Bys = Binv * y
        denom = dot(s, Bys)
        if abs(denom) > 1e-30                  # Sherman-Morrison
            Binv .+= ((s - Bys) * (s' * Binv)) ./ denom
        end
        ell = ell_new
    end
    return InvResult(x, expA(lam, Q), maxit, eighs, 0, err, false, hist)
end

"""
    inv_gft_newton(z; x0=nothing, tol=1e-13, maxit=500, warm=1,
                   safeguard=true)

Full Newton with the exact O(n^4) Hessian recomputed at every iteration,
Armijo backtracking on f, optional fixed-point warm start.

safeguard = false is Newton in the form of Chen, Fei and Yu (2025), one
fixed-point step then exact Newton steps, with an Armijo line search
added (the paper's "Newton with Armijo backtracking"); it can stagnate
at the rounding floor and return converged = false.  safeguard = true (the default) additionally
applies the two rounding-floor safeguards of inv_gft.
"""
function inv_gft_newton(z::AbstractVector; x0 = nothing, tol = 1e-13,
                        maxit = 500, warm = 1, safeguard = true)
    A0, n, x = _prep(z, x0)
    hist = Float64[]
    eighs = 0
    best_err = Inf
    stall = 0
    fp_finish = false
    F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
    for _ in 1:warm
        x .-= logdiagexp(lam, Q)
        F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
    end
    fval(lam, x) = sum(exp, lam) - sum(x)
    err = Inf
    for k in 0:maxit-1
        ell = logdiagexp(lam, Q)
        g = expm1.(ell)
        err = maximum(abs, g)
        push!(hist, err)
        if err < tol
            return InvResult(x, expA(lam, Q), k, eighs, 0, err, true, hist)
        end
        if safeguard
            # near the rounding floor the Newton direction is computed
            # from noise-dominated gradients and the Armijo test is
            # decided by cancellation in f; if progress stalls there,
            # finish with fixed-point steps, whose update x <- x - ell
            # remains contractive
            if err < 0.5 * best_err
                best_err = err
                stall = 0
            elseif err < 1e-9
                stall += 1
            end
            if fp_finish || stall >= 3
                fp_finish = true
                x .-= ell
                F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
                continue
            end
        end
        step = if all(isfinite, g)
            H = hessian(lam, Q)
            s = try
                -(cholesky(Symmetric(H)) \ g)
            catch
                nothing
            end
            s !== nothing && all(isfinite, s) ? s : nothing
        else
            nothing
        end
        if step === nothing                    # safeguard: fixed-point step
            x .-= ell
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            continue
        end
        f0 = fval(lam, x)
        gTs = dot(g, step)
        if safeguard && -gTs <= 1e-12 * (1 + abs(f0))
            # predicted decrease below the float resolution of f: the
            # Armijo test carries no information here, so take the full
            # Newton step untested (we are in the Newton basin)
            x .+= step
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            continue
        end
        t = 1.0
        ok = false
        lam_t, Q_t = lam, Q
        while t >= 1e-8
            F = _eigx(A0, x + t * step); lam_t, Q_t = F.values, F.vectors
            eighs += 1
            ft = fval(lam_t, x + t * step)
            if isfinite(ft) && ft <= f0 + 1e-4 * t * gTs
                ok = true
                break
            end
            t /= 2
        end
        if ok
            x .+= t * step
            lam, Q = lam_t, Q_t
        else
            x .-= logdiagexp(lam, Q)
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
        end
    end
    return InvResult(x, expA(lam, Q), maxit, eighs, 0, err, false, hist)
end

# ------------------------------------------------ quadrature preconditioner

"Gauss-Legendre nodes and weights on [0, 1] (Golub-Welsch)."
function gauss_legendre01(r::Int)
    r == 1 && return [0.5], [1.0]
    b = [k / sqrt(4k^2 - 1) for k in 1:r-1]
    F = eigen(SymTridiagonal(zeros(r), b))
    t = (F.values .+ 1) ./ 2
    a = F.vectors[1, :] .^ 2
    return t, a
end

"""
log phi_r(u): the ratio of the divided difference of exp to its r-point
Gauss-Legendre approximation at eigenvalue gap u,
phi_r(u) = [sinh(u/2)/(u/2)] / sum_j a_j cosh((t_j - 1/2) u) >= 1.
"""
function logphi(r::Int, u::Float64)
    t, a = gauss_legendre01(r)
    h = abs(u) / 2
    ls = h > 1e-8 ? h + log(-expm1(-2h)) - log(2h) : 0.0
    cs = similar(t)
    for j in eachindex(t)                       # log(a_j cosh((t_j - 1/2) u))
        cj = abs(t[j] - 0.5) * abs(u)
        cs[j] = log(a[j]) + cj + log1p(exp(-2cj)) - log(2.0)
    end
    m = maximum(cs)
    return ls - (m + log(sum(exp, cs .- m)))
end

"""
Smallest Gauss-Legendre order r <= rmax with certified condition number
phi_r(spread) <= kappa for the preconditioned system (phi_r increases in
the spread); returns 0 if none.
"""
function choose_order(spread::Float64, kappa::Float64, rmax::Int)
    for r in 1:rmax
        logphi(r, spread) <= log(kappa) && return r
    end
    return 0
end

"""
    quadrature_precond(w, Q; kappa=2.0, rmin=1, rmax=40, nmin=0)

Cholesky factor of M_r = sum_j a_j e^{t_j A} o e^{(1-t_j) A} for the
symmetric matrix with eigenvalues `w` (already shifted so that
max(w) = 0, matching the e^{-c} scaling of the products) and
eigenvectors `Q`, with r the smallest Gauss-Legendre order that
certifies M_r <= H <= kappa M_r from the spectral spread. Returns
`(factor, r)`; `factor === nothing` and `r == 0` means the diagonal
preconditioner should be used (smallest certifying order outside
[rmin, rmax], or n < nmin), and `factor === nothing` with `r > 0` a
failed factorization. `rfixed > 0` prescribes the order.
"""
function quadrature_precond(w::Vector{Float64}, Q::Matrix{Float64};
                            kappa = 2.0, rmin = 1, rmax = 40, nmin = 0,
                            rfixed = 0)
    n = length(w)
    spread = maximum(w) - minimum(w)
    if rfixed > 0                               # prescribed order (tests)
        r = rfixed
    else
        r = spread > 1000.0 ? 0 : choose_order(spread, Float64(kappa), rmax)
        (r == 0 || r < rmin || n < nmin) && return nothing, 0
    end
    t, a = gauss_legendre01(r)
    M = zeros(n, n)
    half = (r + 1) ÷ 2
    for j in 1:half
        k = r + 1 - j                           # paired node 1 - t_j
        Ej = (Q .* exp.(t[j] .* w)') * Q'
        if j == k
            M .+= a[j] .* (Ej .* Ej)
        else
            Ek = (Q .* exp.(t[k] .* w)') * Q'
            M .+= (a[j] + a[k]) .* (Ej .* Ek)
        end
    end
    F = cholesky(Symmetric((M + M') ./ 2); check = false)
    return (issuccess(F) ? F : nothing), r
end

"""
    inv_gft(z; x0=nothing, tol=1e-13, maxit=500, exact_hess=false,
            residual=:log, forcing=:quad, normalize=true, phase=false,
            adaptive=true, delta=1.0)

GFT-FP+N (recommended). Every evaluated point is normalized by the exact
minimization of f along the vector of ones, x <- x - s 1 with
s = log(tr(exp A[x]) / n), which costs no eigendecomposition and makes
tr exp(A[x]) = n. Fixed-point steps x <- x - ell, ell = log diag(exp A),
are taken while some ell_i < -700 (and, with `phase = true`, while
max ell > log(1 + delta)). Otherwise an inexact Newton step for the
equation ell(x) = 0 is computed: the system H delta = -D ell,
D = diag(exp ell), is solved matrix-free by conjugate gradients with
preconditioner D, exact Hessian-vector products (two matrix
multiplications each) and forcing tolerance eta = min(1/2, ||ell||)
on the preconditioned residual (locally quadratic), capped at 2n
products. Step acceptance: a non-finite or non-descent direction
(g' delta >= 0, g = exp(ell) - 1) is replaced by a fixed-point step;
the full step is taken untested when its predicted decrease is below
the floating-point resolution of f and ||ell||_inf < 1e-3; otherwise
Armijo backtracking on f from t = min(1, 2 t_prev), with a fixed-point
step substituted if the search fails. Near the rounding floor, if
||g||_inf < 1e-9 fails to halve three times, the computation is
finished by fixed-point steps (terminal mode).

Variants for the ablation of the paper's Section S5:
`residual = :gradient` solves H delta = -g (Newton for grad f = 0);
`forcing = :sqrt` uses eta = min(1/2, ||.||^{1/2}) (order 3/2);
`normalize = false` skips the normalization; `phase = true` keeps the
initial fixed-point phase while max ell > log(1 + delta);
`adaptive = false` starts every line search at t = 1. The first
version of the paper is `residual = :gradient, forcing = :sqrt,
normalize = false, phase = true, adaptive = false`.
`exact_hess = true` solves the same Newton system with the explicit
O(n^4) Hessian instead of conjugate gradients (the paper's full-Newton
comparator); everything else is identical.

`preconditioner = :quadrature` replaces the diagonal preconditioner of
the conjugate-gradient solve by the certified quadrature model M_r of
Section S6 of the paper at every Newton step (order chosen from the
spectral spread so that kappa(M_r^{-1} H) <= `kappa`, default 2);
`preconditioner = :auto` applies the paper's selection rule: M_r when
an order in [`rmin`, `rmax`] = [2, 8] certifies the bound and
n >= `nmin` (0), the diagonal otherwise (a single certifying node
means the diagonal solve is already fast). The forcing test is
unchanged.
"""
function inv_gft(z::AbstractVector; kwargs...)
    return _inv_gft(z; kwargs...)[1]
end

# normalization: shift so that tr exp(A[x]) = n (no eigendecomposition)
function _normalize!(x, lam, ell)
    c = maximum(lam)
    s = c + log(sum(exp, lam .- c) / length(lam))
    x .-= s; lam .-= s; ell .-= s
    return s
end

function _inv_gft(z::AbstractVector; x0 = nothing, tol = 1e-13,
                  maxit = 500, delta = 1.0, exact_hess = false,
                  residual = :log, forcing = :quad, normalize = true,
                  phase = false, adaptive = true,
                  preconditioner = :diagonal, kappa = 2.0, rmin = 2,
                  rmax = 8, nmin = 0)
    residual in (:log, :gradient) || error("residual must be :log or :gradient")
    forcing in (:quad, :sqrt) || error("forcing must be :quad or :sqrt")
    preconditioner in (:diagonal, :quadrature, :auto) ||
        error("preconditioner must be :diagonal, :quadrature or :auto")
    A0, n, x = _prep(z, x0)
    builds = 0; nodes = 0; cholfail = 0; tsetup = 0.0   # preconditioner stats
    stats() = (builds = builds, nodes = nodes, cholfail = cholfail, tsetup = tsetup)
    hist = Float64[]
    eighs = 0
    hvs = 0
    best_err = Inf
    stall = 0
    fp_finish = false
    tprev = 1.0
    F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
    ell = logdiagexp(lam, Q)
    normalize && _normalize!(x, lam, ell)
    # f evaluated from the eigenvalues, scaled to avoid overflow
    fval(lam, x) = (c = maximum(lam); exp(c) * sum(exp, lam .- c) - sum(x))
    err = Inf
    for k in 0:maxit-1
        if (phase && maximum(ell) > log(1.0 + delta)) ||
           minimum(ell) < -700.0               # ---- fixed-point step
            push!(hist, expm1(min(maximum(ell), 700.0)))
            x .-= ell
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            ell = logdiagexp(lam, Q)
            normalize && _normalize!(x, lam, ell)
            tprev = 1.0
            continue
        end
        g = expm1.(ell)
        err = maximum(abs, g)
        push!(hist, err)
        if err < tol
            return InvResult(x, expA(lam, Q), k, eighs, hvs, err, true, hist), lam, Q, stats()
        end
        # terminal mode near the rounding floor: fixed-point steps
        if err < 0.5 * best_err
            best_err = err
            stall = 0
        elseif err < 1e-9
            stall += 1
        end
        if fp_finish || stall >= 3
            fp_finish = true
            x .-= ell
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            ell = logdiagexp(lam, Q)
            normalize && _normalize!(x, lam, ell)
            continue
        end
        # ---- Newton step, all quantities scaled by e^{-c}, c = max(lam)
        c = maximum(lam)
        dEs = exp.(ell .- c)                   # e^{-c} diag(exp A)
        Ls = loewner_exp(lam .- c)             # e^{-c} L
        local rhs, nres, stopfun
        if residual == :log
            rhs = -dEs .* ell                  # e^{-c} (-D ell)
            nres = norm(ell)
            eta = min(0.5, forcing == :quad ? nres : sqrt(nres))
            stopfun = r -> norm(r ./ dEs) <= eta * nres
        else
            rhs = -exp(-c) .* g                # e^{-c} (-g)
            nres = norm(g)
            eta = min(0.5, forcing == :quad ? nres : sqrt(nres))
            thr = eta * norm(rhs)
            stopfun = r -> norm(r) <= thr
        end
        local step
        if exact_hess
            step = try
                cholesky(Symmetric(hessian(lam, Q, Ls))) \ rhs
            catch
                fill(NaN, n)
            end
        else                                   # preconditioned CG from 0
            # preconditioner: diagonal D, or the quadrature model M_r
            # (scaled by e^{-c} like the products), see quadrature_precond
            CF = nothing
            if preconditioner != :diagonal
                t0 = time_ns()
                CF, r_used = quadrature_precond(lam .- c, Q; kappa = kappa,
                    rmin = preconditioner == :auto ? rmin : 1,
                    rmax = preconditioner == :auto ? rmax : 40,
                    nmin = preconditioner == :auto ? nmin : 0)
                tsetup += (time_ns() - t0) / 1e9
                if r_used > 0
                    builds += 1; nodes += r_used
                    CF === nothing && (cholfail += 1)
                end
            end
            applyM = CF === nothing ? (r -> r ./ dEs) : (r -> CF \ r)
            step = zeros(n)
            r = copy(rhs)
            p = applyM(r)
            rz = dot(r, p)
            for _ in 1:2n
                Hp = hess_vec(lam, Q, Ls, p); hvs += 1
                a = rz / dot(p, Hp)
                step .+= a .* p
                r .-= a .* Hp
                stopfun(r) && break
                zz = applyM(r)
                rz_new = dot(r, zz)
                p .= zz .+ (rz_new / rz) .* p
                rz = rz_new
            end
        end
        f0 = fval(lam, x)
        gTs = dot(g, step)
        # ---- acceptance: finiteness and descent first
        if !all(isfinite, step) || !(gTs < 0.0) || !isfinite(f0)
            x .-= ell
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            ell = logdiagexp(lam, Q)
            normalize && _normalize!(x, lam, ell)
            tprev = 1.0
            continue
        end
        if -gTs <= 1e-12 * (1.0 + abs(f0)) && maximum(abs, ell) < 1e-3
            # predicted decrease below the float resolution of f, and in
            # the neighbourhood of the solution: full step, untested
            x .+= step
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            ell = logdiagexp(lam, Q)
            normalize && _normalize!(x, lam, ell)
            continue
        end
        t = adaptive ? min(1.0, 2.0 * tprev) : 1.0
        ok = false
        lam_t, Q_t, ell_t, x_t = lam, Q, ell, x
        while t >= 1e-8                        # Armijo backtracking on f
            x_t = x + t * step
            F = _eigx(A0, x_t); lam_t, Q_t = F.values, F.vectors
            eighs += 1
            ell_t = logdiagexp(lam_t, Q_t)
            normalize && _normalize!(x_t, lam_t, ell_t)
            ft = fval(lam_t, x_t)
            if isfinite(ft) && ft <= f0 + 1e-4 * t * gTs
                ok = true
                break
            end
            t /= 2
        end
        if ok
            x, lam, Q, ell = x_t, lam_t, Q_t, ell_t
            tprev = t
        else                                   # safeguard: fixed-point step
            x .-= ell
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            ell = logdiagexp(lam, Q)
            normalize && _normalize!(x, lam, ell)
            tprev = 1.0
        end
    end
    return InvResult(x, expA(lam, Q), maxit, eighs, hvs, err, false, hist), lam, Q, stats()
end

"""
    gft_predict(z_prev, x_prev, lam, Q, z_new; rtol=1e-3)

Tangent predictor for sequential inversion. Given the solution `x_prev`
of `z_prev` with its eigendecomposition `(lam, Q)` of A[x_prev; z_prev],
returns the first-order prediction x_prev + p of the solution for
`z_new`, with H p = -(B dz + D ell) solved by preconditioned conjugate
gradients to relative residual `rtol` (no eigendecomposition). B dz is
the derivative of diag(exp A) in the direction of the off-diagonal
change, three matrix multiplications. Returns `(xhat, hvs)`.
"""
function gft_predict(z_prev::AbstractVector, x_prev::AbstractVector,
                     lam::Vector{Float64}, Q::Matrix{Float64},
                     z_new::AbstractVector; rtol = 1e-3)
    n = length(x_prev)
    dA = unvecl(z_new .- z_prev)
    c = maximum(lam)
    Ls = loewner_exp(lam .- c)
    ell = logdiagexp(lam, Q)
    dEs = exp.(ell .- c)
    M = Q' * dA * Q                            # B dz (scaled): three mults
    bdz = vec(sum((Q * (Ls .* M)) .* Q, dims = 2))
    rhs = -(bdz .+ dEs .* ell)
    nr = norm(rhs)
    p = zeros(n); r = copy(rhs); pp = r ./ dEs; rz = dot(r, pp); hvs = 0
    for _ in 1:2n
        norm(r) <= rtol * nr && break
        Hp = hess_vec(lam, Q, Ls, pp); hvs += 1
        a = rz / dot(pp, Hp)
        p .+= a .* pp
        r .-= a .* Hp
        rz_new = dot(r, r ./ dEs)
        pp .= r ./ dEs .+ (rz_new / rz) .* pp
        rz = rz_new
    end
    return x_prev .+ p, hvs
end

"""
    inv_gft_path(zs; predictor=true, rtol=1e-3, tol=1e-13, kwargs...)

Sequential inversion of a vector of z's with warm starts: each solve
starts from the previous solution, or, with `predictor = true`, from
the tangent prediction of `gft_predict`. Returns a vector of
`InvResult` (the `hvs` field of each includes the predictor's
products) and the number of B dz evaluations.
"""
function inv_gft_path(zs::AbstractVector; predictor = true, rtol = 1e-3,
                      tol = 1e-13, kwargs...)
    out = InvResult[]
    x0 = nothing; lam = Float64[]; Q = zeros(0, 0); nbdz = 0
    for (t, z) in enumerate(zs)
        hv_pred = 0
        if t > 1 && predictor
            x0, hv_pred = gft_predict(zs[t-1], x0, lam, Q, z; rtol = rtol)
            nbdz += 1
        end
        r, lam, Q, _ = _inv_gft(z; x0 = x0, tol = tol, kwargs...)
        push!(out, InvResult(r.x, r.C, r.iters, r.eighs, r.hvs + hv_pred,
                             r.err, r.converged, r.hist))
        x0 = r.x
    end
    return out, nbdz
end

"""
    inv_gft_anderson(z; x0=nothing, tol=1e-13, maxit=5000, m=5,
                     guarded=false, theta=0.25)

Anderson acceleration (type II, memory `m`) of the fixed-point map
`g(x) = x - ell(x)`, `ell = log diag(exp(A[x]))`, with residual
`r(x) = -ell(x)` evaluated in the log domain. One eigendecomposition per
iteration; the mixing coefficients solve a small least-squares problem.
A standard Jacobian-free accelerator, included as a comparator.

With `guarded = true` an Anderson proposal y is accepted only if
f(y) <= f(x) - theta V(x), V(x) = sum_i (e^{ell_i} - 1 - ell_i), the
lower bound on the decrease of the fixed-point step (Proposition 1 of
the paper); otherwise the memory is cleared and the fixed-point step
is taken. Proposals are accepted untested once
theta V(x) <= 1e-12 (1 + |f(x)|). Every step then decreases f by at
least theta V(x), which gives global convergence (Section S5).
"""
function inv_gft_anderson(z::AbstractVector; x0 = nothing, tol = 1e-13,
                          maxit = 5000, m = 5, guarded = false, theta = 0.25)
    A0, n, x = _prep(z, x0)
    hist = Float64[]
    X = Vector{Vector{Float64}}(); R = Vector{Vector{Float64}}()
    fval(lam, x) = (c = maximum(lam); exp(c) * sum(exp, lam .- c) - sum(x))
    F = _eigx(A0, x); lam, Q = F.values, F.vectors
    eighs = 1
    ell = logdiagexp(lam, Q)
    err = Inf
    for k in 0:maxit-1
        err = maximum(abs, expm1.(ell))
        push!(hist, err)
        if err < tol
            return InvResult(x, expA(lam, Q), k, eighs, 0, err, true, hist)
        end
        r = -ell
        push!(X, copy(x)); push!(R, copy(r))
        if length(X) > m + 1
            popfirst!(X); popfirst!(R)
        end
        mk = length(X) - 1
        local y
        plain = mk == 0
        if plain
            y = x + r
        else
            dR = hcat([R[i+1] - R[i] for i in 1:mk]...)
            dX = hcat([X[i+1] - X[i] for i in 1:mk]...)
            gam = _lstsq(dR, r)                # least squares
            y = x + r - (dX + dR) * gam
        end
        if !all(isfinite, y)
            return InvResult(x, expA(lam, Q), k + 1, eighs, 0, Inf, false, hist)
        end
        F = _eigx(A0, y); lam_y, Q_y = F.values, F.vectors; eighs += 1
        ell_y = logdiagexp(lam_y, Q_y)
        if guarded && !plain
            f0 = fval(lam, x)
            V0 = _vfun(ell)
            if theta * V0 > 1e-12 * (1.0 + abs(f0))
                fy = fval(lam_y, y)
                if !(isfinite(fy) && fy <= f0 - theta * V0)
                    empty!(X); empty!(R)       # reject: fixed-point step
                    y = x + r
                    F = _eigx(A0, y); lam_y, Q_y = F.values, F.vectors; eighs += 1
                    ell_y = logdiagexp(lam_y, Q_y)
                end
            end
        end
        x, lam, Q, ell = y, lam_y, Q_y, ell_y
        if !all(isfinite, ell)
            return InvResult(x, expA(lam, Q), k + 1, eighs, 0, Inf, false, hist)
        end
    end
    return InvResult(x, expA(lam, Q), maxit, eighs, 0, err, false, hist)
end

# minimum-norm least-squares solution, robust to rank deficiency (for
# memory m >= n the difference matrix is square or wide, and a plain
# backslash would attempt an LU solve and can throw on singularity)
function _lstsq(A::Matrix{Float64}, b::Vector{Float64})
    F = svd(A)
    tol = eps(Float64) * max(size(A)...) * (isempty(F.S) ? 0.0 : F.S[1])
    sinv = [s > tol ? 1.0 / s : 0.0 for s in F.S]
    return F.V * (sinv .* (F.U' * b))
end

# V(x) = sum_i (e^{ell_i} - 1 - ell_i), cancellation-free
function _vfun(ell)
    s = 0.0
    @inbounds for t in ell
        s += abs(t) < 1e-3 ? t * t * (0.5 + t * (1 / 6 + t * (1 / 24 + t / 120))) :
                             expm1(t) - t
    end
    return s
end

"""
    inv_gft_lbfgs(z; x0=nothing, tol=1e-13, maxit=500, m=10, globalized=false)

Limited-memory BFGS on f(x) = tr(exp(A[x])) - sum(x) with gradient
g = diag(exp(A[x])) - 1, two-loop recursion with memory `m`, initial
scaling (s'y)/(y'y), and Armijo backtracking on f (constant 1e-4, step
halving over eight decades; nonfinite trial values are rejected; a step
whose predicted decrease is below roundoff in f is accepted untested,
as in GFT-FP+N). The first step after a start or reset has unit
sup-norm length. Every
function or gradient evaluation is one eigendecomposition. Started at
x = 0, or, with `globalized = true`, after the same log-domain
fixed-point phase as GFT-FP+N (fixed-point steps until
`max_i ell_i <= log 2`). A standard quasi-Newton optimizer, included as
a comparator.
"""
function inv_gft_lbfgs(z::AbstractVector; x0 = nothing, tol = 1e-13,
                       maxit = 500, m = 10, globalized = false)
    A0, n, x = _prep(z, x0)
    hist = Float64[]
    eighs = 0
    fval(lam, x) = sum(exp, lam) - sum(x)
    F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
    if globalized
        while maximum(logdiagexp(lam, Q)) > log(2.0)
            x .-= logdiagexp(lam, Q)
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
        end
    end
    S = Vector{Vector{Float64}}(); Y = Vector{Vector{Float64}}()
    ell = logdiagexp(lam, Q)
    g = expm1.(ell)
    f0 = fval(lam, x)
    err = maximum(abs, g)
    for k in 0:maxit-1
        push!(hist, err)
        if err < tol
            return InvResult(x, expA(lam, Q), k, eighs, 0, err, true, hist)
        end
        if !all(isfinite, g) || !isfinite(f0)
            return InvResult(x, expA(lam, Q), k, eighs, 0, Inf, false, hist)
        end
        # two-loop recursion
        q = copy(g)
        L = length(S)
        alpha = zeros(L)
        for i in L:-1:1
            rho_i = 1.0 / dot(Y[i], S[i])
            alpha[i] = rho_i * dot(S[i], q)
            q .-= alpha[i] .* Y[i]
        end
        if L > 0
            q .*= dot(S[L], Y[L]) / dot(Y[L], Y[L])
        end
        for i in 1:L
            rho_i = 1.0 / dot(Y[i], S[i])
            beta = rho_i * dot(Y[i], q)
            q .+= (alpha[i] - beta) .* S[i]
        end
        d = -q
        gTd = dot(g, d)
        if gTd >= 0                            # not a descent direction: reset
            empty!(S); empty!(Y); d = -g; gTd = -dot(g, g)
        end
        # first step after a (re)start: unit length in the sup norm
        t = isempty(S) ? min(1.0, 1.0 / maximum(abs, d)) : 1.0
        ok = false
        lam_t, Q_t, ft = lam, Q, f0
        tmin = 1e-8 * t
        while t >= tmin
            F = _eigx(A0, x + t * d); lam_t, Q_t = F.values, F.vectors
            eighs += 1
            ft = fval(lam_t, x + t * d)
            # accept on sufficient decrease, or untested when the
            # predicted decrease is below roundoff in f (same rule as
            # the Newton phase of GFT-FP+N)
            if isfinite(ft) && (ft <= f0 + 1e-4 * t * gTd ||
                                abs(t * gTd) <= 1e-12 * (1 + abs(f0)))
                ok = true
                break
            end
            t /= 2
        end
        if !ok
            return InvResult(x, expA(lam, Q), k + 1, eighs, 0, err, false, hist)
        end
        xn = x + t * d
        ell_n = logdiagexp(lam_t, Q_t)
        gn = expm1.(ell_n)
        s = xn - x; y = gn - g
        if dot(s, y) > 1e-12 * dot(y, y)      # curvature condition
            push!(S, s); push!(Y, y)
            if length(S) > m
                popfirst!(S); popfirst!(Y)
            end
        end
        x, lam, Q, g, f0 = xn, lam_t, Q_t, gn, ft
        err = maximum(abs, g)
    end
    return InvResult(x, expA(lam, Q), maxit, eighs, 0, err, false, hist)
end

end # module
