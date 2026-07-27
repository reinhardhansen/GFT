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
    inv_gft_newton(z; x0=nothing, tol=1e-13, maxit=500, warm=1)

Full Newton with the exact O(n^4) Hessian recomputed at every iteration,
Armijo backtracking on f, optional fixed-point warm start.
"""
function inv_gft_newton(z::AbstractVector; x0 = nothing, tol = 1e-13,
                        maxit = 500, warm = 1)
    A0, n, x = _prep(z, x0)
    hist = Float64[]
    eighs = 0
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

"""
    inv_gft(z; x0=nothing, tol=1e-13, maxit=200, delta=1.0)

GFT-FP+N (recommended). Phase 1: fixed-point steps in the log domain
while `||diag(exp(A))-1||_inf > delta` (robust to overflow). Phase 2:
inexact Newton; the system `H step = -g` is solved matrix-free by
conjugate gradients with Jacobi preconditioner `diag(exp(A))`, exact
Hessian-vector products (two matrix multiplications each) and forcing
tolerance `eta = min(1/2, sqrt(||g||))` (Eisenstat and Walker, 1996;
superlinear of order 3/2). Armijo backtracking on f, with the full
Newton step taken untested once the predicted decrease is below the
floating-point resolution of f, and a fixed-point step substituted if
the line search fails.
"""
function inv_gft(z::AbstractVector; x0 = nothing, tol = 1e-13,
                 maxit = 500, delta = 1.0, exact_hess = false)
    A0, n, x = _prep(z, x0)
    hist = Float64[]
    eighs = 0
    hvs = 0
    best_err = Inf
    stall = 0
    fp_finish = false
    F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
    fval(lam, x) = sum(exp, lam) - sum(x)
    err = Inf
    for k in 0:maxit-1
        ell = logdiagexp(lam, Q)               # log domain: no overflow
        if maximum(ell) > log(1.0 + delta) ||
           minimum(ell) < -700.0               # ---- phase 1: fixed point
            # equivalent to ||diag(exp A) - 1||_inf > delta, tested without
            # exponentiating (g is never formed while it could overflow);
            # the second test keeps the fixed point in charge while any
            # diagonal of exp(A) would underflow (exp(ell) == 0 in double)
            push!(hist, expm1(min(maximum(ell), 700.0)))
            x .-= ell
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            continue
        end
        g = expm1.(ell)                        # finite here: |g| <= delta
        err = maximum(abs, g)
        push!(hist, err)
        if err < tol
            return InvResult(x, expA(lam, Q), k, eighs, hvs, err, true, hist)
        end
        # near the rounding floor the Newton direction is computed from
        # noise-dominated gradients; if progress stalls there, finish with
        # fixed-point steps, whose update x <- x - ell remains contractive
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
        dE = exp.(ell)                         # = diag(exp A) directly; the
        # algebraically equal 1 + expm1(ell) cancels to 0 for ell <= -37
        # ---- phase 2: Newton
        L = loewner_exp(lam)
        local step
        if exact_hess                          # explicit O(n^4) Hessian:
            # identical algorithm, only the linear solver differs
            step = -(cholesky(Symmetric(hessian(lam, Q, L))) \ g)
        else                                   # matrix-free preconditioned CG
            eta = min(0.5, sqrt(norm(g)))
            step = zeros(n)
            r = -copy(g)
            p = r ./ dE
            rz = dot(r, p)
            normg = norm(g)
            for _ in 1:2n
                Hp = hess_vec(lam, Q, L, p); hvs += 1
                a = rz / dot(p, Hp)
                step .+= a .* p
                r .-= a .* Hp
                norm(r) <= eta * normg && break
                rz_new = dot(r, r ./ dE)
                p .= r ./ dE .+ (rz_new / rz) .* p
                rz = rz_new
            end
        end
        f0 = fval(lam, x)
        gTs = dot(g, step)
        if -gTs <= 1e-12 * (1.0 + abs(f0))
            # predicted decrease below float resolution of f: Newton basin
            x .+= step
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
            continue
        end
        t = 1.0
        ok = false
        lam_t, Q_t = lam, Q
        while t >= 1e-8                        # Armijo backtracking
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
        else                                   # safeguard: fixed-point step
            x .-= ell
            F = _eigx(A0, x); lam, Q = F.values, F.vectors; eighs += 1
        end
    end
    return InvResult(x, expA(lam, Q), maxit, eighs, hvs, err, false, hist)
end

end # module
