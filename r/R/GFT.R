# GFT: generalized Fisher transformation of correlation matrices and its
# inverse.
#
# The inverse is computed from the variational characterization
#
#     x*(z) = argmin_x  tr(exp(A[x;z])) - sum(x),
#
# where A[x;z] is symmetric with off-diagonal elements z and diagonal x
# (Archakov and Hansen, 2021; this paper).  Solvers:
#
#   inv_gft          -- GFT-FP+N: normalized log-residual Newton steps,
#                       matrix-free via preconditioned conjugate gradients,
#                       with fixed-point safeguards (recommended)
#   inv_gft_fp       -- Archakov-Hansen fixed point (log domain)
#   inv_gft_broyden  -- Broyden's method as in Chen, Fei and Yu (2025)
#   inv_gft_newton   -- Newton with the exact O(n^4) Hessian and an Armijo
#                       line search (comparator)
#   inv_gft_anderson -- Anderson acceleration of the fixed point, optionally
#                       safeguarded by the fixed-point decrease bound
#   inv_gft_lbfgs    -- limited-memory BFGS on the objective (comparator)
#   gft_predict, inv_gft_path -- tangent predictor and sequential inversion
#
# All solvers count eigendecompositions ("eighs"), the dominant O(n^3)
# kernel, and Hessian-vector products ("hvs") where applicable.
#
# This file is a line-faithful port of the Julia reference implementation
# (julia/src/GFT.jl); the two sources are kept in one-to-one
# correspondence to ease cross-language verification.
#
# Uses base R only.

# ------------------------------------------------------------------ types

.inv_result <- function(x, C, iters, eighs, hvs, err, converged, hist) {
    structure(list(x = x, C = C, iters = iters, eighs = eighs, hvs = hvs,
                   err = err, converged = converged, hist = hist),
              class = "gft_inv")
}

# -------------------------------------------------------------- utilities

# Stack the below-diagonal elements of M column by column.
vecl <- function(M) {
    M <- as.matrix(M)
    as.numeric(M[lower.tri(M)])
}

# Symmetric matrix with zero diagonal and below-diagonal elements z.
unvecl <- function(z) {
    z <- as.numeric(z)
    d <- length(z)
    n <- round((1 + sqrt(1 + 8 * d)) / 2)
    if (n * (n - 1) / 2 != d)
        stop("length(z) must be n(n-1)/2")
    M <- matrix(0, n, n)
    M[lower.tri(M)] <- z
    M + t(M)
}

# Generalized Fisher transformation: gamma = vecl(log C).
gft <- function(C) {
    C <- as.matrix(C)
    if (nrow(C) != ncol(C) || !is.numeric(C))
        stop("C must be a square numeric matrix")
    C <- (C + t(C)) / 2
    F <- eigen(C, symmetric = TRUE)
    lam <- F$values
    Q <- F$vectors
    if (min(lam) <= 0)
        stop("C must be positive definite")
    vecl(Q %*% (log(lam) * t(Q)))
}

# exp(A) from the eigendecomposition, symmetrized.
.expA <- function(lam, Q) {
    E <- Q %*% (exp(lam) * t(Q))
    (E + t(E)) / 2
}

# log(diag(exp(A))) via row-wise weighted log-sum-exp: no overflow for any
# spectrum, and no underflow when a row of Q is (nearly) orthogonal to the
# top eigenspace (a global shift m = max(lam) returns -Inf for such rows
# when the spectrum is very spread).
.logdiagexp <- function(lam, Q) {
    a <- sweep(2 * log(abs(Q)), 2, lam, "+")  # -Inf where Q[i,k] == 0
    m <- apply(a, 1, max)
    m + log(rowSums(exp(a - m)))
}

# Divided differences of exp: L[k,l] = (e^{lam_k}-e^{lam_l})/(lam_k-lam_l),
# with the continuous value e^{max(lam_k,lam_l)} whenever lam_k == lam_l
# (including repeated eigenvalues with k != l).  Evaluated in the
# symmetric, overflow-safe form e^{max} * (-expm1(-r))/r with
# r = |lam_k - lam_l|, which avoids the 0 * Inf hazard of one-sided
# formulas for widely spread spectra.
.loewner_exp <- function(lam) {
    r <- abs(outer(lam, lam, "-"))
    em <- exp(outer(lam, lam, pmax))
    ratio <- -expm1(-r) / r                   # NaN where r == 0
    ratio[r == 0] <- 1
    em * ratio
}

# Hessian-vector product (Daleckii-Krein), two matrix multiplications:
# H(x) v = diag( Q (L * (Q' Diag(v) Q)) Q' ).
.hess_vec <- function(lam, Q, L, v) {
    M <- crossprod(Q, v * Q)                  # Q' Diag(v) Q
    rowSums((Q %*% (L * M)) * Q)
}

# Exact Hessian H[i,j] = sum_{k,l} Q[i,k]Q[j,k] L[k,l] Q[i,l]Q[j,l],
# formed in O(n^4) time and O(n^2) memory.  Used by inv_gft_newton and
# inv_gft_broyden; GFT-FP+N does not need it.
.hessian <- function(lam, Q, L = .loewner_exp(lam)) {
    n <- length(lam)
    H <- matrix(0, n, n)
    for (j in seq_len(n)) {
        R <- Q * rep(Q[j, ], each = n)        # R[i,k] = Q[i,k] Q[j,k]
        H[, j] <- rowSums((R %*% L) * R)
    }
    (H + t(H)) / 2
}

.prep <- function(z, x0) {
    A0 <- unvecl(z)
    n <- nrow(A0)
    x <- if (is.null(x0)) numeric(n) else as.numeric(x0)
    if (length(x) != n)
        stop("x0 must have length n")
    list(A0 = A0, n = n, x = x)
}

.eigx <- function(A0, x) {
    A <- A0
    diag(A) <- x
    eigen(A, symmetric = TRUE)
}


# ---------------------------------------------------------------- solvers

# Archakov-Hansen fixed point x <- x - log diag(exp(A[x])), evaluated in
# the log domain throughout.
inv_gft_fp <- function(z, x0 = NULL, tol = 1e-13, maxit = 5000) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    lam <- numeric(n); Q <- matrix(0, n, n)
    err <- Inf
    for (k in 0:(maxit - 1)) {
        F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
        ell <- .logdiagexp(lam, Q)
        err <- max(abs(expm1(ell)))
        hist[nh <- nh + 1] <- err
        if (err < tol)
            return(.inv_result(x, .expA(lam, Q), k, k + 1, 0, err, TRUE,
                               hist[seq_len(nh)]))
        x <- x - ell
    }
    .inv_result(x, .expA(lam, Q), maxit, maxit, 0, err, FALSE,
                hist[seq_len(nh)])
}

# Broyden's method as in Chen, Fei and Yu (2025): residual
# F(x) = log diag(exp(A[x])), exact Jacobian computed once (after `warm`
# fixed-point steps), then rank-one updates with Sherman-Morrison updates
# of the inverse.  One eigendecomposition per iteration.  With
# globalized = TRUE, the one-step initialization is replaced by the same
# log-domain fixed-point phase as GFT-FP+N: fixed-point steps until
# max_i ell_i <= log 2, and only then is the Jacobian formed.
inv_gft_broyden <- function(z, x0 = NULL, tol = 1e-13, maxit = 500,
                            warm = 1, globalized = FALSE) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    eighs <- 0
    F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors; eighs <- eighs + 1
    if (globalized) {
        while (max(.logdiagexp(lam, Q)) > log(2)) {
            x <- x - .logdiagexp(lam, Q)
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
        }
    } else {
        for (i in seq_len(warm)) {
            x <- x - .logdiagexp(lam, Q)
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
        }
    }
    ell <- .logdiagexp(lam, Q)
    dE <- exp(ell)
    J <- .hessian(lam, Q) / dE                # Jacobian of log-diag: D^{-1} H
    Binv <- tryCatch(solve(J), error = function(e) NULL)
    if (is.null(Binv))
        return(.inv_result(x, .expA(lam, Q), 0, eighs, 0, Inf, FALSE,
                           hist[seq_len(nh)]))
    err <- Inf
    for (k in 0:(maxit - 1)) {
        err <- max(abs(expm1(ell)))
        hist[nh <- nh + 1] <- err
        if (err < tol)
            return(.inv_result(x, .expA(lam, Q), k, eighs, 0, err, TRUE,
                               hist[seq_len(nh)]))
        s <- -as.numeric(Binv %*% ell)
        x <- x + s
        if (!all(is.finite(x)))               # diverged: fail gracefully
            return(.inv_result(x, .expA(lam, Q), k + 1, eighs, 0, Inf,
                               FALSE, hist[seq_len(nh)]))
        F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
        eighs <- eighs + 1
        ell_new <- .logdiagexp(lam, Q)
        y <- ell_new - ell
        Bys <- as.numeric(Binv %*% y)
        denom <- sum(s * Bys)
        if (abs(denom) > 1e-30)               # Sherman-Morrison
            Binv <- Binv + outer(s - Bys, as.numeric(crossprod(Binv, s))) /
                denom
        ell <- ell_new
    }
    .inv_result(x, .expA(lam, Q), maxit, eighs, 0, err, FALSE,
                hist[seq_len(nh)])
}

# Full Newton with the exact O(n^4) Hessian recomputed at every iteration,
# Armijo backtracking on f, optional fixed-point warm start.
#
# safeguard = FALSE reproduces the published comparator of Chen, Fei and
# Yu (2025) with only the Armijo line search added, which is the variant
# benchmarked in the paper; it can stagnate at the rounding floor and
# return converged = FALSE.  safeguard = TRUE (the default) additionally
# applies the two rounding-floor safeguards of inv_gft: the full Newton
# step is taken untested once the predicted decrease falls below the
# resolution of f, and a persistent lack of progress hands the iteration
# to the contractive fixed point.
inv_gft_newton <- function(z, x0 = NULL, tol = 1e-13, maxit = 500,
                           warm = 1, safeguard = TRUE) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    eighs <- 0
    best_err <- Inf
    stall <- 0
    fp_finish <- FALSE
    F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors; eighs <- eighs + 1
    for (i in seq_len(warm)) {
        x <- x - .logdiagexp(lam, Q)
        F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
        eighs <- eighs + 1
    }
    err <- Inf
    for (k in 0:(maxit - 1)) {
        ell <- .logdiagexp(lam, Q)
        g <- expm1(ell)
        err <- max(abs(g))
        hist[nh <- nh + 1] <- err
        if (err < tol)
            return(.inv_result(x, .expA(lam, Q), k, eighs, 0, err, TRUE,
                               hist[seq_len(nh)]))
        if (safeguard) {
            # near the rounding floor the Newton direction is computed
            # from noise-dominated gradients and the Armijo test is
            # decided by cancellation in f; if progress stalls there,
            # finish with fixed-point steps, whose update x <- x - ell
            # remains contractive
            if (err < 0.5 * best_err) {
                best_err <- err
                stall <- 0
            } else if (err < 1e-9) {
                stall <- stall + 1
            }
            if (fp_finish || stall >= 3) {
                fp_finish <- TRUE
                x <- x - ell
                F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
                eighs <- eighs + 1
                next
            }
        }
        step <- NULL
        if (all(is.finite(g))) {
            H <- .hessian(lam, Q)
            ch <- tryCatch(chol(H), error = function(e) NULL)
            if (!is.null(ch)) {
                s <- -backsolve(ch, backsolve(ch, g, transpose = TRUE))
                if (all(is.finite(s)))
                    step <- s
            }
        }
        if (is.null(step)) {                  # safeguard: fixed-point step
            x <- x - ell
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
            next
        }
        f0 <- .fval(lam, x)
        gTs <- sum(g * step)
        if (safeguard && -gTs <= 1e-12 * (1 + abs(f0))) {
            # predicted decrease below the float resolution of f: the
            # Armijo test carries no information here, so take the full
            # Newton step untested (we are in the Newton basin)
            x <- x + step
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
            next
        }
        tstep <- 1
        ok <- FALSE
        lam_t <- lam; Q_t <- Q
        while (tstep >= 1e-8) {
            F <- .eigx(A0, x + tstep * step)
            lam_t <- F$values; Q_t <- F$vectors
            eighs <- eighs + 1
            ft <- .fval(lam_t, x + tstep * step)
            if (is.finite(ft) && ft <= f0 + 1e-4 * tstep * gTs) {
                ok <- TRUE
                break
            }
            tstep <- tstep / 2
        }
        if (ok) {
            x <- x + tstep * step
            lam <- lam_t; Q <- Q_t
        } else {
            x <- x - .logdiagexp(lam, Q)
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
        }
    }
    .inv_result(x, .expA(lam, Q), maxit, eighs, 0, err, FALSE,
                hist[seq_len(nh)])
}

# ------------------------------------------- quadrature preconditioner

# Gauss-Legendre nodes and weights on [0, 1] (Golub-Welsch).
.gauss_legendre01 <- function(r) {
    if (r == 1) return(list(t = 0.5, a = 1))
    k <- seq_len(r - 1)
    b <- k / sqrt(4 * k^2 - 1)
    J <- matrix(0, r, r)
    J[cbind(k, k + 1)] <- b
    J[cbind(k + 1, k)] <- b
    F <- eigen(J, symmetric = TRUE)
    o <- order(F$values)
    list(t = (F$values[o] + 1) / 2, a = F$vectors[1, o]^2)
}

# log phi_r(u): the ratio of the divided difference of exp to its r-point
# Gauss-Legendre approximation at eigenvalue gap u,
# phi_r(u) = [sinh(u/2)/(u/2)] / sum_j a_j cosh((t_j - 1/2) u) >= 1.
.logphi <- function(r, u) {
    gl <- .gauss_legendre01(r)
    h <- abs(u) / 2
    ls <- if (h > 1e-8) h + log(-expm1(-2 * h)) - log(2 * h) else 0
    cj <- abs(gl$t - 0.5) * abs(u)
    cs <- log(gl$a) + cj + log1p(exp(-2 * cj)) - log(2)
    m <- max(cs)
    ls - (m + log(sum(exp(cs - m))))
}

# Smallest Gauss-Legendre order r <= rmax with certified condition number
# phi_r(spread) <= kappa for the preconditioned system; 0 if none.
.choose_order <- function(spread, kappa, rmax) {
    for (r in seq_len(rmax))
        if (.logphi(r, spread) <= log(kappa)) return(r)
    0L
}

# Cholesky factor of M_r = sum_j a_j e^{t_j A} o e^{(1-t_j) A} for the
# symmetric matrix with eigenvalues w (shifted so that max(w) = 0) and
# eigenvectors Q, with r the smallest Gauss-Legendre order certifying
# M_r <= H <= kappa M_r from the spectral spread.  Returns list(R, r):
# R = NULL with r = 0 means "use the diagonal preconditioner"; R = NULL
# with r > 0 a failed factorization.  rfixed > 0 prescribes the order.
.quadrature_precond <- function(w, Q, kappa = 2, rmin = 1, rmax = 40,
                                nmin = 0, rfixed = 0) {
    n <- length(w)
    spread <- max(w) - min(w)
    if (rfixed > 0) {
        r <- rfixed
    } else {
        r <- if (spread > 1000) 0L else .choose_order(spread, kappa, rmax)
        if (r == 0 || r < rmin || n < nmin) return(list(R = NULL, r = 0L))
    }
    gl <- .gauss_legendre01(r)
    M <- matrix(0, n, n)
    half <- (r + 1) %/% 2
    for (j in seq_len(half)) {
        k <- r + 1 - j                        # paired node 1 - t_j
        Ej <- Q %*% (exp(gl$t[j] * w) * t(Q))
        if (j == k) {
            M <- M + gl$a[j] * (Ej * Ej)
        } else {
            Ek <- Q %*% (exp(gl$t[k] * w) * t(Q))
            M <- M + (gl$a[j] + gl$a[k]) * (Ej * Ek)
        }
    }
    R <- tryCatch(chol((M + t(M)) / 2), error = function(e) NULL)
    list(R = R, r = as.integer(r))
}

# ---------------------------------------------------------------- GFT-FP+N

# Exact normalization: shift x so that tr exp(A[x]) = n (no
# eigendecomposition; the eigenvectors are unchanged).
.normalize <- function(x, lam, ell) {
    c <- max(lam)
    s <- c + log(sum(exp(lam - c)) / length(lam))
    list(x = x - s, lam = lam - s, ell = ell - s, s = s)
}

# f evaluated from the eigenvalues, scaled to avoid overflow.
.fval <- function(lam, x) {
    c <- max(lam)
    exp(c) * sum(exp(lam - c)) - sum(x)
}

# GFT-FP+N (recommended).  Every evaluated point is normalized by the
# exact minimization of f along the vector of ones.  Fixed-point steps
# x <- x - ell, ell = log diag(exp A), are taken while some ell_i < -700
# (and, with phase = TRUE, while max ell > log(1 + delta)).  Otherwise an
# inexact Newton step for the equation ell(x) = 0 is computed: the
# system H delta = -D ell, D = diag(exp ell), is solved matrix-free by
# conjugate gradients with preconditioner D (or the certified quadrature
# model M_r), exact Hessian-vector products and forcing tolerance
# eta = min(1/2, ||ell||) on the preconditioned residual, capped at 2n
# products.  A non-finite or non-descent direction is replaced by a
# fixed-point step; the full step is taken untested when its predicted
# decrease is below the floating-point resolution of f and
# ||ell||_inf < 1e-3; otherwise Armijo backtracking on f from
# t = min(1, 2 t_prev), with a fixed-point step substituted if the search
# fails.  Near the rounding floor, if ||g||_inf < 1e-9 fails to halve
# three times, the computation is finished by fixed-point steps.
#
# The first-draft version of the algorithm is residual = "gradient",
# forcing = "sqrt", normalize = FALSE, phase = TRUE, adaptive = FALSE.
# exact_hess = TRUE solves the same Newton system with the explicit
# O(n^4) Hessian instead of conjugate gradients.
inv_gft <- function(z, x0 = NULL, tol = 1e-13, maxit = 500, delta = 1,
                    exact_hess = FALSE, residual = c("log", "gradient"),
                    forcing = c("quad", "sqrt"), normalize = TRUE,
                    phase = FALSE, adaptive = TRUE,
                    preconditioner = c("diagonal", "quadrature", "auto"),
                    kappa = 2, rmin = 2, rmax = 8, nmin = 0) {
    .inv_gft(z, x0, tol, maxit, delta, exact_hess, match.arg(residual),
             match.arg(forcing), normalize, phase, adaptive,
             match.arg(preconditioner), kappa, rmin, rmax, nmin)$result
}

.inv_gft <- function(z, x0, tol, maxit, delta, exact_hess, residual,
                     forcing, normalize, phase, adaptive, preconditioner,
                     kappa, rmin, rmax, nmin) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    eighs <- 0; hvs <- 0
    builds <- 0; nodes <- 0; cholfail <- 0
    best_err <- Inf; stall <- 0; fp_finish <- FALSE; tprev <- 1
    F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors; eighs <- eighs + 1
    ell <- .logdiagexp(lam, Q)
    if (normalize) { nz <- .normalize(x, lam, ell); x <- nz$x; lam <- nz$lam; ell <- nz$ell }
    # one eigendecomposition at a point, normalized if requested
    ev <- function(x) {
        F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
        ell <- .logdiagexp(lam, Q)
        if (normalize) { nz <- .normalize(x, lam, ell); x <- nz$x; lam <- nz$lam; ell <- nz$ell }
        list(x = x, lam = lam, Q = Q, ell = ell)
    }
    stats <- function() list(builds = builds, nodes = nodes, cholfail = cholfail)
    err <- Inf
    for (k in 0:(maxit - 1)) {
        if ((phase && max(ell) > log(1 + delta)) || min(ell) < -700) {
            hist[nh <- nh + 1] <- expm1(min(max(ell), 700))     # fixed-point step
            e <- ev(x - ell); x <- e$x; lam <- e$lam; Q <- e$Q; ell <- e$ell
            eighs <- eighs + 1; tprev <- 1
            next
        }
        g <- expm1(ell)
        err <- max(abs(g))
        hist[nh <- nh + 1] <- err
        if (err < tol)
            return(list(result = .inv_result(x, .expA(lam, Q), k, eighs, hvs, err,
                                             TRUE, hist[seq_len(nh)]),
                        lam = lam, Q = Q, stats = stats()))
        # terminal mode near the rounding floor: fixed-point steps
        if (err < 0.5 * best_err) { best_err <- err; stall <- 0 }
        else if (err < 1e-9) stall <- stall + 1
        if (fp_finish || stall >= 3) {
            fp_finish <- TRUE
            e <- ev(x - ell); x <- e$x; lam <- e$lam; Q <- e$Q; ell <- e$ell
            eighs <- eighs + 1
            next
        }
        # ---- Newton step, all quantities scaled by e^{-c}, c = max(lam)
        c <- max(lam)
        dEs <- exp(ell - c)                   # e^{-c} diag(exp A)
        Ls <- .loewner_exp(lam - c)           # e^{-c} L
        if (residual == "log") {
            rhs <- -dEs * ell                 # e^{-c} (-D ell)
            nres <- sqrt(sum(ell^2))
            eta <- min(0.5, if (forcing == "quad") nres else sqrt(nres))
            stopfun <- function(r) sqrt(sum((r / dEs)^2)) <= eta * nres
        } else {
            rhs <- -exp(-c) * g               # e^{-c} (-g)
            nres <- sqrt(sum(g^2))
            eta <- min(0.5, if (forcing == "quad") nres else sqrt(nres))
            thr <- eta * sqrt(sum(rhs^2))
            stopfun <- function(r) sqrt(sum(r^2)) <= thr
        }
        if (exact_hess) {
            step <- tryCatch({
                ch <- chol(.hessian(lam, Q, Ls))
                backsolve(ch, backsolve(ch, rhs, transpose = TRUE))
            }, error = function(e) rep(NaN, n))
        } else {                              # preconditioned CG from 0
            R <- NULL
            if (preconditioner != "diagonal") {
                qp <- .quadrature_precond(lam - c, Q, kappa = kappa,
                          rmin = if (preconditioner == "auto") rmin else 1,
                          rmax = if (preconditioner == "auto") rmax else 40,
                          nmin = if (preconditioner == "auto") nmin else 0)
                R <- qp$R
                if (qp$r > 0) {
                    builds <- builds + 1; nodes <- nodes + qp$r
                    if (is.null(R)) cholfail <- cholfail + 1
                }
            }
            applyM <- if (is.null(R)) function(r) r / dEs else
                function(r) backsolve(R, backsolve(R, r, transpose = TRUE))
            step <- numeric(n)
            r <- rhs
            pcg <- applyM(r)
            rz <- sum(r * pcg)
            for (it in seq_len(2 * n)) {
                Hp <- .hess_vec(lam, Q, Ls, pcg); hvs <- hvs + 1
                a <- rz / sum(pcg * Hp)
                step <- step + a * pcg
                r <- r - a * Hp
                if (stopfun(r)) break
                zz <- applyM(r)
                rz_new <- sum(r * zz)
                pcg <- zz + (rz_new / rz) * pcg
                rz <- rz_new
            }
        }
        f0 <- .fval(lam, x)
        gTs <- sum(g * step)
        # ---- acceptance: finiteness and descent first
        if (!all(is.finite(step)) || !(gTs < 0) || !is.finite(f0)) {
            e <- ev(x - ell); x <- e$x; lam <- e$lam; Q <- e$Q; ell <- e$ell
            eighs <- eighs + 1; tprev <- 1
            next
        }
        if (-gTs <= 1e-12 * (1 + abs(f0)) && max(abs(ell)) < 1e-3) {
            # predicted decrease below the float resolution of f, near the
            # solution: full step, untested
            e <- ev(x + step); x <- e$x; lam <- e$lam; Q <- e$Q; ell <- e$ell
            eighs <- eighs + 1
            next
        }
        tstep <- if (adaptive) min(1, 2 * tprev) else 1
        ok <- FALSE
        while (tstep >= 1e-8) {               # Armijo backtracking on f
            e <- ev(x + tstep * step)
            eighs <- eighs + 1
            ft <- .fval(e$lam, e$x)
            if (is.finite(ft) && ft <= f0 + 1e-4 * tstep * gTs) { ok <- TRUE; break }
            tstep <- tstep / 2
        }
        if (ok) {
            x <- e$x; lam <- e$lam; Q <- e$Q; ell <- e$ell
            tprev <- tstep
        } else {                              # safeguard: fixed-point step
            e <- ev(x - ell); x <- e$x; lam <- e$lam; Q <- e$Q; ell <- e$ell
            eighs <- eighs + 1; tprev <- 1
        }
    }
    list(result = .inv_result(x, .expA(lam, Q), maxit, eighs, hvs, err, FALSE,
                              hist[seq_len(nh)]),
         lam = lam, Q = Q, stats = stats())
}

# ------------------------------------------------ predictor and sequences

# Tangent predictor for sequential inversion: given the solution x_prev
# of z_prev with its eigendecomposition (lam, Q) of A[x_prev; z_prev],
# the first-order prediction x_prev + p of the solution for z_new, with
# H p = -(B dz + D ell) solved by preconditioned conjugate gradients to
# relative residual rtol (no eigendecomposition).  Returns list(x, hvs).
gft_predict <- function(z_prev, x_prev, lam, Q, z_new, rtol = 1e-3) {
    n <- length(x_prev)
    dA <- unvecl(as.numeric(z_new) - as.numeric(z_prev))
    c <- max(lam)
    Ls <- .loewner_exp(lam - c)
    ell <- .logdiagexp(lam, Q)
    dEs <- exp(ell - c)
    M <- crossprod(Q, dA %*% Q)               # B dz (scaled): three mults
    bdz <- rowSums((Q %*% (Ls * M)) * Q)
    rhs <- -(bdz + dEs * ell)
    nr <- sqrt(sum(rhs^2))
    p <- numeric(n); r <- rhs; pp <- r / dEs; rz <- sum(r * pp); hvs <- 0
    for (it in seq_len(2 * n)) {
        if (sqrt(sum(r^2)) <= rtol * nr) break
        Hp <- .hess_vec(lam, Q, Ls, pp); hvs <- hvs + 1
        a <- rz / sum(pp * Hp)
        p <- p + a * pp
        r <- r - a * Hp
        rz_new <- sum(r * (r / dEs))
        pp <- r / dEs + (rz_new / rz) * pp
        rz <- rz_new
    }
    list(x = x_prev + p, hvs = hvs)
}

# Sequential inversion of a list of z's with warm starts: each solve
# starts from the previous solution or, with predictor = TRUE, from the
# tangent prediction.  Returns a list of gft_inv results (hvs including
# the predictor's products) with attribute "nbdz", the number of
# predictor evaluations.
inv_gft_path <- function(zs, predictor = TRUE, rtol = 1e-3, tol = 1e-13, ...) {
    out <- vector("list", length(zs))
    x0 <- NULL; lam <- NULL; Q <- NULL; nbdz <- 0
    for (t in seq_along(zs)) {
        hv_pred <- 0
        if (t > 1 && predictor) {
            pr <- gft_predict(zs[[t - 1]], x0, lam, Q, zs[[t]], rtol = rtol)
            x0 <- pr$x; hv_pred <- pr$hvs; nbdz <- nbdz + 1
        }
        args <- list(z = zs[[t]], x0 = x0, tol = tol, ...)
        full <- do.call(.inv_gft_args, args)
        r <- full$result
        r$hvs <- r$hvs + hv_pred
        out[[t]] <- r
        x0 <- r$x; lam <- full$lam; Q <- full$Q
    }
    attr(out, "nbdz") <- nbdz
    out
}

# argument-matching wrapper around .inv_gft with inv_gft's defaults
.inv_gft_args <- function(z, x0 = NULL, tol = 1e-13, maxit = 500, delta = 1,
                          exact_hess = FALSE, residual = "log",
                          forcing = "quad", normalize = TRUE, phase = FALSE,
                          adaptive = TRUE, preconditioner = "diagonal",
                          kappa = 2, rmin = 2, rmax = 8, nmin = 0) {
    .inv_gft(z, x0, tol, maxit, delta, exact_hess, residual, forcing,
             normalize, phase, adaptive, preconditioner, kappa, rmin, rmax,
             nmin)
}

# --------------------------------------------- Anderson acceleration

# V(x) = sum_i (e^{ell_i} - 1 - ell_i), cancellation-free.
.vfun <- function(ell) {
    small <- abs(ell) < 1e-3
    t <- ell
    v <- ifelse(small, t * t * (0.5 + t * (1 / 6 + t * (1 / 24 + t / 120))),
                expm1(t) - t)
    sum(v)
}

# minimum-norm least-squares solution, robust to rank deficiency
.lstsq <- function(A, b) {
    sv <- svd(A)
    tol <- .Machine$double.eps * max(dim(A)) * sv$d[1]
    sinv <- ifelse(sv$d > tol, 1 / sv$d, 0)
    sv$v %*% (sinv * crossprod(sv$u, b))
}

# Anderson acceleration (type II, memory m) of the fixed-point map
# g(x) = x - ell(x), residual r(x) = -ell(x) in the log domain.  With
# guarded = TRUE a proposal y is accepted only if
# f(y) <= f(x) - theta V(x), V(x) = sum_i (e^{ell_i} - 1 - ell_i), the
# lower bound on the decrease of the fixed-point step; otherwise the
# memory is cleared and the fixed-point step is taken.  Proposals are
# accepted untested once theta V(x) <= 1e-12 (1 + |f(x)|).
inv_gft_anderson <- function(z, x0 = NULL, tol = 1e-13, maxit = 5000, m = 5,
                             guarded = FALSE, theta = 0.25) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    X <- list(); R <- list()
    F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors; eighs <- 1
    ell <- .logdiagexp(lam, Q)
    err <- Inf
    for (k in 0:(maxit - 1)) {
        err <- max(abs(expm1(ell)))
        hist[nh <- nh + 1] <- err
        if (err < tol)
            return(.inv_result(x, .expA(lam, Q), k, eighs, 0, err, TRUE,
                               hist[seq_len(nh)]))
        r <- -ell
        X[[length(X) + 1]] <- x; R[[length(R) + 1]] <- r
        if (length(X) > m + 1) { X <- X[-1]; R <- R[-1] }
        mk <- length(X) - 1
        plain <- mk == 0
        if (plain) {
            y <- x + r
        } else {
            dR <- sapply(seq_len(mk), function(i) R[[i + 1]] - R[[i]])
            dX <- sapply(seq_len(mk), function(i) X[[i + 1]] - X[[i]])
            dR <- matrix(dR, n, mk); dX <- matrix(dX, n, mk)
            gam <- .lstsq(dR, r)
            y <- x + r - (dX + dR) %*% gam
            y <- as.numeric(y)
        }
        if (!all(is.finite(y)))
            return(.inv_result(x, .expA(lam, Q), k + 1, eighs, 0, Inf, FALSE,
                               hist[seq_len(nh)]))
        F <- .eigx(A0, y); lam_y <- F$values; Q_y <- F$vectors; eighs <- eighs + 1
        ell_y <- .logdiagexp(lam_y, Q_y)
        if (guarded && !plain) {
            f0 <- .fval(lam, x)
            V0 <- .vfun(ell)
            if (theta * V0 > 1e-12 * (1 + abs(f0))) {
                fy <- .fval(lam_y, y)
                if (!(is.finite(fy) && fy <= f0 - theta * V0)) {
                    X <- list(); R <- list()   # reject: fixed-point step
                    y <- x + r
                    F <- .eigx(A0, y); lam_y <- F$values; Q_y <- F$vectors
                    eighs <- eighs + 1
                    ell_y <- .logdiagexp(lam_y, Q_y)
                }
            }
        }
        x <- y; lam <- lam_y; Q <- Q_y; ell <- ell_y
        if (!all(is.finite(ell)))
            return(.inv_result(x, .expA(lam, Q), k + 1, eighs, 0, Inf, FALSE,
                               hist[seq_len(nh)]))
    }
    .inv_result(x, .expA(lam, Q), maxit, eighs, 0, err, FALSE, hist[seq_len(nh)])
}

# ------------------------------------------------------------- L-BFGS

# Limited-memory BFGS on f(x) = tr(exp(A[x])) - sum(x) with gradient
# g = diag(exp(A[x])) - 1, two-loop recursion with memory m, scaling
# (s'y)/(y'y), and Armijo backtracking on f (constant 1e-4, halving over
# eight decades; a step whose predicted decrease is below roundoff in f
# is accepted untested, as in GFT-FP+N).  The first step after a start
# or reset has unit sup-norm length.  Started at x = 0 or, with
# globalized = TRUE, after a log-domain fixed-point phase until
# max_i ell_i <= log 2.  A comparator.
inv_gft_lbfgs <- function(z, x0 = NULL, tol = 1e-13, maxit = 500, m = 10,
                          globalized = FALSE) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    eighs <- 0
    F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors; eighs <- eighs + 1
    if (globalized) {
        while (max(.logdiagexp(lam, Q)) > log(2)) {
            x <- x - .logdiagexp(lam, Q)
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors; eighs <- eighs + 1
        }
    }
    S <- list(); Y <- list()
    ell <- .logdiagexp(lam, Q)
    g <- expm1(ell)
    f0 <- .fval(lam, x)
    err <- max(abs(g))
    for (k in 0:(maxit - 1)) {
        hist[nh <- nh + 1] <- err
        if (err < tol)
            return(.inv_result(x, .expA(lam, Q), k, eighs, 0, err, TRUE,
                               hist[seq_len(nh)]))
        if (!all(is.finite(g)) || !is.finite(f0))
            return(.inv_result(x, .expA(lam, Q), k, eighs, 0, Inf, FALSE,
                               hist[seq_len(nh)]))
        q <- g
        L <- length(S)
        alpha <- numeric(L)
        for (i in rev(seq_len(L))) {
            rho_i <- 1 / sum(Y[[i]] * S[[i]])
            alpha[i] <- rho_i * sum(S[[i]] * q)
            q <- q - alpha[i] * Y[[i]]
        }
        if (L > 0) q <- q * (sum(S[[L]] * Y[[L]]) / sum(Y[[L]] * Y[[L]]))
        for (i in seq_len(L)) {
            rho_i <- 1 / sum(Y[[i]] * S[[i]])
            beta <- rho_i * sum(Y[[i]] * q)
            q <- q + (alpha[i] - beta) * S[[i]]
        }
        d <- -q
        gTd <- sum(g * d)
        if (gTd >= 0) { S <- list(); Y <- list(); d <- -g; gTd <- -sum(g * g) }
        tstep <- if (length(S) == 0) min(1, 1 / max(abs(d))) else 1
        ok <- FALSE
        tmin <- 1e-8 * tstep
        while (tstep >= tmin) {
            F <- .eigx(A0, x + tstep * d); lam_t <- F$values; Q_t <- F$vectors
            eighs <- eighs + 1
            ft <- .fval(lam_t, x + tstep * d)
            if (is.finite(ft) && (ft <= f0 + 1e-4 * tstep * gTd ||
                                  abs(tstep * gTd) <= 1e-12 * (1 + abs(f0)))) {
                ok <- TRUE; break
            }
            tstep <- tstep / 2
        }
        if (!ok)
            return(.inv_result(x, .expA(lam, Q), k + 1, eighs, 0, err, FALSE,
                               hist[seq_len(nh)]))
        xn <- x + tstep * d
        ell_n <- .logdiagexp(lam_t, Q_t)
        gn <- expm1(ell_n)
        s <- xn - x; y <- gn - g
        if (sum(s * y) > 1e-12 * sum(y * y)) {
            S[[length(S) + 1]] <- s; Y[[length(Y) + 1]] <- y
            if (length(S) > m) { S <- S[-1]; Y <- Y[-1] }
        }
        x <- xn; lam <- lam_t; Q <- Q_t; g <- gn; f0 <- ft
        err <- max(abs(g))
    }
    .inv_result(x, .expA(lam, Q), maxit, eighs, 0, err, FALSE, hist[seq_len(nh)])
}

# ---------------------------------------------------------------- methods

print.gft_inv <- function(x, ...) {
    n <- length(x$x)
    cat("Inverse GFT result (n = ", n, ")\n", sep = "")
    cat("  converged: ", x$converged, "  (err = ",
        formatC(x$err, format = "e", digits = 2), ")\n", sep = "")
    cat("  iterations: ", x$iters, "   eigendecompositions: ", x$eighs,
        if (x$hvs > 0) paste0("   Hessian-vector products: ", x$hvs)
        else "", "\n", sep = "")
    invisible(x)
}
