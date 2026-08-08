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
#   inv_gft         -- GFT-FP+N: fixed-point phase in the log domain, then
#                      matrix-free inexact Newton via preconditioned
#                      conjugate gradients (recommended)
#   inv_gft_fp      -- Archakov-Hansen fixed point (log domain)
#   inv_gft_broyden -- Broyden's method as in Chen, Fei and Yu (2025)
#   inv_gft_newton  -- full Newton with exact O(n^4) Hessian
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

.fval <- function(lam, x) sum(exp(lam)) - sum(x)

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
inv_gft_newton <- function(z, x0 = NULL, tol = 1e-13, maxit = 500,
                           warm = 1) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    eighs <- 0
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

# GFT-FP+N (recommended).  Phase 1: fixed-point steps in the log domain
# while ||diag(exp(A))-1||_inf > delta (robust to overflow).  Phase 2:
# inexact Newton; the system H step = -g is solved matrix-free by
# conjugate gradients with Jacobi preconditioner diag(exp(A)), exact
# Hessian-vector products (two matrix multiplications each) and forcing
# tolerance eta = min(1/2, sqrt(||g||)) (Eisenstat and Walker, 1996;
# superlinear of order 3/2).  Armijo backtracking on f, with the full
# Newton step taken untested once the predicted decrease is below the
# floating-point resolution of f, and a fixed-point step substituted if
# the line search fails.
inv_gft <- function(z, x0 = NULL, tol = 1e-13, maxit = 500, delta = 1,
                    exact_hess = FALSE) {
    p <- .prep(z, x0)
    A0 <- p$A0; n <- p$n; x <- p$x
    hist <- numeric(maxit + 1); nh <- 0
    eighs <- 0
    hvs <- 0
    best_err <- Inf
    stall <- 0
    fp_finish <- FALSE
    F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors; eighs <- eighs + 1
    err <- Inf
    for (k in 0:(maxit - 1)) {
        ell <- .logdiagexp(lam, Q)            # log domain: no overflow
        if (max(ell) > log(1 + delta) || min(ell) < -700) {
            # ---- phase 1: fixed point.  Equivalent to
            # ||diag(exp A) - 1||_inf > delta, tested without
            # exponentiating (g is never formed while it could overflow);
            # the second test keeps the fixed point in charge while any
            # diagonal of exp(A) would underflow (exp(ell) == 0 in double)
            hist[nh <- nh + 1] <- expm1(min(max(ell), 700))
            x <- x - ell
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
            next
        }
        g <- expm1(ell)                       # finite here: |g| <= delta
        err <- max(abs(g))
        hist[nh <- nh + 1] <- err
        if (err < tol)
            return(.inv_result(x, .expA(lam, Q), k, eighs, hvs, err, TRUE,
                               hist[seq_len(nh)]))
        # near the rounding floor the Newton direction is computed from
        # noise-dominated gradients; if progress stalls there, finish with
        # fixed-point steps, whose update x <- x - ell remains contractive
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
        dE <- exp(ell)                        # = diag(exp A) directly; the
        # algebraically equal 1 + expm1(ell) cancels to 0 for ell <= -37
        # ---- phase 2: Newton
        L <- .loewner_exp(lam)
        if (exact_hess) {                     # explicit O(n^4) Hessian:
            # identical algorithm, only the linear solver differs
            ch <- chol(.hessian(lam, Q, L))
            step <- -backsolve(ch, backsolve(ch, g, transpose = TRUE))
        } else {                              # matrix-free preconditioned CG
            eta <- min(0.5, sqrt(sqrt(sum(g * g))))  # min(1/2, ||g||^{1/2})
            step <- numeric(n)
            r <- -g
            pcg <- r / dE
            rz <- sum(r * pcg)
            normg <- sqrt(sum(g * g))
            for (it in seq_len(2 * n)) {
                Hp <- .hess_vec(lam, Q, L, pcg); hvs <- hvs + 1
                a <- rz / sum(pcg * Hp)
                step <- step + a * pcg
                r <- r - a * Hp
                if (sqrt(sum(r * r)) <= eta * normg)
                    break
                rz_new <- sum(r * (r / dE))
                pcg <- r / dE + (rz_new / rz) * pcg
                rz <- rz_new
            }
        }
        f0 <- .fval(lam, x)
        gTs <- sum(g * step)
        if (-gTs <= 1e-12 * (1 + abs(f0))) {
            # predicted decrease below float resolution of f: Newton basin
            x <- x + step
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
            next
        }
        tstep <- 1
        ok <- FALSE
        lam_t <- lam; Q_t <- Q
        while (tstep >= 1e-8) {               # Armijo backtracking
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
        } else {                              # safeguard: fixed-point step
            x <- x - ell
            F <- .eigx(A0, x); lam <- F$values; Q <- F$vectors
            eighs <- eighs + 1
        }
    }
    .inv_result(x, .expA(lam, Q), maxit, eighs, hvs, err, FALSE,
                hist[seq_len(nh)])
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
