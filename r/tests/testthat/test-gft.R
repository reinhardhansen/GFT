# Test suite for the GFT package.
#
# The golden values were generated from the independently verified Python
# implementation (NumPy/SciPy, double precision) and are shared with the
# Julia test suite, so a passing run constitutes a cross-language
# verification of the solver.

test_that("golden values from Python implementation", {
    # Case 1: Toeplitz rho = 0.9, n = 5
    z1 <- c(1.153290016506162, 0.5558856623939625, 0.35800387580539283,
            0.27329448930908706, 1.0109967961569906, 0.5069766634781976,
            0.358003875805393, 1.0109967961569903, 0.5558856623939642,
            1.1532900165061641)
    x1 <- c(-1.0295579541972146, -1.511633306658801, -1.5605423055745682,
            -1.5116333066588024, -1.0295579541972182)
    r <- inv_gft(z1, tol = 1e-15)
    expect_true(r$converged)
    expect_lt(max(abs(r$x - x1)), 1e-11)
    # forward map round-trip
    C <- 0.9^abs(outer(1:5, 1:5, "-"))
    expect_lt(max(abs(gft(C) - z1)), 1e-12)
    expect_lt(max(abs(r$C - C)), 1e-11)

    # Case 2: equicorrelation rho = 0.99, n = 4 (repeated eigenvalues)
    z2 <- c(1.4959840701717646, 1.4959840701718055, 1.4959840701718055,
            1.4959840701717937, 1.4959840701717941, 1.495984070171792)
    x2 <- c(-3.1091861158162764, -3.1091861158162533, -3.10918611581629,
            -3.109186115816293)
    r <- inv_gft(z2, tol = 1e-15)
    expect_true(r$converged)
    expect_lt(max(abs(r$x - x2)), 1e-10)

    # Case 3: fixed z, n = 3, with the reconstructed C
    z3 <- c(0.5, -1.25, 2.0)
    x3 <- c(-0.4468779844556956, -1.1593398688810534, -1.7913253190785425)
    C3 <- matrix(c(1.0, -0.19502680396870392, -0.6143366326910233,
                   -0.19502680396870378, 1.000000000000004,
                   0.8717792808904407,
                   -0.6143366326910232, 0.871779280890441,
                   1.0000000000000056), 3, 3)
    r <- inv_gft(z3, tol = 1e-15)
    expect_lt(max(abs(r$x - x3)), 1e-11)
    expect_lt(max(abs(r$C - C3)), 1e-11)
})

test_that("round trips, all solvers", {
    set.seed(1)
    for (n in c(10, 50)) {
        X <- matrix(rnorm(n * 2 * n), n, 2 * n)
        S <- X %*% t(X)
        Dh <- 1 / sqrt(diag(S))
        C <- S * outer(Dh, Dh)
        C <- (C + t(C)) / 2
        z <- gft(C)
        for (solver in list(inv_gft, inv_gft_fp, inv_gft_broyden,
                            inv_gft_newton)) {
            r <- solver(z)
            expect_true(r$converged)
            expect_lt(max(abs(diag(r$C) - 1)), 1e-12)
            expect_lt(max(abs(r$C - C)), 1e-10)
        }
    }
})

test_that("extreme z (log-domain robustness)", {
    set.seed(2)
    z <- 8 * rnorm(40 * 39 / 2)          # spectra spanning ~200 log-units
    r <- inv_gft(z)
    expect_true(r$converged)
    expect_lt(max(abs(diag(r$C) - 1)), 1e-12)
    rfp <- inv_gft_fp(z, maxit = 20000)
    expect_true(rfp$converged)
    expect_lt(max(abs(r$x - rfp$x)), 1e-8)
})

test_that("near-singular one-factor", {
    set.seed(3)
    n <- 100
    b <- 0.995 + 0.0049 * runif(n)
    C <- outer(b, b)
    diag(C) <- 1
    z <- gft(C)
    r <- inv_gft(z)
    expect_true(r$converged)
    expect_lt(r$eighs, 30)
    expect_lt(max(abs(r$C - C)), 1e-9)
})

test_that("gradient, Hessian-vector products, exact Hessian", {
    set.seed(4)
    n <- 12
    A0 <- matrix(rnorm(n * n), n, n)
    A0 <- (A0 + t(A0)) / 2
    diag(A0) <- 0
    x <- rnorm(n)
    A <- A0
    diag(A) <- x
    F <- eigen(A, symmetric = TRUE)
    lam <- F$values
    Q <- F$vectors
    f <- function(x) {
        A <- A0
        diag(A) <- x
        sum(exp(eigen(A, symmetric = TRUE, only.values = TRUE)$values)) -
            sum(x)
    }
    g <- exp(GFT:::.logdiagexp(lam, Q)) - 1
    eps <- 1e-6
    for (i in seq_len(n)) {
        e <- numeric(n)
        e[i] <- eps
        expect_lt(abs((f(x + e) - f(x - e)) / (2 * eps) - g[i]), 1e-6)
    }
    L <- GFT:::.loewner_exp(lam)
    H <- GFT:::.hessian(lam, Q, L)
    v <- rnorm(n)
    expect_lt(max(abs(GFT:::.hess_vec(lam, Q, L, v) - H %*% v)), 1e-10)
    # Proposition 1: H*1 = d, lmin(E) I <= H <= D
    E <- GFT:::.expA(lam, Q)
    d <- diag(E)
    expect_lt(max(abs(H %*% rep(1, n) - d)), 1e-8 * max(d))
    eigmin <- function(M)
        min(eigen((M + t(M)) / 2, symmetric = TRUE,
                  only.values = TRUE)$values)
    expect_gt(eigmin(diag(d) - H), -1e-8 * max(d))
    expect_gt(eigmin(H - exp(min(lam)) * diag(n)), -1e-10)
})

test_that("repeated eigenvalues in divided differences", {
    lam <- c(1.0, 1.0, 2.0, 2.0 + 1e-15)
    L <- GFT:::.loewner_exp(lam)
    expect_equal(L[1, 2], exp(1))
    expect_equal(L[3, 4], exp(2), tolerance = 1e-12)
    expect_true(isSymmetric(round(L, 10)))
})

test_that("vecl and unvecl", {
    M <- matrix(1:16, 4, 4)
    z <- vecl(M)
    expect_equal(z, c(2, 3, 4, 7, 8, 12))    # column-major below-diagonal
    A <- unvecl(z)
    expect_equal(vecl(A), z)
    expect_equal(A, t(A))
    expect_equal(diag(A), numeric(4))
    expect_error(unvecl(1:2), "n\\(n-1\\)/2")
})

test_that("input validation", {
    expect_error(gft(matrix(1, 2, 3)), "square")
    expect_error(gft(diag(c(1, -1))), "positive definite")
    expect_error(inv_gft(c(0.1, 0.2), x0 = 1), "n\\(n-1\\)/2")
    expect_error(inv_gft(c(0.1, 0.2, 0.3), x0 = 1), "x0")
})

test_that("warm start and print method", {
    C <- 0.5^abs(outer(1:6, 1:6, "-"))
    z <- gft(C)
    r0 <- inv_gft(z)
    r1 <- inv_gft(z, x0 = r0$x)
    expect_true(r1$converged)
    expect_lte(r1$eighs, r0$eighs)
    expect_output(print(r0), "Inverse GFT result")
    expect_output(print(r0), "converged")
})

# ---- 1.2.0: revised Newton step, quadrature preconditioner, comparators

test_that("normalization and shift identities", {
    set.seed(5)
    n <- 12
    z <- rnorm(n * (n - 1) / 2)
    x <- rnorm(n)
    A <- unvecl(z); diag(A) <- x
    e <- eigen(A, symmetric = TRUE)
    ell <- GFT:::.logdiagexp(e$values, e$vectors)
    nz <- GFT:::.normalize(x, e$values, ell)
    # after the shift tr exp(A) = n, eigenvectors unchanged, ell shifted
    expect_lt(abs(sum(exp(nz$lam)) - n), 1e-10 * n)
    An <- A; diag(An) <- nz$x
    en <- eigen(An, symmetric = TRUE)
    expect_lt(max(abs(en$values - nz$lam)), 1e-10)
    expect_lt(max(abs(GFT:::.logdiagexp(en$values, en$vectors) - nz$ell)), 1e-10)
    # shifting x along 1 by s scales exp(A) by exp(-s)
    E0 <- GFT:::.expA(e$values, e$vectors)
    expect_lt(max(abs(GFT:::.expA(en$values, en$vectors) - exp(-nz$s) * E0)),
              1e-10 * max(abs(E0)))
})

test_that("algorithm variants agree", {
    set.seed(6)
    n <- 20
    z <- 2 * rnorm(n * (n - 1) / 2)
    ref <- inv_gft(z)
    expect_true(ref$converged)
    variants <- list(
        list(residual = "gradient", forcing = "sqrt", normalize = FALSE,
             phase = TRUE, adaptive = FALSE),           # 1.0.x algorithm
        list(residual = "gradient", forcing = "quad"),
        list(residual = "log", forcing = "sqrt"),
        list(normalize = FALSE),
        list(phase = TRUE),
        list(adaptive = FALSE),
        list(exact_hess = TRUE),
        list(preconditioner = "quadrature"),
        list(preconditioner = "auto"))
    for (kw in variants) {
        r <- do.call(inv_gft, c(list(z = z), kw))
        expect_true(r$converged)
        expect_lt(max(abs(r$x - ref$x)), 1e-10)
    }
})

test_that("quadrature preconditioner: certificate and rule", {
    set.seed(7)
    n <- 40
    b <- 0.8 + 0.195 * runif(n)
    C <- outer(b, b); diag(C) <- 1
    z <- gft(C)
    r <- inv_gft(z)
    A <- unvecl(z); diag(A) <- r$x
    e <- eigen(A, symmetric = TRUE)
    w <- e$values - max(e$values); Q <- e$vectors   # scaled by exp(-max)
    Delta <- max(w) - min(w)
    H <- GFT:::.hessian(w, Q)
    for (rr in 1:4) {
        qp <- GFT:::.quadrature_precond(w, Q, rfixed = rr)
        expect_false(is.null(qp$R))
        M <- crossprod(qp$R)
        ev <- Re(eigen(solve(M, H), only.values = TRUE)$values)
        expect_gte(min(ev), 1 - 1e-6)                            # M_r <= H
        expect_lte(max(ev), exp(GFT:::.logphi(rr, Delta)) + 1e-6) # H <= phi M_r
    }
    # the rule chooses the smallest order that certifies kappa <= 2
    r1 <- GFT:::.choose_order(Delta, kappa = 2, rmax = 40)
    expect_true(r1 >= 1)
    expect_lte(exp(GFT:::.logphi(r1, Delta)), 2 + 1e-12)
    if (r1 > 1) expect_gt(exp(GFT:::.logphi(r1 - 1, Delta)), 2)
    # the quadrature solve uses no more products than the diagonal one
    rq <- inv_gft(z, preconditioner = "quadrature")
    expect_true(rq$converged)
    expect_lte(rq$hvs, r$hvs)
})

test_that("Anderson acceleration, plain and guarded", {
    set.seed(8)
    z <- 2 * rnorm(45)                         # n = 10
    ref <- inv_gft(z)
    for (g in c(FALSE, TRUE)) {
        r <- inv_gft_anderson(z, guarded = g)
        expect_true(r$converged)
        expect_lt(max(abs(r$x - ref$x)), 1e-9)
        expect_lt(r$eighs, inv_gft_fp(z)$eighs)
    }
})

test_that("L-BFGS", {
    set.seed(9)
    z <- 2 * rnorm(45)
    ref <- inv_gft(z)
    for (g in c(FALSE, TRUE)) {
        r <- inv_gft_lbfgs(z, globalized = g)
        expect_true(r$converged)
        expect_lt(max(abs(r$x - ref$x)), 1e-9)
    }
})

test_that("tangent predictor and inv_gft_path", {
    set.seed(10)
    n <- 30
    b <- 0.85 + 0.149 * runif(n)
    C <- outer(b, b); diag(C) <- 1
    z0 <- gft(C)
    d <- length(z0)
    # prediction error is O(h^2): quartering the step cuts it by ~16
    dz <- rnorm(d) / sqrt(d)
    r0 <- GFT:::.inv_gft_args(z0)
    err <- sapply(c(0.04, 0.01), function(h) {
        z1 <- z0 + h * dz
        p <- gft_predict(z0, r0$result$x, r0$lam, r0$Q, z1, rtol = 1e-10)
        max(abs(p$x - inv_gft(z1)$x))
    })
    expect_gt(err[1] / err[2], 8)
    # a path: every solve converges, predictor starts no worse than warm
    zs <- lapply(0:6, function(k) z0 + 0.01 * k * dz)
    outp <- inv_gft_path(zs)
    outw <- inv_gft_path(zs, predictor = FALSE)
    expect_true(all(sapply(outp, `[[`, "converged")))
    expect_true(all(sapply(outw, `[[`, "converged")))
    expect_equal(attr(outp, "nbdz"), length(zs) - 1)
    expect_lte(sum(sapply(outp, `[[`, "eighs")), sum(sapply(outw, `[[`, "eighs")))
    for (k in seq_along(zs))
        expect_lt(max(abs(outp[[k]]$x - inv_gft(zs[[k]])$x)), 1e-10)
})
