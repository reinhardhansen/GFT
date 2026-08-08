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
