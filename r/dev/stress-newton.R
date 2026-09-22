# Regression sweep for the inv_gft_newton rounding-floor stall fixed in
# 1.0.1.  Before the fix this reports a few percent of failures at n = 10
# with max eighs in the thousands; after the fix it should report 0
# failures and max eighs in the single digits.
#
# Run:  Rscript dev/stress-newton.R

library(GFT)

N <- 200
cat(sprintf("GFT %s -- inv_gft_newton stress sweep, %d draws per n\n\n",
            as.character(utils::packageVersion("GFT")), N))

for (n in c(10, 50)) {
    bad <- 0L
    mx <- 0L
    worst <- 0
    t0 <- proc.time()[["elapsed"]]
    set.seed(20260822)
    for (rep in seq_len(N)) {
        X <- matrix(rnorm(n * 2 * n), n, 2 * n)
        S <- X %*% t(X)
        Dh <- 1 / sqrt(diag(S))
        C <- S * outer(Dh, Dh)
        C <- (C + t(C)) / 2
        z <- gft(C)
        r <- inv_gft_newton(z)
        if (!r$converged) {
            bad <- bad + 1L
            worst <- max(worst, r$err)
        }
        mx <- max(mx, r$eighs)
    }
    cat(sprintf("n = %2d   not converged %3d/%d   max eighs %6d   %s%.1fs\n",
                n, bad, N, mx,
                if (bad > 0) sprintf("worst stalled err %.2e   ", worst) else "",
                proc.time()[["elapsed"]] - t0))
}

cat("\nthe seeded round-trip case from the test suite:\n")
set.seed(1)
for (n in c(10, 50)) {
    X <- matrix(rnorm(n * 2 * n), n, 2 * n)
    S <- X %*% t(X)
    Dh <- 1 / sqrt(diag(S))
    C <- S * outer(Dh, Dh)
    C <- (C + t(C)) / 2
    z <- gft(C)
    for (nm in c("inv_gft", "inv_gft_fp", "inv_gft_broyden",
                 "inv_gft_newton")) {
        r <- get(nm)(z)
        cat(sprintf("  n = %2d  %-16s converged %-5s err %.2e  eighs %5d  "
                    , n, nm, r$converged, r$err, r$eighs))
        cat(sprintf("max|C-C0| %.2e\n", max(abs(r$C - C))))
    }
}
