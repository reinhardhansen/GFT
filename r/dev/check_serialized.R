# Cross-language agreement on serialized random inputs (cf. supplement S4
# and julia/check_serialized.jl).  Reruns the four methods in R on the
# identical z draws and compares eigendecomposition counts and convergence
# flags with the NumPy reference counts in py_counts.csv.
#
# Expects `serialized_dir` to be defined before sourcing.  Under webR
# (dev/run-webr-serialized.mjs), Newton draws whose reference count
# exceeds `newton_cap` are skipped: those are the stagnating draws, and
# rerunning ~500 O(n^4) Hessians per draw is prohibitive in WebAssembly.
# Set newton_cap <- Inf on a native R installation to rerun everything.
#
# Dev only, not shipped (dev/ is in .Rbuildignore).

TOL <- 1e-13
if (!exists("newton_cap")) newton_cap <- 300
if (!exists("methods_filter"))
    methods_filter <- c("fp", "broyden", "newton", "fpn")

readz <- function(path)
    lapply(strsplit(readLines(path), ","), as.numeric)

meths <- list(
    fp      = function(z) inv_gft_fp(z, tol = TOL, maxit = 5000),
    broyden = function(z) inv_gft_broyden(z, tol = TOL),
    newton  = function(z) inv_gft_newton(z, tol = TOL, warm = 1),
    fpn     = function(z) inv_gft(z, tol = TOL)
)

py <- read.csv(file.path(serialized_dir, "py_counts.csv"))
stopifnot(nrow(py) == 300)
pykey <- paste(py$design, py$draw, py$method, sep = "|")

cat("method    exact-match  |count diff|>0 (max)   conv mismatches",
    "  skipped\n")
for (m in methods_filter) {
    nmatch <- 0; ndiff <- 0; maxdiff <- 0; nconv <- 0; tot <- 0
    nskip <- 0
    for (d in c("wishart_n100", "factor_n100", "z_sd4_n50")) {
        zs <- readz(file.path(serialized_dir, paste0("zs_", d, ".csv")))
        for (j in seq_along(zs)) {
            i <- match(paste(d, j, m, sep = "|"), pykey)
            ref_e <- py$eighs[i]
            ref_c <- py$converged[i]
            if (m == "newton" && ref_e > newton_cap) {
                nskip <- nskip + 1
                next
            }
            r <- meths[[m]](zs[[j]])
            tot <- tot + 1
            if (as.integer(r$converged) != ref_c) {
                nconv <- nconv + 1
            } else if (r$eighs == ref_e) {
                nmatch <- nmatch + 1
            } else {
                ndiff <- ndiff + 1
                maxdiff <- max(maxdiff, abs(r$eighs - ref_e))
            }
        }
    }
    cat(sprintf("%-8s  %3d/%-3d      %2d (max %3d)            %2d       %7d\n",
                m, nmatch, tot, ndiff, maxdiff, nconv, nskip))
}
