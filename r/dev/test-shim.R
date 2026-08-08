# Minimal testthat emulation so the test suite can run without testthat
# installed (used by dev/run-webr.mjs in the development sandbox).
# NOT shipped: dev/ is excluded via .Rbuildignore.  On a normal system,
# run the real thing instead:  testthat::test_local()  or R CMD check.

.shim <- new.env()
.shim$pass <- 0L
.shim$fail <- 0L
.shim$msgs <- character(0)
.shim$current <- ""

.record <- function(ok, msg) {
    if (isTRUE(ok)) {
        .shim$pass <- .shim$pass + 1L
    } else {
        .shim$fail <- .shim$fail + 1L
        .shim$msgs <- c(.shim$msgs, msg)
        cat("FAILED: ", msg, "\n", sep = "")
    }
    invisible(ok)
}

test_that <- function(desc, code) {
    .shim$current <- desc
    tryCatch({
        force(code)
        cat("ok: ", desc, "\n", sep = "")
    }, error = function(e) {
        .record(FALSE, paste0("[", desc, "] ERROR: ", conditionMessage(e)))
    })
    invisible(NULL)
}

expect_true <- function(x)
    .record(isTRUE(all(x)), paste0("[", .shim$current, "] expect_true"))

expect_lt <- function(a, b)
    .record(is.finite(a) && a < b,
            sprintf("[%s] expect_lt: %g !< %g", .shim$current, a, b))

expect_lte <- function(a, b)
    .record(is.finite(a) && a <= b,
            sprintf("[%s] expect_lte: %g !<= %g", .shim$current, a, b))

expect_gt <- function(a, b)
    .record(is.finite(a) && a > b,
            sprintf("[%s] expect_gt: %g !> %g", .shim$current, a, b))

expect_equal <- function(object, expected, tolerance = 1.5e-8, ...)
    .record(isTRUE(all.equal(object, expected, tolerance = tolerance,
                             check.attributes = FALSE)),
            paste0("[", .shim$current, "] expect_equal"))

expect_error <- function(expr, regexp = NULL) {
    err <- tryCatch({ expr; NULL }, error = function(e) conditionMessage(e))
    ok <- !is.null(err) && (is.null(regexp) || grepl(regexp, err))
    .record(ok, paste0("[", .shim$current, "] expect_error: got ",
                       if (is.null(err)) "no error" else err))
}

expect_output <- function(expr, regexp) {
    out <- paste(capture.output(expr), collapse = "\n")
    .record(grepl(regexp, out),
            paste0("[", .shim$current, "] expect_output: ", regexp))
}

`:::` <- function(pkg, name) {
    get(deparse(substitute(name)), envir = globalenv())
}

shim_summary <- function() {
    cat("\n== PASS ", .shim$pass, "  FAIL ", .shim$fail, " ==\n", sep = "")
    if (.shim$fail > 0)
        cat(paste(.shim$msgs, collapse = "\n"), "\n")
}
