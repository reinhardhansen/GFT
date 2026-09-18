# cran-comments

## Submission

GFT 1.0.1.  This release fixes the ERROR reported by the ATLAS check on
2026-08-22 (`tests/testthat/test-gft.R:58`, `expect_true(r$converged)` in
"round trips, all solvers").  It also applies the DESCRIPTION wording
change requested at acceptance of 1.0.0.

## The ATLAS failure

The failure was a genuine defect in `inv_gft_newton()`, not a tolerance
set too tight and not an artifact of the ATLAS BLAS.

Close to the solution the objective f = tr(exp(A)) - sum(x) is flat to
within cancellation, so the Armijo condition
f(x + t s) <= f(x) + 1e-4 t g's carries no information.  The line search
therefore exhausted all 27 halvings at every iteration, each costing one
eigendecomposition, the fixed-point fallback was taken, and the iteration
locked into a cycle: the recorded error was constant to the last bit for
over 490 consecutive iterations before `maxit` was reached and
`converged = FALSE` returned.

Over random correlation matrices of the class used in that test, the
failure rate at n = 10 was about 5%, with the error stalling as high as
4e-10 and the cost reaching 12,375 eigendecompositions.  The defect was
present on every platform; ATLAS's arithmetic merely moved the
`set.seed(1)` draw into the affected set, which is why the reference-BLAS
checks passed.

The fix ports into `inv_gft_newton()` the two safeguards that
`inv_gft()` already carried: the full Newton step is taken untested once
the predicted decrease falls below the floating-point resolution of f, and
a persistent lack of progress near the rounding floor hands the iteration
to the contractive fixed point.  After the fix the failure rate over the
same sweep is 0 and the worst-case cost is 6 eigendecompositions.  The
cross-language golden values are unchanged and still agree to 1e-13 or
better.

These are controlled by a new `safeguard` argument, default TRUE, so the
package's tests and any ordinary use take the fixed path.  The published
comparator that the accompanying paper benchmarks is the unsafeguarded
variant, which is retained under `safeguard = FALSE` for reproducibility
and is documented as able to return `converged = FALSE`.  No test
exercises that mode.

`inv_gft()`, `inv_gft_fp()` and `inv_gft_broyden()` are untouched.
`inv_gft_broyden()` was checked over 300 additional random problems for
the same stalling behaviour and showed none.

## Test environments

- local: macOS 26.5.2 (aarch64-apple-darwin20), R 4.4.2:
  0 errors, 0 warnings, 3 NOTEs
- win-builder, R-devel (2026-08-21 r90440 ucrt, the same revision on
  which the ATLAS check reported the ERROR):
  0 errors, 0 warnings, 1 NOTE
- win-builder, R-release (R 4.6.1, 2026-06-24 ucrt):
  0 errors, 0 warnings, 1 NOTE

The ATLAS test failure is resolved: "checking tests ... OK".

## NOTEs

- "Days since last update: 1".  This release exists only to correct the
  ERROR reported by the ATLAS check on 2026-08-22, within the requested
  window.  I do not expect to submit again soon.
- "Possibly misspelled words in DESCRIPTION: FP, GFT".  Both are
  acronyms, not misspellings: GFT is the generalized Fisher
  transformation and GFT-FP+N is the name of the algorithm.  They are
  flagged in this version because the single quotes around them were
  removed at your request when 1.0.0 was accepted.
- "unable to verify current time": local environment artifact
  (timestamp verification service unreachable).
- HTML validation problems ("<main> is not recognized"): artifact of the
  outdated HTML Tidy shipped with macOS; not present with a current Tidy.

## Notes for reviewers

- The package uses base R only (no compiled code, no dependencies).
- Golden-value tests verify agreement with the authors' independently
  written Julia and NumPy implementations of the same algorithms.  Both
  reference implementations have received the identical fix, so the
  cross-language check remains meaningful.
- GFT in the Title and Description is the generalized Fisher
  transformation of Archakov and Hansen (2021) <doi:10.3982/ECTA16910>.
