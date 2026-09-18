# GFT 1.0.1

## Bug fixes

* `inv_gft_newton()` gains a `safeguard` argument, default `TRUE`.  The
  new default fixes the stall described below.  `safeguard = FALSE`
  reproduces the previous behaviour, which is the published comparator of
  Chen, Fei and Yu (2025) with only the Armijo line search added, and is
  the variant benchmarked in the accompanying paper.  Set it when
  reproducing those results.

* With the new default, `inv_gft_newton()` no longer stalls short of the
  requested tolerance and returns `converged = FALSE` on a small fraction
  of well conditioned problems.  Close to the solution the objective
  `f = tr(exp(A)) - sum(x)` is flat to within cancellation, so the Armijo
  test carries no information; the line search then exhausted all of its
  halvings at every iteration, the fixed-point fallback was taken instead,
  and the error could lock at a constant value several orders of magnitude
  above the attainable floor until `maxit` was reached.  Two safeguards
  that `inv_gft()` already used have been added: the full Newton step is
  now taken untested once the predicted decrease falls below the
  floating-point resolution of `f`, and a persistent lack of progress near
  the rounding floor hands the iteration to the (contractive) fixed point.

  In a sweep over random correlation matrices at n = 10, the failure rate
  falls from about 5% to 0, and the worst-case cost from 12,375
  eigendecompositions to 6.

  The defect was present on all platforms and is not specific to any BLAS
  or LAPACK build; it surfaced on CRAN's ATLAS check because that
  arithmetic moved one test matrix into the affected set.

* `inv_gft()`, `inv_gft_fp()` and `inv_gft_broyden()` are unchanged.  Their
  results are unaffected by this release.

## Documentation

* Removed the single quotes around acronyms and author names in
  `DESCRIPTION`, as requested by the CRAN reviewer.  Single quotes are now
  used only for software names.
