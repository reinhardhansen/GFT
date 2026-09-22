# cran-comments

## Submission

GFT 1.2.0.  A feature release: the recommended solver `inv_gft()` is
revised (fewer eigendecompositions, same solutions, same interface with
new optional arguments), and four functions are added
(`inv_gft_anderson()`, `inv_gft_lbfgs()`, `inv_gft_path()`,
`gft_predict()`).  The version number jumps from 1.0.1 to 1.2.0 to match
the authors' Julia reference implementation, which this package ports
function for function.

## Test environments

- local: macOS 26.6.2 (aarch64-apple-darwin23), R 4.6.1:
  0 errors, 0 warnings, 1 NOTE
- win-builder, R-release (R 4.6.1, 2026-06-24 ucrt):
  0 errors, 0 warnings, 1 NOTE
- win-builder, R-devel (2026-09-21 r90579 ucrt):
  0 errors, 0 warnings, 1 NOTE

## NOTEs

- "Possibly misspelled words in DESCRIPTION: preconditioner".  This is
  the standard term in numerical linear algebra for a matrix that
  approximates the coefficient matrix of a linear system to accelerate
  an iterative solver; the package implements one.
- Local only: "Skipping checking HTML validation: 'tidy' doesn't look
  like recent enough HTML Tidy" (the HTML Tidy shipped with macOS).

## Notes for reviewers

- The package uses base R only (no compiled code, no dependencies).
- The golden-value tests of 1.0.x are unchanged and still pass to 1e-13
  or better; new tests check the algebraic identities behind the revised
  solver, agreement of all solver variants, the preconditioner's
  certified bounds, and the comparison solvers against `inv_gft()`.
- GFT in the Title and Description is the generalized Fisher
  transformation of Archakov and Hansen (2021) <doi:10.3982/ECTA16910>.
