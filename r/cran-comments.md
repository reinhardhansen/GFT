# cran-comments

## Submission

First submission of GFT 1.0.0.

## Test environments

- local: macOS (Apple Silicon), R 4.4.2:
  0 errors, 0 warnings, 3 NOTEs
- win-builder, R-devel (2026-08-07 r90377 ucrt):
  0 errors, 0 warnings, 1 NOTE (new submission)
- win-builder, R-release (R 4.6.1):
  0 errors, 0 warnings, 1 NOTE (new submission)

## NOTEs

- "New submission": first submission.
- "unable to verify current time": local environment artifact
  (timestamp verification service unreachable).
- HTML validation problems ("<main> is not recognized"): artifact of
  the outdated HTML Tidy shipped with macOS; not present with a
  current Tidy.

## Notes for reviewers

- The package uses base R only (no compiled code, no dependencies).
- Golden-value tests verify agreement with the authors' independently
  written Julia and NumPy implementations of the same algorithms.
- 'GFT' in the Title/Description is the generalized Fisher
  transformation of Archakov and Hansen (2021) <doi:10.3982/ECTA16910>.
