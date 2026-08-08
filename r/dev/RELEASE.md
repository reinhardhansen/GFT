# Release checklist (run on a machine with native R)

The package was developed and validated under webR 4.6.0 (see
run-webr.mjs); `R CMD check --as-cran` itself must run on native R.

1. Full check (from `AlgoGFT/`):

       R CMD build r
       R CMD check --as-cran GFT_1.0.0.tar.gz

   Expect: 0 errors, 0 warnings; NOTE "new submission" only.
   Requires testthat, knitr, rmarkdown installed.

2. Serialized cross-check on native R (fast; reruns everything
   including the stagnating Newton draws):

       R -e "source('r/R/GFT.R'); serialized_dir <- 'julia/serialized'; \
             newton_cap <- Inf; source('r/dev/check_serialized.R')"

   Native R 4.4.2 results (macOS arm64, Aug 8 2026, newton_cap = Inf):
   fpn 75/75 EXACT vs NumPy counts; fp 49/75 (26 diffs, max 9);
   broyden 74/75 (1 diff, max 2); newton 37/75 (25 diffs, max 463,
   13 conv mismatches -- all in the stagnation regime, cf. the 5
   Julia-vs-NumPy conv mismatches in supplement S4).
   webR 4.6.0 results for comparison: fpn 75/75; fp 48/75 (max 8);
   broyden 72/75 (max 3); newton 46/71 (max 100, 1 conv mismatch,
   4 stagnating draws skipped).

3. Optional additional environments before submission:
   - https://win-builder.r-project.org (devel + release)
   - macOS builder: https://mac.r-project.org/macbuilder/submit.html

4. Update cran-comments.md with actual check results.

5. Submit at https://cran.r-project.org/submit.html
   (maintainer: hansen@unc.edu confirms via email link).

6. Name check at submission: CRAN rejects case-insensitive clashes
   automatically; as of Aug 2026 no CRAN package named GFT was found.

7. After acceptance: tag r-v1.0.0 in the GitHub repo; add the r/
   subdir note to the repo README.
