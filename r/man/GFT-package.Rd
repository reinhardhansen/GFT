\name{GFT-package}
\alias{GFT-package}
\alias{GFT}
\docType{package}
\title{Generalized Fisher Transformation of Correlation Matrices}
\description{
  Forward and inverse generalized Fisher transformation (GFT) of
  correlation matrices.  The GFT maps a non-singular \eqn{n \times n}{n x n}
  correlation matrix \eqn{C} to the real vector
  \eqn{\gamma = \mathrm{vecl}(\log C)}{gamma = vecl(log C)} of
  below-diagonal elements of the matrix logarithm of \eqn{C}.  The map is
  a bijection between the set of positive definite correlation matrices
  and \eqn{R^d} with \eqn{d = n(n-1)/2} (Archakov and Hansen, 2021), and
  generalizes Fisher's \eqn{z}-transformation, to which it reduces for
  \eqn{n = 2}.
}
\details{
  The forward map is computed by \code{\link{gft}}.  The inverse is
  computed from the variational characterization
  \deqn{x^*(z) = \arg\min_x \; \mathrm{tr}\, e^{A[x;z]} - \textstyle\sum_i x_i,}{
        x*(z) = argmin_x tr(exp(A[x;z])) - sum(x),}
  where \eqn{A[x;z]} is symmetric with off-diagonal elements \eqn{z} and
  diagonal \eqn{x}, by the following solvers:
  \describe{
    \item{\code{\link{inv_gft}}}{GFT-FP+N (recommended): fixed-point
      phase in the log domain, then matrix-free inexact Newton via
      preconditioned conjugate gradients.}
    \item{\code{\link{inv_gft_fp}}}{The Archakov-Hansen fixed point.}
    \item{\code{\link{inv_gft_broyden}}}{Broyden's method as in Chen,
      Fei and Yu (2025).}
    \item{\code{\link{inv_gft_newton}}}{Full Newton with the exact
      \eqn{O(n^4)} Hessian.}
  }
  The implementation is a line-faithful port of the Julia reference
  implementation by the same authors and uses base R only.
}
\references{
  Archakov, I. and Hansen, P. R. (2021).
  A new parametrization of correlation matrices.
  \emph{Econometrica}, \bold{89}(4), 1699--1715.
  \doi{10.3982/ECTA16910}

  Archakov, I. and Hansen, P. R. (2026).
  A variational approach to the generalized Fisher transformation of
  correlation matrices.  Working paper.

  Chen, H., Fei, Y. and Yu, J. (2025).
  Multivariate stochastic volatility models based on generalized Fisher
  transformation.
  \emph{Journal of Econometrics}, \bold{251}, 106041.
}
\author{
  Ilya Archakov and Peter Reinhard Hansen.

  Maintainer: Peter Reinhard Hansen \email{hansen@unc.edu}
}
\examples{
C <- 0.9^abs(outer(1:5, 1:5, "-"))   # Toeplitz correlation matrix
z <- gft(C)                          # forward transformation
r <- inv_gft(z)                      # inverse transformation
max(abs(r$C - C))                    # round trip at machine precision
}
\keyword{package}
