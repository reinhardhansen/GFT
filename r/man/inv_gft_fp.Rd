\name{inv_gft_fp}
\alias{inv_gft_fp}
\title{Inverse GFT by the Archakov-Hansen Fixed Point}
\description{
  Reconstructs the correlation matrix \eqn{C} with
  \eqn{\mathrm{vecl}(\log C) = z}{vecl(log C) = z} by the fixed-point
  iteration of Archakov and Hansen (2021),
  \eqn{x \leftarrow x - \log \mathrm{diag}(e^{A[x;z]})}{x <- x - log
  diag(exp(A[x;z]))}, evaluated in the log domain throughout.
}
\usage{
inv_gft_fp(z, x0 = NULL, tol = 1e-13, maxit = 5000)
}
\arguments{
  \item{z}{numeric vector of length \eqn{n(n-1)/2}.}
  \item{x0}{optional starting value of length \eqn{n}; defaults to zero.}
  \item{tol}{convergence tolerance on
    \eqn{\|\mathrm{diag}(e^A) - 1\|_\infty}{max |diag(exp(A)) - 1|}.}
  \item{maxit}{maximum number of iterations.}
}
\value{
  An object of class \code{"gft_inv"}; see \code{\link{inv_gft}} for the
  components.
}
\details{
  Globally convergent, with one eigendecomposition per iteration.  The
  local linear rate degrades as the spectrum of \eqn{C} spreads;
  \code{\link{inv_gft}} switches to an inexact Newton phase precisely to
  avoid this slowdown while retaining the fixed point's robustness.
}
\references{
  Archakov, I. and Hansen, P. R. (2021).
  A new parametrization of correlation matrices.
  \emph{Econometrica}, \bold{89}(4), 1699--1715.
  \doi{10.3982/ECTA16910}
}
\seealso{\code{\link{inv_gft}}.}
\examples{
z <- gft(0.9^abs(outer(1:5, 1:5, "-")))
r <- inv_gft_fp(z)
r$eighs
}
\keyword{multivariate}
\keyword{optimize}
