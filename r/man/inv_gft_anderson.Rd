\name{inv_gft_anderson}
\alias{inv_gft_anderson}
\title{Anderson Acceleration of the GFT Fixed Point}
\description{
  Anderson acceleration (type II, window \code{m}) of the fixed-point
  map \eqn{x \mapsto x - \log\mathrm{diag}(e^{A[x;z]})}{x -> x - log diag(exp(A[x;z]))},
  with an optional safeguard that makes it globally convergent.  A
  Jacobian-free comparator for \code{\link{inv_gft}}.
}
\usage{
inv_gft_anderson(z, x0 = NULL, tol = 1e-13, maxit = 5000, m = 5,
                 guarded = FALSE, theta = 0.25)
}
\arguments{
  \item{z, x0, tol, maxit}{as in \code{\link{inv_gft}}.}
  \item{m}{memory: number of stored differences of iterates and
    residuals.}
  \item{guarded}{if \code{TRUE}, an Anderson proposal \eqn{y} is accepted
    only if \eqn{f(y) \le f(x) - \theta V(x)}{f(y) <= f(x) - theta V(x)},
    where \eqn{V(x) = \sum_i (e^{\ell_i} - 1 - \ell_i)}{V(x) = sum_i (exp(ell_i) - 1 - ell_i)}
    is the guaranteed decrease of the fixed-point step; otherwise the
    memory is cleared and a fixed-point step is taken.  Proposals are
    accepted untested once \eqn{\theta V(x)}{theta V(x)} is below the
    floating-point resolution of \eqn{f}.}
  \item{theta}{fraction of the fixed-point decrease required by the
    safeguard.}
}
\value{An object of class \code{"gft_inv"}; see \code{\link{inv_gft}}.
  The \code{hvs} component is zero.}
\details{
  In the difference form of Walker and Ni (2011) each iteration costs one
  eigendecomposition and a small least-squares solve.  The ordinary
  scheme has no convergence guarantee and can diverge for large
  \code{m}; with \code{guarded = TRUE} every step decreases \eqn{f} by at
  least \eqn{\theta V(x)}{theta V(x)}, which implies convergence from
  every starting point (Archakov and Hansen, 2026, Section S5).
}
\references{
  Walker, H. F. and Ni, P. (2011).
  Anderson acceleration for fixed-point iterations.
  \emph{SIAM Journal on Numerical Analysis}, \bold{49}, 1715--1735.

  Archakov, I. and Hansen, P. R. (2026).
  Fast inversion of the generalized Fisher transformation of
  correlation matrices.  Working paper, arXiv:2609.19028.
}
\seealso{\code{\link{inv_gft}}, \code{\link{inv_gft_fp}}.}
\examples{
set.seed(2)
z <- rnorm(45, sd = 2)                 # n = 10
r <- inv_gft_anderson(z, guarded = TRUE)
r
max(abs(r$x - inv_gft(z)$x)) < 1e-9
}
\keyword{optimize}
