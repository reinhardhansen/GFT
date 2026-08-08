\name{inv_gft_broyden}
\alias{inv_gft_broyden}
\title{Inverse GFT by Broyden's Method}
\description{
  Reconstructs the correlation matrix \eqn{C} with
  \eqn{\mathrm{vecl}(\log C) = z}{vecl(log C) = z} by Broyden's method
  applied to the residual
  \eqn{F(x) = \log \mathrm{diag}(e^{A[x;z]})}{F(x) = log diag(exp(A[x;z]))},
  as in Chen, Fei and Yu (2025).  Reference implementation for
  benchmarking against \code{\link{inv_gft}}.
}
\usage{
inv_gft_broyden(z, x0 = NULL, tol = 1e-13, maxit = 500, warm = 1,
                globalized = FALSE)
}
\arguments{
  \item{z}{numeric vector of length \eqn{n(n-1)/2}.}
  \item{x0}{optional starting value of length \eqn{n}; defaults to zero.}
  \item{tol}{convergence tolerance on
    \eqn{\|\mathrm{diag}(e^A) - 1\|_\infty}{max |diag(exp(A)) - 1|}.}
  \item{maxit}{maximum number of iterations.}
  \item{warm}{number of initial fixed-point steps before the Jacobian is
    formed (ignored when \code{globalized = TRUE}).}
  \item{globalized}{if \code{TRUE}, replace the one-step initialization
    with the same log-domain fixed-point phase as GFT-FP+N (fixed-point
    steps until \eqn{\max_i \ell_i \le \log 2}{max ell_i <= log 2}), and
    only then form the Jacobian.}
}
\value{
  An object of class \code{"gft_inv"}; see \code{\link{inv_gft}} for the
  components.  On divergence the result has \code{converged = FALSE} and
  \code{err = Inf}.
}
\details{
  The exact Jacobian is computed once (an \eqn{O(n^4)} Hessian), then
  updated by rank-one Sherman-Morrison updates of its inverse, with one
  eigendecomposition per iteration and no line search.  Without
  globalization the method can diverge for large \eqn{\|z\|}{||z||};
  divergence is reported gracefully via \code{converged = FALSE}.
}
\references{
  Chen, H., Fei, Y. and Yu, J. (2025).
  Multivariate stochastic volatility models based on generalized Fisher
  transformation.
  \emph{Journal of Econometrics}, \bold{251}, 106041.
}
\seealso{\code{\link{inv_gft}}.}
\examples{
z <- gft(0.9^abs(outer(1:5, 1:5, "-")))
r <- inv_gft_broyden(z)
r$converged
}
\keyword{multivariate}
\keyword{optimize}
