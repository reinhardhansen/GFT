\name{inv_gft_lbfgs}
\alias{inv_gft_lbfgs}
\title{Limited-Memory BFGS for the Inverse GFT}
\description{
  Limited-memory BFGS on the objective
  \eqn{f(x) = \mathrm{tr}\, e^{A[x;z]} - \sum_i x_i}{f(x) = tr(exp(A[x;z])) - sum(x)},
  with two-loop recursion, Armijo backtracking on \eqn{f} and the
  untested-step rule of \code{\link{inv_gft}} near the rounding floor.
  A general-purpose comparator.
}
\usage{
inv_gft_lbfgs(z, x0 = NULL, tol = 1e-13, maxit = 500, m = 10,
              globalized = FALSE)
}
\arguments{
  \item{z, x0, tol, maxit}{as in \code{\link{inv_gft}}.}
  \item{m}{memory of the quasi-Newton update.}
  \item{globalized}{if \code{TRUE}, precede the iteration by fixed-point
    steps in the log domain until
    \eqn{\max_i \ell_i \le \log 2}{max(ell) <= log(2)}.}
}
\value{An object of class \code{"gft_inv"}; see \code{\link{inv_gft}}.}
\references{
  Liu, D. C. and Nocedal, J. (1989).
  On the limited memory BFGS method for large scale optimization.
  \emph{Mathematical Programming}, \bold{45}, 503--528.
}
\seealso{\code{\link{inv_gft}}, \code{\link{inv_gft_anderson}}.}
\examples{
set.seed(3)
z <- rnorm(45, sd = 2)
r <- inv_gft_lbfgs(z, globalized = TRUE)
c(lbfgs = r$eighs, gft_fpn = inv_gft(z)$eighs)
}
\keyword{optimize}
