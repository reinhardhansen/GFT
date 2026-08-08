\name{inv_gft_newton}
\alias{inv_gft_newton}
\title{Inverse GFT by Full Newton}
\description{
  Reconstructs the correlation matrix \eqn{C} with
  \eqn{\mathrm{vecl}(\log C) = z}{vecl(log C) = z} by a full Newton
  method with the exact \eqn{O(n^4)} Hessian recomputed at every
  iteration, Armijo backtracking on the objective, and an optional
  fixed-point warm start.  Reference implementation for benchmarking
  against \code{\link{inv_gft}}.
}
\usage{
inv_gft_newton(z, x0 = NULL, tol = 1e-13, maxit = 500, warm = 1)
}
\arguments{
  \item{z}{numeric vector of length \eqn{n(n-1)/2}.}
  \item{x0}{optional starting value of length \eqn{n}; defaults to zero.}
  \item{tol}{convergence tolerance on
    \eqn{\|\mathrm{diag}(e^A) - 1\|_\infty}{max |diag(exp(A)) - 1|}.}
  \item{maxit}{maximum number of iterations.}
  \item{warm}{number of initial fixed-point steps.}
}
\value{
  An object of class \code{"gft_inv"}; see \code{\link{inv_gft}} for the
  components.
}
\details{
  Each iteration forms the exact Hessian in \eqn{O(n^4)} time and solves
  the Newton system by Cholesky factorization; if the factorization
  fails or the line search cannot make progress, a fixed-point step is
  substituted.  \code{\link{inv_gft}} obtains the same fast local
  convergence at \eqn{O(n^3)} cost per iteration by solving the Newton
  system inexactly and matrix-free.
}
\references{
  Archakov, I. and Hansen, P. R. (2026).
  A variational approach to the generalized Fisher transformation of
  correlation matrices.  Working paper.
}
\seealso{\code{\link{inv_gft}}.}
\examples{
z <- gft(0.9^abs(outer(1:5, 1:5, "-")))
r <- inv_gft_newton(z)
r$converged
}
\keyword{multivariate}
\keyword{optimize}
