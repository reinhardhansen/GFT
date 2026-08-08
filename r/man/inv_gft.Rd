\name{inv_gft}
\alias{inv_gft}
\title{Inverse Generalized Fisher Transformation (GFT-FP+N)}
\description{
  Reconstructs the unique positive definite correlation matrix \eqn{C}
  with \eqn{\mathrm{vecl}(\log C) = z}{vecl(log C) = z} by the GFT-FP+N
  algorithm (recommended solver).
}
\usage{
inv_gft(z, x0 = NULL, tol = 1e-13, maxit = 500, delta = 1,
        exact_hess = FALSE)
}
\arguments{
  \item{z}{numeric vector of length \eqn{n(n-1)/2}: the below-diagonal
    elements of \eqn{\log C}, stacked column by column.}
  \item{x0}{optional numeric vector of length \eqn{n}: starting value
    for the diagonal of \eqn{\log C} (e.g. from a previous solution,
    for warm starts).  Defaults to zero.}
  \item{tol}{convergence tolerance on
    \eqn{\|\mathrm{diag}(e^A) - 1\|_\infty}{max |diag(exp(A)) - 1|}.}
  \item{maxit}{maximum number of iterations.}
  \item{delta}{phase switch threshold: fixed-point steps are used while
    \eqn{\|\mathrm{diag}(e^A) - 1\|_\infty > \delta}{max |diag(exp(A)) - 1| > delta}.}
  \item{exact_hess}{if \code{TRUE}, solve the Newton system with the
    explicit \eqn{O(n^4)} Hessian and a Cholesky factorization instead
    of matrix-free conjugate gradients.  Identical algorithm otherwise;
    mainly for benchmarking.}
}
\value{
  An object of class \code{"gft_inv"}: a list with components
  \item{x}{the solution: diagonal of \eqn{\log C}.}
  \item{C}{the reconstructed correlation matrix \eqn{e^{A[x^*;z]}}{exp(A[x*;z])}.}
  \item{iters}{number of iterations.}
  \item{eighs}{number of eigendecompositions, the dominant \eqn{O(n^3)}
    kernel.}
  \item{hvs}{number of Hessian-vector products.}
  \item{err}{final value of
    \eqn{\|\mathrm{diag}(e^A) - 1\|_\infty}{max |diag(exp(A)) - 1|}.}
  \item{converged}{logical.}
  \item{hist}{the error after each iteration.}
}
\details{
  The solution solves the strictly convex problem
  \deqn{x^*(z) = \arg\min_x \; \mathrm{tr}\, e^{A[x;z]} - \textstyle\sum_i x_i,}{
        x*(z) = argmin_x tr(exp(A[x;z])) - sum(x),}
  where \eqn{A[x;z]} is symmetric with off-diagonal elements \eqn{z} and
  diagonal \eqn{x}.  Phase 1 applies fixed-point steps
  \eqn{x \leftarrow x - \log \mathrm{diag}(e^A)}{x <- x - log diag(exp(A))}
  in the log domain (robust to overflow for any spectrum).  Phase 2 is an
  inexact Newton method: the system \eqn{H s = -g} is solved matrix-free
  by conjugate gradients with Jacobi preconditioner
  \eqn{\mathrm{diag}(e^A)}{diag(exp(A))}, exact Hessian-vector products
  (two matrix multiplications each), and forcing tolerance
  \eqn{\eta = \min(1/2, \sqrt{\|g\|})}{eta = min(1/2, sqrt(||g||))}
  (Eisenstat and Walker, 1996), giving superlinear convergence of order
  3/2.  Steps are safeguarded by Armijo backtracking on the objective,
  with a fixed-point step substituted if the line search fails, and a
  fixed-point finisher if progress stalls at the rounding floor.
}
\references{
  Archakov, I. and Hansen, P. R. (2021).
  A new parametrization of correlation matrices.
  \emph{Econometrica}, \bold{89}(4), 1699--1715.
  \doi{10.3982/ECTA16910}

  Archakov, I. and Hansen, P. R. (2026).
  A variational approach to the generalized Fisher transformation of
  correlation matrices.  Working paper.

  Eisenstat, S. C. and Walker, H. F. (1996).
  Choosing the forcing terms in an inexact Newton method.
  \emph{SIAM Journal on Scientific Computing}, \bold{17}(1), 16--32.
}
\seealso{
  \code{\link{gft}} for the forward map; \code{\link{inv_gft_fp}},
  \code{\link{inv_gft_broyden}}, \code{\link{inv_gft_newton}} for the
  reference solvers.
}
\examples{
C <- 0.9^abs(outer(1:5, 1:5, "-"))
z <- gft(C)
r <- inv_gft(z)
r
max(abs(r$C - C))

# warm start from a nearby solution
z2 <- z + 0.01
r2 <- inv_gft(z2, x0 = r$x)

# an arbitrary vector in R^d is always a valid input
set.seed(1)
ra <- inv_gft(rnorm(45, sd = 2))   # n = 10
range(diag(ra$C))                  # unit diagonal
min(eigen(ra$C)$values) > 0        # positive definite
}
\keyword{multivariate}
\keyword{optimize}
