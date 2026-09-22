\name{inv_gft_path}
\alias{inv_gft_path}
\alias{gft_predict}
\title{Sequential Inversion with Warm Starts and the Tangent Predictor}
\description{
  \code{inv_gft_path} inverts a sequence of GFT vectors, starting each
  solve from the previous solution or from its first-order (tangent)
  prediction; \code{gft_predict} computes that prediction.
}
\usage{
inv_gft_path(zs, predictor = TRUE, rtol = 1e-3, tol = 1e-13, ...)
gft_predict(z_prev, x_prev, lam, Q, z_new, rtol = 1e-3)
}
\arguments{
  \item{zs}{a list of numeric vectors of length \eqn{n(n-1)/2}.}
  \item{predictor}{if \code{TRUE}, start each solve from the tangent
    prediction; otherwise from the previous solution.}
  \item{rtol}{relative residual to which the predictor's linear system is
    solved.}
  \item{tol, \dots}{passed to \code{\link{inv_gft}}.}
  \item{z_prev, x_prev}{the previous input and its solution.}
  \item{lam, Q}{eigenvalues and eigenvectors of \eqn{A[x_{prev}; z_{prev}]}{A[x_prev; z_prev]}.}
  \item{z_new}{the new input.}
}
\value{
  \code{inv_gft_path}: a list of \code{"gft_inv"} objects, one per input,
  whose \code{hvs} components include the predictor's Hessian-vector
  products, with attribute \code{"nbdz"} counting the predictor
  evaluations.  \code{gft_predict}: a list with the predicted diagonal
  \code{x} and the number \code{hvs} of products used.
}
\details{
  Differentiating \eqn{\ell(x^*(z); z) = 0} gives the prediction
  \eqn{\hat x = x_{prev} + p}{xhat = x_prev + p} with
  \eqn{H p = -(B \Delta z + D \ell)}{H p = -(B dz + D ell)}, where
  \eqn{B\Delta z}{B dz} is the derivative of \eqn{\mathrm{diag}(e^A)}{diag(exp(A))}
  in the direction of the off-diagonal change (three matrix
  multiplications) and \eqn{D\ell}{D ell} corrects for the residual left
  by the previous solve.  The system is solved by preconditioned
  conjugate gradients with the stored eigendecomposition, so the
  prediction costs no eigendecomposition.  The prediction error is
  \eqn{O(h^2)} in the step size.  The predictor pays at moderate step
  sizes; for very small steps warm starts alone are nearly as good, and
  for large steps its linear solve can cost more than the
  eigendecomposition it saves.
}
\seealso{\code{\link{inv_gft}}.}
\examples{
set.seed(4)
b <- 0.85 + 0.149 * runif(30)
z0 <- gft(b \%*\% t(b) + diag(1 - b^2))
zs <- lapply(0:5, function(k) z0 + 0.02 * k * rnorm(length(z0)) / sqrt(length(z0)))
out <- inv_gft_path(zs)
sapply(out, `[[`, "eighs")
}
\keyword{optimize}
