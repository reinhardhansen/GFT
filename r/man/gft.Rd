\name{gft}
\alias{gft}
\title{Generalized Fisher Transformation}
\description{
  Computes the generalized Fisher transformation
  \eqn{\gamma = \mathrm{vecl}(\log C)}{gamma = vecl(log C)}: the
  below-diagonal elements, stacked column by column, of the matrix
  logarithm of a positive definite correlation matrix \eqn{C}.
}
\usage{
gft(C)
}
\arguments{
  \item{C}{a positive definite correlation matrix (square, symmetric,
    numeric).  The symmetric part \eqn{(C + C')/2} is used.}
}
\value{
  A numeric vector of length \eqn{n(n-1)/2} containing
  \eqn{\mathrm{vecl}(\log C)}{vecl(log C)}.
}
\details{
  The matrix logarithm is computed from the eigendecomposition of
  \eqn{C}.  For \eqn{n = 2} the transformation reduces to Fisher's
  classical \eqn{z}-transformation \eqn{z = \mathrm{atanh}(\rho)}{z =
  atanh(rho)}.  The map is a bijection between the positive definite
  correlation matrices and \eqn{R^{n(n-1)/2}}; its inverse is computed
  by \code{\link{inv_gft}}.
}
\references{
  Archakov, I. and Hansen, P. R. (2021).
  A new parametrization of correlation matrices.
  \emph{Econometrica}, \bold{89}(4), 1699--1715.
  \doi{10.3982/ECTA16910}
}
\seealso{\code{\link{inv_gft}}, \code{\link{vecl}}, \code{\link{unvecl}}.}
\examples{
# n = 2: reduces to Fisher's z-transformation
C <- matrix(c(1, 0.5, 0.5, 1), 2, 2)
all.equal(gft(C), atanh(0.5))

C <- 0.9^abs(outer(1:5, 1:5, "-"))
z <- gft(C)
max(abs(inv_gft(z)$C - C))
}
\keyword{multivariate}
