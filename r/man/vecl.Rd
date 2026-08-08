\name{vecl}
\alias{vecl}
\alias{unvecl}
\title{Stack and Unstack Below-Diagonal Elements}
\description{
  \code{vecl} stacks the below-diagonal elements of a square matrix
  column by column.  \code{unvecl} is its right inverse on symmetric
  matrices with zero diagonal: it forms the symmetric matrix with zero
  diagonal and below-diagonal elements \code{z}.
}
\usage{
vecl(M)
unvecl(z)
}
\arguments{
  \item{M}{a square matrix.}
  \item{z}{a numeric vector of length \eqn{n(n-1)/2} for some integer
    \eqn{n}.}
}
\value{
  \code{vecl} returns a numeric vector of length \eqn{n(n-1)/2}.
  \code{unvecl} returns an \eqn{n \times n}{n x n} symmetric numeric
  matrix with zero diagonal.
}
\examples{
M <- matrix(1:9, 3, 3)
vecl(M)                 # c(2, 3, 6)
unvecl(c(0.1, 0.2, 0.3))
}
\keyword{array}
