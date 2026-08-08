\name{print.gft_inv}
\alias{print.gft_inv}
\title{Print an Inverse GFT Result}
\description{
  Compact display of an object of class \code{"gft_inv"} as returned by
  \code{\link{inv_gft}} and the other inverse-GFT solvers.
}
\usage{
\method{print}{gft_inv}(x, ...)
}
\arguments{
  \item{x}{an object of class \code{"gft_inv"}.}
  \item{\dots}{ignored.}
}
\value{
  \code{x}, invisibly.
}
\examples{
r <- inv_gft(gft(0.9^abs(outer(1:5, 1:5, "-"))))
print(r)
}
\keyword{print}
