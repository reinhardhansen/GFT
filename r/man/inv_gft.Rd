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
        exact_hess = FALSE, residual = c("log", "gradient"),
        forcing = c("quad", "sqrt"), normalize = TRUE, phase = FALSE,
        adaptive = TRUE,
        preconditioner = c("diagonal", "quadrature", "auto"),
        kappa = 2, rmin = 2, rmax = 8, nmin = 0)
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
  \item{delta}{threshold of the optional initial fixed-point phase
    (\code{phase = TRUE}): fixed-point steps are used while
    \eqn{\|\mathrm{diag}(e^A) - 1\|_\infty > \delta}{max |diag(exp(A)) - 1| > delta}.}
  \item{exact_hess}{if \code{TRUE}, solve the Newton system with the
    explicit \eqn{O(n^4)} Hessian and a Cholesky factorization instead
    of matrix-free conjugate gradients.  Identical algorithm otherwise;
    mainly for benchmarking.}
  \item{residual}{\code{"log"} (default): Newton steps for the equation
    \eqn{\ell(x) = 0}, \eqn{\ell = \log\mathrm{diag}(e^A)}{ell = log diag(exp(A))},
    i.e. the system \eqn{H \delta = -D\ell}; \code{"gradient"}: Newton
    steps for \eqn{\nabla f = 0}, the system \eqn{H \delta = -g}.}
  \item{forcing}{inexact-Newton forcing tolerance:
    \eqn{\eta = \min(1/2, \|r\|)}{eta = min(1/2, ||r||)} (\code{"quad"},
    locally quadratic) or
    \eqn{\min(1/2, \|r\|^{1/2})}{min(1/2, ||r||^(1/2))} (\code{"sqrt"},
    order 3/2), with \eqn{r} the outer residual.}
  \item{normalize}{if \code{TRUE}, every evaluated point is shifted along
    the vector of ones so that \eqn{\mathrm{tr}\, e^{A} = n}{tr(exp(A)) = n},
    the exact minimizer of \eqn{f} along that direction; costs no
    eigendecomposition and removes overflow.}
  \item{phase}{if \code{TRUE}, take fixed-point steps while
    \eqn{\max_i \ell_i > \log(1 + \delta)}{max(ell) > log(1 + delta)}
    before the Newton phase.}
  \item{adaptive}{if \code{TRUE}, start each Armijo backtracking at
    \eqn{\min(1, 2 t_{\mathrm{prev}})}{min(1, 2 t_prev)}, with
    \eqn{t_{\mathrm{prev}}}{t_prev} the previously accepted step; otherwise
    at 1.}
  \item{preconditioner}{preconditioner of the conjugate-gradient solve:
    the diagonal \eqn{D = \mathrm{diag}(e^{\ell})}{D = diag(exp(ell))}
    (\code{"diagonal"}, default); the certified quadrature model
    \eqn{M_r} at every Newton step (\code{"quadrature"}); or the
    selection rule (\code{"auto"}): \eqn{M_r} whenever an order in
    \code{[rmin, rmax]} certifies
    \eqn{\kappa(M_r^{-1}H) \le}{kappa(M_r^-1 H) <=} \code{kappa} and
    \eqn{n \ge}{n >=} \code{nmin}, the diagonal otherwise.  The
    forcing test is unchanged.}
  \item{kappa, rmin, rmax, nmin}{parameters of the quadrature
    preconditioner and its selection rule; see \code{preconditioner}.}
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
  diagonal \eqn{x}.  Every evaluated point is normalized by the exact
  minimization of \eqn{f} along the vector of ones.  Fixed-point steps
  \eqn{x \leftarrow x - \ell}{x <- x - ell},
  \eqn{\ell = \log \mathrm{diag}(e^A)}{ell = log diag(exp(A))}, are
  taken while some \eqn{\ell_i < -700}{ell_i < -700}; otherwise an
  inexact Newton step for the equation \eqn{\ell(x) = 0} is computed:
  the system \eqn{H \delta = -D\ell},
  \eqn{D = \mathrm{diag}(e^{\ell})}{D = diag(exp(ell))}, is solved
  matrix-free by conjugate gradients with preconditioner \eqn{D}, exact
  Hessian-vector products (two matrix multiplications each), and
  forcing tolerance \eqn{\eta = \min(1/2, \|\ell\|)}{eta = min(1/2, ||ell||)}
  on the preconditioned residual (locally quadratic; Dembo, Eisenstat
  and Steihaug, 1982), capped at \eqn{2n} products.  A non-finite or
  non-descent direction is replaced by a fixed-point step; the full step
  is taken untested when its predicted decrease is below the
  floating-point resolution of \eqn{f} and
  \eqn{\|\ell\|_\infty < 10^{-3}}{max |ell| < 1e-3}; otherwise
  Armijo backtracking on \eqn{f}, with a fixed-point step substituted if
  the search fails.  Near the rounding floor the computation is finished
  by fixed-point steps.  Every fixed-point step and every tested Newton
  step decreases \eqn{f}, which gives global convergence.

  The options \code{residual}, \code{forcing}, \code{normalize},
  \code{phase} and \code{adaptive} reproduce the variants of the
  ablation in the paper; \code{residual = "gradient", forcing = "sqrt",
  normalize = FALSE, phase = TRUE, adaptive = FALSE} is the version of
  GFT-FP+N in package versions 1.0.x.

  The quadrature preconditioner replaces \eqn{D} by
  \eqn{M_r = \sum_j a_j e^{t_j A} \circ e^{(1-t_j)A}}{M_r = sum_j a_j exp(t_j A) o exp((1-t_j) A)}
  for the \eqn{r}-point Gauss-Legendre rule on \eqn{[0,1]}, built from
  the same eigendecomposition at \eqn{r} matrix multiplications and one
  Cholesky factorization.  It satisfies
  \eqn{M_r \preceq H \preceq \phi_r(\Delta) M_r}{M_r <= H <= phi_r(Delta) M_r}
  with \eqn{\Delta} the spectral spread of \eqn{A}, so the order is
  chosen from the extreme eigenvalues; it pays whenever two or more
  nodes are needed, increasingly with \eqn{n}.
}
\references{
  Archakov, I. and Hansen, P. R. (2021).
  A new parametrization of correlation matrices.
  \emph{Econometrica}, \bold{89}(4), 1699--1715.
  \doi{10.3982/ECTA16910}

  Archakov, I. and Hansen, P. R. (2026).
  Fast inversion of the generalized Fisher transformation of
  correlation matrices.  Working paper, arXiv:2609.19028.

  Dembo, R. S., Eisenstat, S. C. and Steihaug, T. (1982).
  Inexact Newton methods.
  \emph{SIAM Journal on Numerical Analysis}, \bold{19}, 400--408.

  Eisenstat, S. C. and Walker, H. F. (1996).
  Choosing the forcing terms in an inexact Newton method.
  \emph{SIAM Journal on Scientific Computing}, \bold{17}, 16--32.
}
\seealso{
  \code{\link{gft}} for the forward map; \code{\link{inv_gft_fp}},
  \code{\link{inv_gft_broyden}}, \code{\link{inv_gft_newton}},
  \code{\link{inv_gft_anderson}}, \code{\link{inv_gft_lbfgs}} for the
  comparison solvers; \code{\link{inv_gft_path}} for sequential
  inversion with the tangent predictor.
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

# the quadrature preconditioner halves the Hessian-vector products on
# ill-conditioned inputs
b <- 0.8 + 0.195 * runif(60)
zb <- gft(b \%*\% t(b) + diag(1 - b^2))
c(diagonal = inv_gft(zb)$hvs,
  quadrature = inv_gft(zb, preconditioner = "quadrature")$hvs)
}
\keyword{multivariate}
\keyword{optimize}
