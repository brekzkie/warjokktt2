logistic_regression <- function(X, y, tol = 1e-5, max_iter = 1000, method = "NR") {
  # Inisialisasi koefisien beta dengan nol
  beta <- rep(0, ncol(X))

  # Fungsi Likelihood
  log_likelihood <- function(beta, X, y) {
    p <- 1 / (1 + exp(-X %*% beta))
    sum(y * log(p) + (1 - y) * log(1 - p))
  }

  # Gradient Function
  gradient <- function(beta, X, y) {
    p <- 1 / (1 + exp(-X %*% beta))
    t(X) %*% (y - p)
  }

  # Hessian Matrix
  hessian <- function(beta, X) {
    p <- 1 / (1 + exp(-X %*% beta))
    W <- diag(as.vector(p * (1 - p)))
    t(X) %*% W %*% X
  }

  # Algoritma Newton-Raphson (NR) atau IWLS
  for (i in 1:max_iter) {
    p <- 1 / (1 + exp(-X %*% beta))
    W <- diag(as.vector(p * (1 - p)))

    if (method == "NR") {
      H <- hessian(beta, X)
      grad <- gradient(beta, X, y)
      beta_new <- beta - solve(H) %*% grad
    } else if (method == "IWLS") {
      z <- X %*% beta + solve(W) %*% (y - p)
      XTWX_inv <- solve(t(X) %*% W %*% X)
      beta_new <- XTWX_inv %*% (t(X) %*% W %*% z)
    } else {
      stop("Method not recognized. Use 'NR' or 'IWLS'.")
    }

    if (sqrt(sum((beta_new - beta)^2)) < tol) {
      cat("Converged in", i, "iterations\n")
      return(list(method = method, beta = beta_new, fit = p))
    }
    beta <- beta_new
  }

  warning("Maximum iterations reached without convergence")
  return(list(method = method, beta = beta, fit = p))
}

# Load library
library(dplyr)

# Contoh dataset
set.seed(123)
n <- 100
X1 <- rnorm(n)
X2 <- rnorm(n)
y <- ifelse(0.5 + 1.2 * X1 - 0.8 * X2 + rnorm(n) > 0, 1, 0)

# Menyiapkan matrix X (dengan intercept)
X <- cbind(1, X1, X2)

# Menjalankan regresi logistik dengan Newton-Raphson
result_IWLS <- logistic_regression(X, y, method = "IWLS")

# Menampilkan hasil
print(result_IWLS)
