# if (!require('devtools')) install.packages('devtools')
# devtools::install_github('fhernanb/stests', force=TRUE)
# install.packages("tidyverse")

library("tidyverse")
library("stests")

prices_mogd_linear <- read_csv("data/prices/prices_mogd_linear_high.csv") 
prices_ngd_linear <- read_csv("data/prices/prices_ngd_linear_high.csv")
prices_mogd_cd <- read_csv("data/prices/prices_mogd_cd_high.csv")
prices_ngd_cd <- read_csv("data/prices/prices_ngd_cd_high.csv")
prices_mogd_leontief <- read_csv("data/prices/prices_mogd_leontief_high.csv")
prices_ngd_leontief <- read_csv("data/prices/prices_ngd_leontief_high.csv")

# Function to perform a multivariate test for two mean vectors
one_simul <- function(delta, df1, df2) {
  X <- df1 %>% remove_rownames() %>%
    column_to_rownames(var = 1)
  Y <- df2 %>% remove_rownames() %>%
    column_to_rownames(var = 1)
  
  mean_1 <-  sapply(X,FUN=mean)
  cov_1 <- as.matrix(cov(X))
  n_1 <- dim(X)[1]
  mean_2 <-  sapply(Y,FUN=mean)
  
  cov_2 <- as.matrix(cov(Y))
  n_2 <- dim(Y)[1]
  
  # Use function two_mean_vector_test from stests package to perform the test
  res <- two_mean_vector_test(xbar1=mean_1, s1=cov_1, n1=n_1,
                              xbar2=mean_2, s2=cov_2, n2=n_2, delta0 = delta,
                              # method = 'james',
                              alpha=0.05)
  # Return the statistic and the critical value of the test
  # plot(res, from=0, to=10, shade.col="lightgreen")
  # return(cbind(res$statistic[1], res$statistic[2]))
  return(res$p.value)
}

plot_p_val <- function(X, Y) {
  delta <- seq(from=0, to=10, by=0.01)
  p_vals <- sapply(delta, FUN = one_simul, df1 = X, df2 = Y)
  return (plot(delta, p_vals))
}

delta <- 0.05
print(paste("Running First Order James Test to see if outputs differ on average by more than:", delta ))
print("Linear p-value:\n")
print(one_simul(delta, prices_mogd_linear, prices_ngd_linear))
print("Cobb-Douglas p-value:\n")
print(one_simul(delta, prices_mogd_cd, prices_ngd_cd))
print("Leontief p-value:\n")
print(one_simul(delta, prices_mogd_leontief, prices_ngd_leontief))



