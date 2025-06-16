source("Data/Postprocessing/load_data.R")

library(parallel)
library(abind)

compute_sim_matrix <- function(data, file_name) {
  
  # Load necessary data
  transformed_data <- load_transformed_data(file_name)
  # pp_out <- transformed_data$pp_out
  # pp_out_init <- transformed_data$pp_out_init
  # pp_all <- abind(pp_out_init, pp_out, along = 1)
  ensfc <- transformed_data$ensfc
  ensfc_init <- transformed_data$ensfc_init
  ensfc_all <- abind(ensfc_init, ensfc, along = 1)
  n <- dim(ensfc_all)[1]
  d <- dim(ensfc_all)[3]
  ensfc_statistics <- array(0, dim = c(n, d, 2))

  for (nn in 1:n)
  {
    for (dd in 1:d)
    {
      ensfc_statistics[nn, dd, 1] <-  mean(ensfc_all[nn, ,dd])
      ensfc_statistics[nn, dd, 2] <-  sd(ensfc_all[nn, ,dd])
    }
  }
  
  # Generate index pairs (upper triangle)
  indices <- combn(n, 2, simplify = FALSE)
  
  # Parallel computation of similarity scores
  sim_values <- mclapply(indices, function(idx) {
    Sim(idx[1], idx[2], ensfc_statistics)
  }, mc.cores = detectCores())
  
  sim_values <- unlist(sim_values)
  
  # Construct the similarity matrix
  sim_matrix <- matrix(0, n, n)
  idx_mat <- do.call(rbind, indices)

  # Assign values row-wise
  for (k in 1:(dim(idx_mat)[1])) {
    i <- idx_mat[k, 1]
    j <- idx_mat[k, 2]
    sim_matrix[i, j] <- sim_values[k]
  }

  sim_matrix <- sim_matrix + t(sim_matrix)

  # Diagonal has perfect similarity
  diag(sim_matrix) <- 1
  
  # Save results
  savename <- paste0("Data/SimilarityMatrix/simMatrix_", file_name, ".Rdata")
  save(sim_matrix, file = savename)
  
}



#####################################################################
#  Similarity criterion as in Schefzik 2016 (Sim Schaake) Formula (4)
#  Elisa Perrone
#  2021 - 05 - 25
####################################################################

Sim <- function(t,td, ensfc_statistics){
  
  Mean<-0
  Sd<-0

  # Loop over the dimensions
  d <- dim(ensfc_statistics)[2]
  
  for(dd in 1:d)
  {

    # Retrieve the ensemble statistics
    dat <- ensfc_statistics[,dd,]
    
    # Extract the variables
    xbar_t  <- dat[t, 1] #unlist(unname(dat[t, ]["ens_mu"]))
    s_t     <- dat[t, 2] #unlist(unname(dat[t, ]["ens_sd"]))
    
    xbar_td <- dat[td, 1] #unlist(unname(dat[td, ]["ens_mu"]))
    s_td    <- dat[td, 2] #unlist(unname(dat[td, ]["ens_sd"]))
    
    # Use the statistics to compute the criterion
    Mean <- Mean + (xbar_t - xbar_td)^2
    Sd   <- Sd   + (s_t - s_td)^2

  }
  
  z <- sqrt( (1/d) * sum(Mean) + (1/d) * sum(Sd)  )

  # New: apply transform as suggested in Schefzik from [0, infty) to (0, 1]
  # 0 means no similarity and 1 means perfect similarity
  z2 <- exp(-z)
  return(z2)
}