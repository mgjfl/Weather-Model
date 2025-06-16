source("Data/Postprocessing/load_data.R")

library(boostCopula)

compute_boostRVineMatrix <- function(file_name)
{
  ###########################
  ## Obtain the CDF values ##
  ###########################
  
  
  # The observations and EMOS output
  transformed_data  <- load_transformed_data(file_name)
  obs_init          <- transformed_data$obs_init
  postproc_out_init <- transformed_data$pp_out_init
  
  # Select days with non NA EMOS output (all days > UVPP training window)
  valid_days <- sapply(1:dim(postproc_out_init)[1], function(x) return(!any(is.na(postproc_out_init[x,,]))))
  
  # Select the valid observations and EMOS
  obs_valid   <- obs_init[valid_days,]
  pp_valid    <- postproc_out_init[valid_days,,]
  n_valid     <- sum(valid_days)
  
  # Obtain the CDF input values
  obs_latent_gaussian   <- (obs_valid - pp_valid[,,1]) / pp_valid[,,2]
  obs_train_CDF         <- pnorm(obs_latent_gaussian)
  
  # Get the data in the correct format
  U <- as.data.frame(obs_train_CDF)
  X <- as.data.frame(pp_valid)
  d <- dim(obs_init)[2]
  
  ##################################
  ## Fit a boostRVine to the data ##
  ##################################
  
  # Construct the RVineMatrix
  # Note: Vine structure in Matrix and families in Family are ignored!
  Matrix    <- matrix(0, d, d, byrow = TRUE)
  Family    <- matrix(0, nrow = d, ncol = d)
  Formula   <- matrix("~ .", d, d, byrow = TRUE)
  boostRVM  <- boostRVineMatrix(Matrix = Matrix,
                               family = Family,
                               formula = Formula)
  
  
  # Fit Rvine to full historical record
  rvine <- boostRVineStructureSelect(U = U,
                                      X = X,
                                      boostRVM = boostRVM,
                                      familyset = c(1, 301, 304, 401, 404),
                                      selectioncrit = "aic",
                                      treecrit = "tau",
                                      vine_type = 0,
                                      control = boost_control(maxit = 10000,
                                                              deselection = "none"),
                                      cores = 20)
  
  ##########################
  ## Save the fitted vine ##
  ##########################
  
  # Extract the boostRVineMatrix and save it
  boostRVM <- rvine$boostRVM
  save(boostRVM, file = paste0("Data/RVineMatrix/rvine_matrix_", file_name, ".RData"))
}
