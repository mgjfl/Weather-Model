
source("Data/Postprocessing/scores.R")
library(copula)
library(abind)
library(boostCopula)

mvpp <- function(method, 
                 transformed_data,
                 score.env,
                 trainingWindow = NULL,
                 variant = NULL, 
                 EMOS_sample = NULL,
                 MVPP_sample = NULL,
                 ECC_out = NULL,
                 saveScores = TRUE,
                 shuffle = FALSE,
                 mvsample_fallback = NULL,
                 output_dim = NULL,
                 sim_matrix = NULL,
                 boostRVM = NULL,
                 parallelization = TRUE,
                 addTrainingDays = FALSE) 
{

    # Whether to use Parallelization
    .GlobalEnv$parallelization <- parallelization

    method_name <- get_method_name(method, variant, shuffle)
    
    if (addTrainingDays)
    {
        method_name <- paste0(method_name, "-", trainingWindow)
    }
    cat("Starting ", method_name, "...\n")
  
    # Time function duration
    start_time <- Sys.time()
    
    # Ensure reproducibility
    set.seed(2025)
    
    # Extract necessary data
    ensfc             <- transformed_data$ensfc
    ensfc_init        <- transformed_data$ensfc_init
    obs               <- transformed_data$obs
    obs_init          <- transformed_data$obs_init
    postproc_out      <- transformed_data$pp_out
    postproc_out_init <- transformed_data$pp_out_init
    
    # generate array for ouput
    n             <- dim(ensfc)[1]
    m             <- dim(ensfc)[2]
    d             <- dim(ensfc)[3]
    params        <- array(NA, dim = n)
    chosenCopula  <- array(NA, dim = n)
    
    if (is.null(output_dim))
    {
        output_dim <- m
    }
    
    if (!shuffle)
    {
    res <- switch(
        method,
        "EMOS"            = mvpp_emos(variant, n, m, d, ensfc, postproc_out, output_dim),
        "ECC"             = mvpp_ecc(n, m, d, ensfc, EMOS_sample),
        "dECC"            = mvpp_decc(n, m, d, ensfc, postproc_out, obs_init, ensfc_init, EMOS_sample, ECC_out),
        "SSh-H"           = mvpp_ssh(n, m, d, obs_init, obs, EMOS_sample, output_dim),
        "SimSchaake"      = mvpp_sim_ssh(n, m, d, obs_init, obs, EMOS_sample, sim_matrix, output_dim),
        "SimSchaake-H"    = mvpp_sim_ssh_hist(n, m, d, obs_init, obs, EMOS_sample, sim_matrix, output_dim),
        "SSh-I14"         = mvpp_ssh_interval14(n, m, d, obs_init, obs, EMOS_sample, output_dim),
        "GCA"             = mvpp_gca(n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim),
        "SimGCA"          = mvpp_gca(n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix),
        "CopGCA"          = mvpp_cop_gca(n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim),
        "SimCopGCA"       = mvpp_cop_gca(n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix),
        "Clayton"         = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim), 
        "SimClayton"      = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix), 
        "Frank"           = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim), 
        "SimFrank"        = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix), 
        "Gumbel"          = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim), 
        "SimGumbel"       = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix), 
        "Surv_Gumbel"     = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim), 
        "SimSurv_Gumbel"  = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix), 
        "GOF"             = mvpp_arch(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim),
        "boost_cop"       = mvpp_boost_cop(n, m, d, obs_init, obs, postproc_out_init, postproc_out, mvsample_fallback, trainingWindow, output_dim),
        "boost_cop_fixed" = mvpp_boost_cop_fixed(n, m, d, obs_init, obs, postproc_out_init, postproc_out, mvsample_fallback, trainingWindow, output_dim, boostRVM),
    )
    } else 
    {
        res <- mvpp_shuffle(n, m, d, EMOS_sample, MVPP_sample)
    }
    
    score.env$lastResult <- res
    
    if (saveScores)
    {
        add_scores(score.env, res, method_name, start_time)
        cat("Saving ", method_name, "...\n")
    }
    
    
    return(res) #(list("mvppout" = mvppout, "params" = params, "chosenCopula" = chosenCopula))
    # in random methods: distinguish cases with and without given EMOS_sample, maybe only handle that with sample at first, rest can be included later on
    # if no sample is given, a new one has to be generated, as done for the EMOS methods themselves
}


mvpp_emos <- function(variant, n, m, d, ensfc, postproc_out, output_dim)
{
  mvppout <- array(NA, dim = c(n, output_dim, d)) 
  
  if (variant == "R") {
    for (nn in 1:n) {
      for (dd in 1:d) {
        par <- postproc_out[nn, dd, ]
        mvppout[nn, , dd] <- rnorm(output_dim, mean = par[1], sd = par[2])
      }
    }
  } else if (variant == "Q") {
    qlevels <- 1:output_dim / (output_dim + 1)
    for (nn in 1:n) {
      for (dd in 1:d) {
        par <- postproc_out[nn, dd, ]
        mvppout[nn, , dd] <- qnorm(qlevels, mean = par[1], sd = par[2])
      }
    }
  } else if (variant == "QO") {
    qlevels <- (1:output_dim - 0.5) / output_dim
    for (nn in 1:n) {
      for (dd in 1:d) {
        par <- postproc_out[nn, dd, ]
        mvppout[nn, , dd] <- qnorm(qlevels, mean = par[1], sd = par[2])
      }
    }
  } else if (variant == "S") {
    breakpoints <- 0:output_dim / output_dim
    qlevels <- runif(output_dim, min = breakpoints[1:output_dim], max = breakpoints[2:(output_dim + 1)])
    for (nn in 1:n) {
      for (dd in 1:d) {
        par <- postproc_out[nn, dd, ]
        mvppout[nn, , dd] <- qnorm(qlevels, mean = par[1], sd = par[2])
      }
    }
  } else if (variant == "T") {
    for (nn in 1:n) {
      for (dd in 1:d) {
        ensfc_tmp <- ensfc[nn, , dd]
        ens_par <- c(mean(ensfc_tmp), sd(ensfc_tmp))
        qlevels <- pnorm(ensfc_tmp, mean = ens_par[1], sd = ens_par[2])
        postproc_par <- postproc_out[nn, dd, ]
        mvppout[nn, , dd] <- qnorm(qlevels, mean = postproc_par[1], sd = postproc_par[2])
      }
    }
  }
  
  return(list("mvppout" = mvppout))
}

mvpp_ecc <- function(n, m, d, ensfc, EMOS_sample)
{
  mvppout <- array(NA, dim = c(n, m, d)) 
  
  # application of ECC is independent of 'variant' parameter, dependency is only through EMOS_sample
  for (nn in 1:n) {
    for (dd in 1:d) {
      ensfc_tmp <- ensfc[nn, , dd]
      mvppout[nn, , dd] <- EMOS_sample[nn, , dd][rank(ensfc_tmp, ties.method = "random")]
    }
  }
  
  return(list("mvppout" = mvppout))
}

mvpp_decc <- function(n, m, d, ensfc, postproc_out, obs_init, ensfc_init, EMOS_sample, ECC_out)
{
  mvppout <- array(NA, dim = c(n, m, d)) 
  
  # e in the dECC paper, eq (14)
  fcerror <- obs_init - apply(ensfc_init, c(1, 3), mean)
  # compute error correlation matrix and root
  R_e <- cor(fcerror)
  eig <- eigen(R_e)
  Re_root <- eig$vectors %*% diag(sqrt(eig$values)) %*% t(eig$vectors)
  
  # c in the dECC paper, eq (18)
  error_cor <- ECC_out - ensfc
  
  # breve c, eq (19)
  breve_c <- array(NA, dim = dim(ensfc))
  for (nn in 1:n) {
    for (mm in 1:m) {
      breve_c[nn, mm, ] <- Re_root %*% error_cor[nn, mm, ]
    }
  }
  
  # adjusted ensemble breve x, eq (22)
  breve_x <- ensfc + breve_c
  
  # apply ECC with breve_x as dependence template (Step 5 in paper)
  for (nn in 1:n) {
    for (dd in 1:d) {
      ensfc_tmp <- breve_x[nn, , dd]
      mvppout[nn, , dd] <- EMOS_sample[nn, , dd][rank(ensfc_tmp, ties.method = "random")]
    }
  }
  
  return(list("mvppout" = mvppout))
}

mvpp_ssh <- function(n, m, d, obs_init, obs, EMOS_sample, output_dim)
{
  mvppout <- array(NA, dim = c(n, output_dim, d)) 
  
  # concatenate obs_init and obs arrays to sample from available forecast cases later on
  obs_all <- rbind(obs_init, obs)
  
  # reorder post-processed forecast sample according to past observations
  for (nn in 1:n) {
    # choose set of past forecast cases to determine dependence template
    #   ... this way, a new set of IDs is drawn for every forecast instance
    #   ... this needs to depend on nn in a more suitable manner if there is temporal change in the simulation setup
    obs_IDs <- sample(x = 1:(dim(obs_init)[1] + nn - 1), size = output_dim, replace = FALSE)
    for (dd in 1:d) {
      obs_tmp <- obs_all[obs_IDs, dd]
      mvppout[nn, , dd] <- EMOS_sample[nn, , dd][rank(obs_tmp, ties.method = "random")]
    }
  }
  
  return(list("mvppout" = mvppout))
}

mvpp_sim_ssh <- function(n, m, d, obs_init, obs, EMOS_sample, sim_matrix, output_dim)
{
  mvppout <- array(NA, dim = c(n, output_dim, d)) 
  
  # concatenate obs_init and obs arrays to sample from available forecast cases later on
  obs_all <- rbind(obs_init, obs)
  
  # reorder post-processed forecast sample according to past observations
  for (nn in 1:n) {
    # choose set of past forecast cases to determine dependence template
    #   ... this way, a new set of IDs is drawn for every forecast instance
    #   ... this needs to depend on nn in a more suitable manner if there is temporal change in the simulation setup

    # Compute the index in the sim_matrix
    t <- nn + dim(obs_init)[1]
    
    # Retrieve similarity scores
    # NOTE: first (UVPP training window) values are NAN, but are sorted last. No problem since length historical record > 2 * UVPP training window
    sim_values <- sim_matrix[t, ]
    
    # Retrieve the m most similar days
    sorted_indices <- order(sim_values, decreasing = TRUE)
    sorted_indices <- sorted_indices[sorted_indices != t]
    sorted_indices  <- sorted_indices[sorted_indices > trainingWindow] # First `trainingWindow` values from pp_out are NA
    obs_IDs <- sorted_indices[1:output_dim]
    
    for (dd in 1:d) 
    {
      obs_tmp <- obs_all[obs_IDs, dd]
      mvppout[nn, , dd] <- EMOS_sample[nn, , dd][rank(obs_tmp, ties.method = "random")]
    }
  }
  
  return(list("mvppout" = mvppout))
}

mvpp_sim_ssh_hist <- function(n, m, d, obs_init, obs, EMOS_sample, sim_matrix, output_dim)
{
  mvppout <- array(NA, dim = c(n, output_dim, d)) 
  
  # concatenate obs_init and obs arrays to sample from available forecast cases later on
  obs_all <- rbind(obs_init, obs)
  
  # reorder post-processed forecast sample according to past observations
  for (nn in 1:n) {
    # choose set of past forecast cases to determine dependence template
    #   ... this way, a new set of IDs is drawn for every forecast instance
    #   ... this needs to depend on nn in a more suitable manner if there is temporal change in the simulation setup

    # Compute the index in the sim_matrix
    t <- nn + dim(obs_init)[1]
    
    # Retrieve similarity scores
    # NOTE: first (UVPP training window) values are NAN, but are sorted last. No problem since length historical record > 2 * UVPP training window
    sim_values <- sim_matrix[t, ]
    
    # Retrieve the m most similar days
    sorted_indices <- order(sim_values, decreasing = TRUE)
    sorted_indices <- sorted_indices[sorted_indices < t]
    sorted_indices  <- sorted_indices[sorted_indices > trainingWindow] # First `trainingWindow` values from pp_out are NA
    obs_IDs <- sorted_indices[1:output_dim]
    
    for (dd in 1:d) 
    {
      obs_tmp <- obs_all[obs_IDs, dd]
      mvppout[nn, , dd] <- EMOS_sample[nn, , dd][rank(obs_tmp, ties.method = "random")]
    }
  }
  
  return(list("mvppout" = mvppout))
}

mvpp_ssh_interval14 <- function(n, m, d, obs_init, obs, EMOS_sample, output_dim)
{
  mvppout <- array(NA, dim = c(n, output_dim, d)) 
  
  # concatenate obs_init and obs arrays to sample from available forecast cases later on
  obs_all <- rbind(obs_init, obs)
  
  # reorder post-processed forecast sample according to past observations
  for (nn in 1:n) {
    # choose set of past forecast cases to determine dependence template
    #   ... this way, a new set of IDs is drawn for every forecast instance
    #   ... this needs to depend on nn in a more suitable manner if there is temporal change in the simulation setup
    obs_IDs <- sample(x = getIntervals(nn, 1, dim(obs_all)[1], 28), size = output_dim, replace = FALSE)
    for (dd in 1:d) {
      obs_tmp <- obs_all[obs_IDs, dd]
      mvppout[nn, , dd] <- EMOS_sample[nn, , dd][rank(obs_tmp, ties.method = "random")]
    }
  }
  
  return(list("mvppout" = mvppout))
}

random_clip <- function(x, delta) {
  # Clips values to be in (delta/2, 1-delta/2) while ensuring no systematic ties that lead to numerical problems
  
  # Generate random values for extreme regions
  lower_extreme <- runif(length(x), min = delta / 2, max = delta)
  upper_extreme <- runif(length(x), min = 1 - delta, max = 1 - delta / 2)
  
  # Apply random reassignment only for extreme values
  x_new <- ifelse(x < delta, lower_extreme, ifelse(x > 1 - delta, upper_extreme, x))
  
  return(x_new)
}

clean_futures <- function(gc_repeats = 2, sleep_sec = 0.5, verbose = TRUE) {
  # Switch to sequential plan to shut down any workers
  plan(sequential)
  if (verbose) message("Future plan set to sequential.")

  # Run garbage collection multiple times to ensure release
  for (i in seq_len(gc_repeats)) {
    invisible(gc())
    Sys.sleep(sleep_sec)
  }

  if (verbose) message("Background workers terminated and memory cleaned up.")
}

mvpp_latent_obs <- function(n, m, d, obs_init, obs, postproc_out_init, postproc_out, function_on_latent_obs, trainingWindow, output_dim, sim_matrix = NULL, workers = NULL)
{
    library(future)
    library(future.apply)
    library(progressr)

    if (is.null(workers))
    {
        workers <- parallel::detectCores() - 2
    }

    if (parallelization)
    {
        plan(multisession, workers = workers) 
    } else 
    {
        plan(sequential)
    }
    options(future.globals.maxSize = +Inf)
    handlers(global = TRUE)
    progressr::handlers("cli") 

    
    temp_env <- new.env()
    temp_env$mvppout <- array(NA, dim = c(n, output_dim, d)) 
    temp_env$chosenCopula <- array(NA, dim = n)
    temp_env$params <- array(NA, dim = n)
    
    # concatenate obs_init and obs arrays to determine covariance matrix for Gaussian copulas
    obs_all <- rbind(obs_init, obs)
    
    # Concatenate UVPP output for boostCopula
    pp_all <- abind(postproc_out_init, postproc_out, along = 1)
    
    with_progress({
        p <- progressor(steps = n + 5) # +5 as buffer for late updates
        # Do all computations in parallel
        results <- future_lapply(1:n, future.seed = TRUE, function(nn) {

        
        p(sprintf("Day %d", nn))

        # Retrieve the training IDs
        if (is.null(sim_matrix))
        { # Past `trainingWindow` days
            train_IDs <- (dim(obs_init)[1] + nn - trainingWindow):(dim(obs_init)[1] + nn - 1)
        } else 
        { # Most similar `trainingWindow` days
            t           <- nn + dim(obs_init)[1]
            sim_values  <- sim_matrix[t, ]
            
            # Retrieve the m most similar days
            sorted_indices  <- order(sim_values, decreasing = TRUE)
            sorted_indices  <- sorted_indices[sorted_indices > trainingWindow] # First `trainingWindow` values from pp_out are NA
            train_IDs       <- sorted_indices[1:output_dim]
        }

        # Only last measurements
        obs_train <- obs_all[train_IDs, ]
        pp_train  <- pp_all[train_IDs, ,]
        pp_test   <- postproc_out[nn,,]

        # Latent Gaussian observations
        obs_latent_gaussian   <- (obs_train - pp_train[,,1]) / pp_train[,,2]
        obs_train_CDF         <- pnorm(obs_latent_gaussian)

        out <-   function_on_latent_obs(
                    obs_latent_gaussian = obs_latent_gaussian,
                    obs_train_CDF = obs_train_CDF,
                    mean_values = pp_test[,1],
                    sd_values = pp_test[,2],
                    pp_train = pp_train,
                    pp_test = pp_test,
                    nn = nn
                    )

        return(out)
        
        })
    })

    # Collect all results
    for (nn in 1:n) {
        temp_env$mvppout[nn,,] <- results[[nn]]$mvppout
        if (!is.null(results[[nn]]$chosenCopula))
        {
        temp_env$chosenCopula[nn] <- results[[nn]]$chosenCopula
        }
        if (!is.null(results[[nn]]$params))
        {
        temp_env$params[nn] <- results[[nn]]$params
        }
    }

    if (parallelization)
    {
        # Clean up resources
        clean_futures()
    }
  
  return(list("mvppout" = temp_env$mvppout, "chosenCopula" = temp_env$chosenCopula, "params" = temp_env$params))
}

mvpp_shuffle <- function(n, m, d, EMOS_sample, MVPP_sample)
{
  mvppout <- array(NA, dim = dim(EMOS_sample)) 
    
    for (nn in 1:n)
    {
      # impose dependence structure on post-processed forecasts
      for (dd in 1:d) 
      {
        obs_tmp <- MVPP_sample[nn, ,dd]
        mvppout[nn, , dd] <- EMOS_sample[nn, , dd][rank(obs_tmp, ties.method = "random")]
      }
    } 
  
  return(list("mvppout" = mvppout))

}

mvpp_gca <- function(n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix = NULL)
{
  require(MASS)

  function_on_latent_obs <- function(obs_latent_gaussian, obs_train_CDF, mean_values, sd_values, pp_train, pp_test, nn)
  {
    # estimate covariance matrix
    cov_obs <- cov(obs_latent_gaussian)
    # draw random sample from multivariate normal distribution with this covariance matrix
    # Make sure to get numeric values
    
    # Dependence structure by Copula
    mvsample <- mvrnorm(n = output_dim, mu = rep(0, d), Sigma = cov_obs)
    

    # impose dependence structure on post-processed forecasts
    # for (dd in 1:d)
    # {
    #   temp_env$mvppout[nn, , dd] <- mvsample[, dd] * sd_values[dd] + mean_values[dd]
    # }
    return(list(
      mvppout = mvsample,
      chosenCopula = NULL,
      params = NULL 
    ))

  }
  
  
  res <- mvpp_latent_obs(n, m, d, obs_init, obs, postproc_out_init, postproc_out, function_on_latent_obs, trainingWindow, output_dim, sim_matrix)
  
  return(list("mvppout" = res$mvppout))

}

mvpp_cop_gca <- function(n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix = NULL)
{
  require(MASS)
  

  function_on_latent_obs <- function(obs_latent_gaussian, obs_train_CDF, mean_values, sd_values, pp_train, pp_test, nn)
  {

    fitcop <- tryCatch(
      {
        fitCopula(normalCopula(dim = d, dispstr = "un"), data = obs_train_CDF, method = "itau")
      },
      error = function(e) {
        print(e)
        indepCopula(dim = d)
      }
    )
    
    if (!(class(fitcop) == "indepCopula")) {
      cop <- fitcop@copula
    } else {
      cop <- fitcop
    }
    

    # draw random sample from multivariate normal distribution with this estimated copula
    paramMargins <- list()
    surv_paramMargins <- list()
    
    for (i in 1:d) {
      paramMargins[[i]] <- list(mean = mean_values[i], sd = sd_values[i])
    }
    
    mvDistribution <- mvdc(
      copula = cop, margins = rep("norm", d),
      paramMargins = paramMargins
    )
    
    mvsample <- rMvdc(output_dim, mvDistribution)
    
    # for (dd in 1:d) 
    # {
    #   temp_env$mvppout[nn, , dd] <- mvsample[, dd]
    # }

    return(list(
      mvppout = mvsample,
      chosenCopula = NULL,
      params = NULL 
    ))
      

  }
  
  res <- mvpp_latent_obs(n, m, d, obs_init, obs, postproc_out_init, postproc_out, function_on_latent_obs, trainingWindow, output_dim, sim_matrix)
  
  return(list("mvppout" = res$mvppout))
}

mvpp_boost_cop_fixed <- function(n, m, d, obs_init, obs, postproc_out_init, postproc_out, mvsample_fallback, trainingWindow, output_dim, boostRVM)
{
  
  function_on_latent_obs <- function(obs_latent_gaussian, obs_train_CDF, mean_values, sd_values, pp_train, pp_test, nn)
  {
    # cat(paste0(format(Sys.time(), "%H:%M:%S"), " | Day ", nn))
    
    U1 <- as.data.frame(obs_train_CDF)
    X1 <- as.data.frame(pp_train)
    
    # Fit Rvine to past
    
    rvine <- tryCatch(
      {
        boostRVineSeqEst(U = U1,
                          X = X1,
                          boostRVM = boostRVM,
                          control = boost_control(maxit = 100,
                                                  deselection = "none"),
                          cores = 20)
      },
      error = function(e) {
        # print(e)
        indepCopula(dim = d)
      }
    )
    
    # Check if fit was successful
    if (!(class(rvine) == "indepCopula")) {
      chosenCopula <- "boost_copula"
      
      U2 <- matrix(runif(output_dim * d, min = 0, max = 1), nrow = output_dim, ncol = d)
      X2 <- data.frame(matrix(rep(as.vector(transformed_data$pp_out[nn, ,]), output_dim), 
                              nrow = output_dim, 
                              byrow = TRUE))
      
      names(X2) <- names(X1)
      
      # Simulate
      sim_data <- boostRVineSim(rvine,
                                  U = U2,
                                  X = X2)

      # To standard normal
      mvsample <- qnorm(sim_data, 
                  mean = transformed_data$pp_out[nn, ,1], 
                  sd = transformed_data$pp_out[nn, ,2])
      
    } else {
      chosenCopula <- "fallback"

      
      mvsample <- mvsample_fallback$mvppout[nn,,]
    }

    
    # cat(paste0(" uses copula: ", temp_env$chosenCopula[nn], "\n"))
        
    # for (dd in 1:d) 
    # {
    #   temp_env$mvppout[nn, , dd] <- mvsample[, dd]
    # }

    return(list(
      mvppout = mvsample,
      chosenCopula = chosenCopula,
      params = NULL 
    ))
    
  }
  
  res <- mvpp_latent_obs(n, m, d, obs_init, obs, postproc_out_init, postproc_out, function_on_latent_obs, trainingWindow, output_dim)
  
  return(res)
  
}

mvpp_boost_cop <- function(n, m, d, obs_init, obs, postproc_out_init, postproc_out, mvsample_fallback, trainingWindow, output_dim)
{
  
  function_on_latent_obs <- function(obs_latent_gaussian, obs_train_CDF, mean_values, sd_values, pp_train, pp_test, nn)
  {
    # cat(paste0(format(Sys.time(), "%H:%M:%S"), " | Day ", nn))
    
    # create boostRVineMatrix
    # NOTE: Matrix and Family are ignored in boostRVineStructureSelect
    Matrix <- matrix(0, d, d, byrow = TRUE)
    Family <- matrix(0, nrow = d, ncol = d)
    Formula <- matrix("~ .", d, d, byrow = TRUE)
    boostRVM <- boostRVineMatrix(Matrix = Matrix,
                                 family = Family,
                                 formula = Formula)
    
    U1 <- as.data.frame(obs_train_CDF)
    X1 <- as.data.frame(pp_train)
    
    # Fit Rvine to past
    
    rvine <- tryCatch(
      {
        boostRVineStructureSelect(U = U1,
                                  X = X1,
                                  boostRVM = boostRVM,
                                  familyset = c(1, 301, 304, 401, 404),
                                  selectioncrit = "aic",
                                  treecrit = "tau",
                                  vine_type = 0,
                                  control = boost_control(maxit = 100,
                                                          deselection = "aic"),
                                  cores = 20)
      },
      error = function(e) {
        # print(e)
        indepCopula(dim = d)
      }
    )
    
    # Check if fit was successful
    if (!(class(rvine) == "indepCopula")) {
      chosenCopula <- "boost_copula"
      
      U2 <- matrix(runif(output_dim * d, min = 0, max = 1), nrow = output_dim, ncol = d)
      X2 <- data.frame(matrix(rep(as.vector(transformed_data$pp_out[nn, ,]), output_dim), 
                              nrow = output_dim, 
                              byrow = TRUE))
      
      names(X2) <- names(X1)
      
      # Simulate
      sim_data <- boostRVineSim(rvine,
                                  U = U2,
                                  X = X2)

      # To standard normal
      mvsample <- qnorm(sim_data, 
                  mean = transformed_data$pp_out[nn, ,1], 
                  sd = transformed_data$pp_out[nn, ,2])
      
    } else {
      chosenCopula <- "fallback"

      
      mvsample <- mvsample_fallback$mvppout[nn,,]
    }

    
    # cat(paste0(" uses copula: ", temp_env$chosenCopula[nn], "\n"))
        
    # for (dd in 1:d) 
    # {
    #   temp_env$mvppout[nn, , dd] <- mvsample[, dd]
    # }

      return(list(
        mvppout = mvsample,
        chosenCopula = chosenCopula,
        params = NULL 
      ))
    
  }
  
  res <- mvpp_latent_obs(n, m, d, obs_init, obs, postproc_out_init, postproc_out, function_on_latent_obs, trainingWindow, output_dim)
  
  return(res)
  
}

mvpp_arch <- function(method, n, m, d, obs_init, obs, postproc_out_init, postproc_out, EMOS_sample, trainingWindow, output_dim, sim_matrix = NULL)
{
  require(MASS)

  if (!is.null(sim_matrix))
  { # Chop of Sim part
      method <- substring(method, 4)
  }
  
  function_on_latent_obs <- function(obs_latent_gaussian, obs_train_CDF, mean_values, sd_values, pp_train, pp_test, nn)
  {
    
    
    # cat("Day ", nn, "\n")
    
    fitMethod <- "itau" # "itau"
    maxIterations <- 5000
    # Estimate the parameter and copula - itau method does not converge for some cases (without giving a warning/ error and runs indefinitely)
    # Copula parameters are bounded to prevent generating inf samples
    if (method == "Clayton") {
      fitcop <- tryCatch(
        {
          fitCopula(claytonCopula(dim = d), data = obs_train_CDF, method = fitMethod, optim.control = list(maxit=maxIterations))
        },
        error = function(e) {
          # print(e)
          indepCopula(dim = d)
        }
      )
    } else if (method == "Frank") {
      fitcop <- tryCatch(
        {
          fitCopula(frankCopula(dim = d), data = obs_train_CDF, method = fitMethod, optim.control = list(maxit=maxIterations))
        },
        error = function(e) {
          # print(e)
          indepCopula(dim = d)
        }
      )
    } else if (method == "Gumbel") {
      fitcop <- tryCatch(
        {
          fitCopula(gumbelCopula(dim = d), data = obs_train_CDF, method = fitMethod, optim.control = list(maxit=maxIterations))
        },
        error = function(e) {
          # print(e)
          indepCopula(dim = d)
        }
      )
    } else if (method == "Surv_Gumbel") {
      fitcop <- tryCatch(
        {
          fitCopula(gumbelCopula(dim = d), data = 1 - obs_train_CDF, method = fitMethod, optim.control = list(maxit=maxIterations))
        },
        error = function(e) {
          # print(e)
          indepCopula(dim = d)
        }
      )
    } else {
      stop("Incorrect copula")
    }
    

    
    # Check if valid parameters have been generated
    if (!(class(fitcop) == "indepCopula")) {
      param <- fitcop@estimate

      if ((method == "Clayton" && param < 0)  ||
          (method == "Frank" && param < 0)    ||
          (method == "Gumbel" && param < 1)   ||
          (method == "Surv_Gumbel" && param < 1))
      {
        fitcop <- indepCopula(dim = d)
      }

    }
    
    
    if (!(class(fitcop) == "indepCopula")) {
      cop <- fitcop@copula
      chosenCopula <- method
      
      # Save the fitted parameter
      params <- fitcop@estimate
    } else {
      chosenCopula <- "indep"
      cop <- fitcop
      params <- NULL
    }
    

    # draw random sample from multivariate normal distribution with this estimated copula
    paramMargins <- list()
    surv_paramMargins <- list()
    
    for (i in 1:d) {
      paramMargins[[i]] <- list(mean = mean_values[i], sd = sd_values[i])
      surv_paramMargins[[i]] <- list(mean = -mean_values[i], sd = sd_values[i])
    }
    
    # Generate observations with normal marginals
    if (method == "Surv_Gumbel") {
      mvDistribution <- mvdc(
        copula = cop, margins = rep("surv_norm", d),
        paramMargins = surv_paramMargins
      )
    } else {
      mvDistribution <- mvdc(
        copula = cop, margins = rep("norm", d),
        paramMargins = paramMargins
      )
    }
    
    mvsample <- rMvdc(output_dim, mvDistribution)

    
    # for (dd in 1:d) {
    #   # Translate CDF values back to forecasts
    #   temp_env$mvppout[nn, , dd] <- mvsample[, dd]
    # }

    return(list(
        mvppout = mvsample,
        chosenCopula = chosenCopula,
        params = params 
      ))

  }
  
  return(mvpp_latent_obs(n, m, d, obs_init, obs, postproc_out_init, postproc_out, function_on_latent_obs, trainingWindow, output_dim, sim_matrix))
}

psurv_norm <- function(x, mean = 0, sd = 1) {
  return(1 - pnorm(x, mean = mean, sd = sd))
}

qsurv_norm <- function(x, mean = 0, sd = 1) {
  return(1 - qnorm(x, mean = mean, sd = sd))
}

dsurv_norm <- function(x, mean = 0, sd = 1) {
  return(-dnorm(x, mean = mean, sd = sd))
}

getCentralDays <- function(value, minRange, maxRange) {
  DAYS_PER_YEAR <- 365
  
  # Find smallest value
  smallest_value <- as.integer(value - DAYS_PER_YEAR * floor((value - minRange) / DAYS_PER_YEAR))
  
  # Find largest value
  largest_value <- as.integer(value + DAYS_PER_YEAR * floor((maxRange - value) / DAYS_PER_YEAR))
  
  return(seq(smallest_value, largest_value, by = DAYS_PER_YEAR))
}

getIntervals <- function(day, minRange, maxRange, interval_length) {
  central_days <- getCentralDays(day, minRange, maxRange)
  
  intervals <- unlist(lapply(central_days, function(x) {
    seq(max(minRange, x - interval_length), min(maxRange, x + interval_length))
  }))
  
  return(intervals)
}
