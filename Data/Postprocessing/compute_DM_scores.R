library(forecast) # for DM test function
library(future)
library(future.apply)
library(progressr)

source("Data/Postprocessing/load_data.R")

compute_DM_scores <- function(file_name, benchmarks, parallelization = TRUE)
{
    # Load all variables
    res <- new.env()
    load_score_env(file_name, res)

    n_bootstrap_samples <- 100
    all_dfmc <- data.frame()

    if (parallelization)
    {
        # Set up parallel plan
        plan(multisession, workers = parallel::detectCores())   
        options(future.globals.maxSize = +Inf)
    } else {
       plan(sequential)
    }

    # Enable progress bar
    handlers(global = TRUE)
    handlers("cli")  # CLI progress bar
    options(future.globals.maxSize = 2 * 1024^3)

    for (benchmark in benchmarks)
    {
        cat("Computing for", benchmark, "\n")

        # Parallelized computation with progress bar
        data_frames <- with_progress({
        p <- progressor(along = 1:(n_bootstrap_samples + 1))

        future_lapply(1:(n_bootstrap_samples + 1), function(i) {
            if (i <= n_bootstrap_samples) {
            df <- compute_dfmc(res, benchmark, i, bootstrap = TRUE)
            } else {
            df <- compute_dfmc(res, benchmark, bootstrap = FALSE)
            }
            p()
            df
        })
        })

        # Merge all data frames
        dfmc <- Reduce(function(x, y) merge(x, y, by = c("model", "score")), data_frames)
        dfmc$benchmark <- benchmark

        # Append to all_dfmc
        all_dfmc <- rbind(all_dfmc, dfmc)
    }

    save(all_dfmc, file = paste0("Data/TestStatistic/dm_statistics_", file_name, ".Rdata"))
}

compute_dfmc <- function(res, benchmark, count = 0, bootstrap = TRUE)
{
  
  # Models with scores
  input_models <- names(res$es_list)
  input_models <- input_models[! input_models %in% c(benchmark, "EMOS")]
  
  # Score names
  input_scores <- load_scores(res)
  
  
  set.seed(count)
  
  # Bootstrap the results
  nout            <- res$nout
  if (bootstrap)
  {
    sample_indices  <- sample(1:nout, size = nout, replace = TRUE)
    column_name     <- paste0("bootstrap_", count)
  } else {
    sample_indices  <- 1:nout
    column_name     <- paste0("full")
  }
  
  dfmc            <- expand.grid(input_models, input_scores, NA)
  names(dfmc)     <- c("model", "score", column_name)
  dm_power        <- 1
  varestimator    <- "bartlett"

  
  for(this_model in input_models){
    for(this_score in input_scores){
      ind <- which(dfmc$model == this_model & dfmc$score == this_score)  
      
      # deal with CRPS specifically (use only first dimension, not all 5 recorded ones)
      
      # print(paste0("( ", this_model, " , ", this_score, ")"))
      if(grepl("[0-9]+$", this_score))
      {
        
        n <- as.numeric(gsub("[^0-9]", "", this_score))

        e1 <- res[["crps_list"]][[this_model]][, n][sample_indices]
        e2 <- res[["crps_list"]][[benchmark]][, n][sample_indices]

        
        
        tmp <- NA
        if (sum(abs(e1 - e2)) != 0)
        {
          tryDM <- try(tmp_DM <- dm.test(e1 = e1,
                                       e2 = e2,
                                       h = 1, power = dm_power, varestimator = varestimator), silent = F)
          tmp <- tmp_DM$statistic
        } else{
          tmp <- 0
        }
        dm_teststat_vec <- tmp
      } else if(this_score == "crps_list"){

        
        e1 <- res[[this_score]][[this_model]][sample_indices]
        e2 <- res[[this_score]][[benchmark]][sample_indices]


        
        tmp <- NA

        if (sum(abs(e1 - e2)) != 0)
        {
          tryDM <- try(tmp_DM <- dm.test(e1 = e1,
                                       e2 = e2,
                                       h = 1, power = dm_power, varestimator = varestimator), silent = F)
          tmp <- tmp_DM$statistic
        } else{
          tmp <- 0
        }
        dm_teststat_vec <- tmp
      } else{
        tmp <- NA
        tryDM <- try(tmp_DM <- dm.test(e1 = res[[this_score]][[this_model]][sample_indices],
                                       e2 = res[[this_score]][[benchmark]][sample_indices],
                                       h = 1, power = dm_power, varestimator = varestimator), silent = F)
        if(class(tryDM) != "try-error"){
          tmp <- tmp_DM$statistic
        } else{
          tmp <- 0
        }
        dm_teststat_vec <- tmp
      }    
      
      dfmc[[column_name]][ind] <- dm_teststat_vec
    }
  }
  
  return(as.data.frame(dfmc))
}


compute_DM_scores_two_forecasts <- function(res_benchmark, res_comparison, savename, parallelization = TRUE)
{

    n_bootstrap_samples <- 100
    all_dfmc <- data.frame()

    if (parallelization)
    {
        # Set up parallel plan
        plan(multisession, workers = parallel::detectCores())   
        options(future.globals.maxSize = +Inf)
    } else {
       plan(sequential)
    }

    # Enable progress bar
    handlers(global = TRUE)
    handlers("cli")  # CLI progress bar
    options(future.globals.maxSize = 2 * 1024^3)

    # Separate benchmark for each in res
    methods <- names(res_benchmark$mvpp_list)
    methods <- methods[! grepl("^EMOS", methods)]

    for (method in methods)
    {
        cat("Computing for", method, "\n")

        # Parallelized computation with progress bar
        data_frames <- with_progress({
        p <- progressor(along = 1:(n_bootstrap_samples + 1))

        future_lapply(1:(n_bootstrap_samples + 1), function(i) {
            if (i <= n_bootstrap_samples) {
            df <- compute_dfmc_two_forecasts(res_benchmark, res_comparison, method, i, bootstrap = TRUE)
            } else {
            df <- compute_dfmc_two_forecasts(res_benchmark, res_comparison, method, bootstrap = FALSE)
            }
            p()
            df
        })
        })

        # Merge all data frames
        dfmc <- Reduce(function(x, y) merge(x, y, by = c("model", "score")), data_frames)
        dfmc$method <- method

        # Append to all_dfmc
        all_dfmc <- rbind(all_dfmc, dfmc)
    }

    save(all_dfmc, file = paste0("Data/TestStatistic/dm_statistics_comparison_", savename, ".Rdata"))
    return(all_dfmc)
}

compute_dfmc_two_forecasts <- function(res_benchmark, res_comparison, method, count = 0, bootstrap = TRUE)
{
  
    # Score names
    input_scores <- load_scores(res_benchmark)
    
    set.seed(count)
    
    # Bootstrap the results
    nout            <- res_benchmark$nout
    nout_forecast   <- res_comparison$nout
    diff            <- nout - nout_forecast
    if (bootstrap)
    {
        sample_indices  <- sample(1:nout_forecast, size = nout_forecast, replace = TRUE)
        column_name     <- paste0("bootstrap_", count)
    } else {
        sample_indices  <- 1:nout_forecast
        column_name     <- paste0("full")
    }
    
    dfmc            <- expand.grid(method, input_scores, NA)
    names(dfmc)     <- c("model", "score", column_name)
    dm_power        <- 1
    varestimator    <- "bartlett"

    # Method to compare against
    if (length(names(res_comparison$es_list)) == 1)
    {
        forecast_method <- names(res_comparison$es_list)[1]
    } else 
    {
        forecast_method <- method
    }

  
    for(this_score in input_scores){
        ind <- which(dfmc$model == method & dfmc$score == this_score)  
        
        # deal with CRPS specifically (use only first dimension, not all 5 recorded ones)
        
        # print(paste0("( ", this_model, " , ", this_score, ")"))
        if(grepl("[0-9]+$", this_score))
        {
        
        n <- as.numeric(gsub("[^0-9]", "", this_score))

        e1 <- res_comparison[["crps_list"]][[forecast_method]][, n][sample_indices]
        e2 <- res_benchmark[["crps_list"]][[method]][diff:nout, n][sample_indices]

        
        
        tmp <- NA
        if (sum(abs(e1 - e2)) != 0)
        {
            tryDM <- try(tmp_DM <- dm.test(e1 = e1,
                                        e2 = e2,
                                        h = 1, power = dm_power, varestimator = varestimator), silent = F)
            tmp <- tmp_DM$statistic
        } else{
            tmp <- 0
        }
        dm_teststat_vec <- tmp
        } else if(this_score == "crps_list"){

        
        e1 <- res_comparison[[this_score]][[forecast_method]][sample_indices]
        e2 <- res_benchmark[[this_score]][[method]][diff:nout][sample_indices]


        
        tmp <- NA

        if (sum(abs(e1 - e2)) != 0)
        {
            tryDM <- try(tmp_DM <- dm.test(e1 = e1,
                                        e2 = e2,
                                        h = 1, power = dm_power, varestimator = varestimator), silent = F)
            tmp <- tmp_DM$statistic
        } else{
            tmp <- 0
        }
        dm_teststat_vec <- tmp
        } else{
        tmp <- NA
        tryDM <- try(tmp_DM <- dm.test(e1 = res_comparison[[this_score]][[forecast_method]][sample_indices],
                                        e2 = res_benchmark[[this_score]][[method]][diff:nout][sample_indices],
                                        h = 1, power = dm_power, varestimator = varestimator), silent = F)
        if(class(tryDM) != "try-error"){
            tmp <- tmp_DM$statistic
        } else{
            tmp <- 0
        }
        dm_teststat_vec <- tmp
        }    
        
        dfmc[[column_name]][ind] <- dm_teststat_vec
    }

  
    return(as.data.frame(dfmc))
}