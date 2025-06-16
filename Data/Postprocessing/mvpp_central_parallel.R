source("Data/Postprocessing/load_data.R")
source("Data/Postprocessing/MVPP/ensfc.R")
source("Data/Postprocessing/scores.R")
source("Data/Postprocessing/MVPP/mvpp_methods.R")

# Data characteristics
file_name           <- "data1" # "data45"
isLAEF              <- TRUE # FALSE # TRUE
trainingDays        <- 365
trainingWindow      <- 60
observation_columns <- c("obs") # c("obs") # c("T_DEWP_10", "T_DRYB_10")
ensemble_regex      <- c("^laef") # c("^laef") # c("^IFS_x2d_", "^IFS_x2t_")
output_dim_standard <- 51


# set random seed
set.seed(1)

# Stations and days from the data
data          <- load_data(file_name, isLAEF)
stations      <- unique(data$station)
days          <- sort(unique(data$td))
nout          <- length(days) - trainingDays
d             <- length(stations) * length(observation_columns)
get_dimension <- load_dimension_transform(stations, observation_columns)

# Ensemble members
m <- sum(grepl(ensemble_regex[1], names(data))) # Each observation should have the same number of ens members

# Environment to store scores in
score.env <- new.env()
load_score_env(file_name, score.env) # Loads the score env from memory
score.env$file_name           <- file_name
score.env$d                   <- d
score.env$nout                <- nout
score.env$trainingDays        <- trainingDays
score.env$trainingWindow      <- trainingWindow
score.env$observation_columns <- observation_columns
score.env$ensemble_regex      <- ensemble_regex

# Load transformed data
transformed_data        <- load_transformed_data(file_name)
score.env$obs           <- transformed_data$obs
score.env$obs_init      <- transformed_data$obs
score.env$ensfc         <- transformed_data$ensfc
score.env$ensfc_init    <- transformed_data$ensfc_init
score.env$mvpp_list$ens <- transformed_data$ensfc

# Load precomputed datastructures
sim_matrix            <- load_sim_matrix(file_name)
boostRVM              <- load_boostRVM(file_name)
score.env$sim_matrix  <- sim_matrix
score.env$boostRVM    <- boostRVM

# Add the ens scores
add_scores(score.env, list(mvppout = transformed_data$ensfc), "ens", 0)

mvpp_adjusted <- function(...)
{
  if ("output_dim" %in% names(list(...)))
  {
    return(mvpp(...,
                transformed_data  = transformed_data, 
                score.env         = score.env))
  }
  
  return(mvpp(...,
              transformed_data  = transformed_data, 
              score.env         = score.env,
              trainingWindow    = trainingWindow,
              output_dim        = output_dim_standard))
  
}

library(future)
library(future.apply)
plan(multisession, workers = parallel::detectCores())

compute_methods_in_parallel <- function(method_calls, inner_sequential = FALSE)
{

    p <- progressor(steps = length(method_calls) + 1)

    results <- future_lapply(method_calls, function(call_def) {

    if (inner_sequential)
    {
        # Force sequential plan within each parallel worker
        oplan <- future::plan()
        future::plan(sequential)
        on.exit(future::plan(oplan), add = TRUE)
    }
    call_name <- call_def$name
    p(sprintf("Method - %s", call_name))
    call_args <- call_def$args
    result <- do.call(mvpp_adjusted, call_args)
    return(list(name = call_name, result = result))
    }, future.seed = TRUE)

    for (res in results) {
    assign(res$name, res$result, envir = .GlobalEnv)
    }
}

compute_methods_in_sequence <- function(method_calls) {

  results <- lapply(method_calls, function(call_def) {
    call_name <- call_def$name
    cat("Method - ", call_name, "\n")
    call_args <- call_def$args
    result <- do.call(mvpp_adjusted, call_args)
    return(list(name = call_name, result = result))
  })

  for (res in results) {
    assign(res$name, res$result, envir = .GlobalEnv)
  }
}

method_calls_p1 <- list(
    list(name = "emos.q.ecc",   args = list(method = "EMOS", variant = "Q", output_dim = m, saveScores = FALSE)),
    list(name = "emos.s.ecc",   args = list(method = "EMOS", variant = "S", output_dim = m, saveScores = FALSE)),
    list(name = "emos.r.ecc",   args = list(method = "EMOS", variant = "R", output_dim = m, saveScores = FALSE)),    
    list(name = "emos.q",       args = list(method = "EMOS", variant = "Q")),
    list(name = "emos.s",       args = list(method = "EMOS", variant = "S")),
    list(name = "emos.r",       args = list(method = "EMOS", variant = "R"))
)

method_calls_s1 <- list(
    list(name = "gca",          args = list(method = "GCA")),
    list(name = "gca.cop",      args = list(method = "CopGCA")),
    list(name = "clayton",      args = list(method = "Clayton")),
    list(name = "frank",        args = list(method = "Frank")),
    list(name = "gumbel",       args = list(method = "Gumbel"))
)

compute_methods_in_parallel(method_calls_p1)
compute_methods_in_sequence(method_calls_s1)

method_calls_p2 <- list(
    list(name = "ecc.q",            args = list(method = "ECC",                 EMOS_sample = emos.q.ecc$mvppout, variant = "Q")),
    list(name = "ecc.s",            args = list(method = "ECC",                 EMOS_sample = emos.s.ecc$mvppout, variant = "S")),
    list(name = "ecc.r",            args = list(method = "ECC",                 EMOS_sample = emos.r.ecc$mvppout, variant = "R")),
    list(name = "ssh.sim",          args = list(method = "SimSchaake",          EMOS_sample = emos.q$mvppout, sim_matrix = sim_matrix)),
    list(name = "ssh.sim.h",        args = list(method = "SimSchaake-H",        EMOS_sample = emos.q$mvppout, sim_matrix = sim_matrix)),
    list(name = "ssh.h",            args = list(method = "SSh-H",               EMOS_sample = emos.q$mvppout)),
    list(name = "ssh.i",            args = list(method = "SSh-I14",             EMOS_sample = emos.q$mvppout)),
    list(name = "gca.sh",           args = list(method = "GCA",                 EMOS_sample = emos.q$mvppout, shuffle = TRUE, MVPP_sample = gca$mvppout)),
    list(name = "gca.cop.sh",       args = list(method = "CopGCA",              EMOS_sample = emos.q$mvppout, shuffle = TRUE, MVPP_sample = gca.cop$mvppout)),
    list(name = "clayton.sh",       args = list(method = "Clayton",             EMOS_sample = emos.q$mvppout, shuffle = TRUE, MVPP_sample = clayton$mvppout)),
    list(name = "frank.sh",         args = list(method = "Frank",               EMOS_sample = emos.q$mvppout, shuffle = TRUE, MVPP_sample = frank$mvppout)),
    list(name = "gumbel.sh",        args = list(method = "Gumbel",              EMOS_sample = emos.q$mvppout, shuffle = TRUE, MVPP_sample = gumbel$mvppout))
)

method_calls_s2 <- list(
    list(name = "boost_cop_fixed",  args = list(method = "boost_cop_fixed",     mvsample_fallback = ssh.i, boostRVM = boostRVM))
)

compute_methods_in_parallel(method_calls_p2)
compute_methods_in_sequence(method_calls_s2)

method_calls_s3 <- list(
    list(name = "decc.q",           args = list(method = "dECC",                EMOS_sample = emos.q.ecc$mvppout, variant = "Q", ECC_out = ecc.q$mvppout)),
    list(name = "boost_cop_fixed",  args = list(method = "boost_cop_fixed",     mvsample_fallback = ssh.i, boostRVM = boostRVM))
)

compute_methods_in_sequence(method_calls_s3)

method_calls_s4 <- list(
    list(name = "boost_cop_fixed",   args = list(method = "boost_cop_fixed",    EMOS_sample = emos.q$mvppout, shuffle = TRUE, MVPP_sample = boost_cop_fixed$mvppout)),
)

compute_methods_in_sequence(method_calls_s4)




save(list = ls(score.env), file = paste0("Data/MVPP/score_env_", file_name, ".RData"), envir = score.env)

