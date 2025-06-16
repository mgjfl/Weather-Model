library(scoringRules)

get_method_name <- function(method, variant = NULL, shuffle = FALSE)
{
  if (!is.null(variant))
  {
    method_name <- paste0(method, "-", variant)
  } else
  {
    method_name <- method
  }
  
  if(shuffle)
  {
    method_name <- paste0(method_name, "sh")
  }
  
  return(method_name)
}

add_scores <- function(score.env, res, method, start_time) {
  
  # CRPS
  score.env$crps_list[[method]] <- crps_wrapper(res$mvppout, score.env$obs)
  
  # ES and VS
  tmp <- eval_all_mult(mvpp_out = res$mvppout, obs = score.env$obs)
  score.env$es_list[[method]]   <- tmp$es
  score.env$vs1_list[[method]]  <- tmp$vs1
  score.env$vs1w_list[[method]] <- tmp$vs1w
  score.env$vs0_list[[method]]  <- tmp$vs0
  score.env$vs0w_list[[method]] <- tmp$vs0w
  score.env$mvpp_list[[method]] <- res$mvppout
  
  if ("chosenCopula" %in% names(res))
  {
    score.env$chosenCopula[[method]] <- res$chosenCopula
    score.env$params[[method]] <- res$params
  }
  
  # Timing
  end_time <- Sys.time()
  score.env$timing_list[[method]] <- end_time - start_time
}

eval_all_mult <- function(mvpp_out, obs) {
  esout   <- es_wrapper(mvpp_out, obs)
  vs1out  <- vs_wrapper(mvpp_out, obs, weight = FALSE, p = 1)
  vs1wout <- vs_wrapper(mvpp_out, obs, weight = TRUE, p = 1)
  vs0out  <- vs_wrapper(mvpp_out, obs, weight = FALSE, p = 0.5)
  vs0wout <- vs_wrapper(mvpp_out, obs, weight = TRUE, p = 0.5)
  return(list("es" = esout, "vs1" = vs1out, "vs1w" = vs1wout, "vs0" = vs0out, "vs0w" = vs0wout))
}

### scoring rules
# crps
# es
# vs, p = 0.5
# vs, p = 1
# vs_w, p = 0.5
# vs_w, p = 1
#   for 1 fixed pre-specified weighting matrix

## -------------------- ##

## Part 1: multivariate scoring rules

# wrapper function for scoringRules functions es_sample and vs_sample 
#   to apply to specific array formats in simulation study

# code requires scoringRules package, version >= 0.9.2

# input:
#   mvpp_out: output of mvpp() function
#   obs: observations in evaluation period, should match mvpp_out (is not checked)
#   weight: (only in vs): logical whether to use weighting 
#       (this will use a pre-defined weight matrix, see function)
#   p: (only in vs): parameter p in VS

# output:
#   vector of scores (length = nout)

library(scoringRules)

es_wrapper <- function(mvpp_out, obs){
  n <- dim(mvpp_out)[1]
  out <- vector(length = n)
  for(nn in 1:n){
    out[nn] <- es_sample(y = obs[nn,], dat = t(mvpp_out[nn,,]))
  }
  out
}

vs_wrapper <- function(mvpp_out, obs, weight, p){
  if(!is.logical(weight)){stop("'weight' must be logical")}
  d <- dim(mvpp_out)[3]
  if(weight){
    w <- matrix(0, d, d)
    for(i in 1:d){
      for(j in 1:d){
        w[i,j] <- 1/abs(i-j)
      }
    }
    diag(w) <- 0
    nf <- 0.5*sum(w)
    w <- 1/nf*w
  } else{
    w <- NULL
  }
  n <- dim(mvpp_out)[1]
  out <- vector(length = n)
  for(nn in 1:n){
    out[nn] <- vs_sample(y = obs[nn,], dat = t(mvpp_out[nn,,]), w_vs = w, p = p)
  }
  out
}

## -------------------- ##

## Part 2: univariate scoring rules (CRPS)

# wrapper function for scoringRules function crps_sample
#   to apply to specific array formats in simulation study
# code requires scoringRules package, version >= 0.9.2

# input:
#   mvpp_out: output of mvpp() function
#   obs: observations in evaluation period, should match mvpp_out (is not checked)

# output:
#   array of scores (dimensions = forecast case (n); dimension (d))

crps_wrapper <- function(mvpp_out, obs){
  n <- dim(mvpp_out)[1]
  d <- dim(mvpp_out)[3]
  out <- matrix(NA, nrow = n, ncol = d)
  
  for(nn in 1:n){
    for(dd in 1:d){
      out[nn, dd] <- crps_sample(y = obs[nn,dd], dat = mvpp_out[nn, , dd])
    }
  }
  out
}

