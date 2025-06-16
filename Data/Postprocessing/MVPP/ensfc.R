source("Data/Postprocessing/load_data.R")

transform_data <- function(data, file_name, trainingDays, observation_columns, ensemble_regex, store_uvpp = TRUE)
{

  # Stations and days from the data
  stations      <- unique(data$station)
  days          <- sort(unique(data$td))
  nout          <- length(days) - trainingDays
  d             <- length(stations) * length(observation_columns)
  get_dimension <- load_dimension_transform(stations, observation_columns)
  
  # Ensemble members
  m               <- sum(grepl(ensemble_regex[1], names(data))) # Each observation should have the same number of ens members
  
  ##############################################
  ## Ensemble and observation data structures ##
  ##############################################
  
  cat("Formatting ensemble and observations...\n")
  
  # Create ensfc and obs data structure
  ensfc_init    <- array(NA, dim = c(trainingDays, m, d))
  ensfc         <- array(NA, dim = c(nout, m, d))
  obs_init      <- array(NA, dim = c(trainingDays, d))
  obs           <- array(NA, dim = c(nout, d))
  
  # Dimensions consists of groups of (station, obs_type)
  for (var_idx in 1:length(observation_columns))
  {
    
    # Column names for the ensemble members
    ensemble_members  <- grepl(ensemble_regex[var_idx], names(data))
    obs_column        <- observation_columns[var_idx]
    
    for (station_nr in stations) 
    {
      # The dimension of the variable
      dd <- get_dimension(station_nr, obs_column)
      
      # Diagnostic information
      cat(paste0(format(Sys.time(), "%H:%M:%S"),
                   " | Creating ensemble structures for (station, obs) = (", station_nr, ", ", obs_column, ")\n"))
      
      for (day in 1:length(days)) 
      {
        
        # Extract forecast for day and dim = index
        dat <- subset(data, td == days[day] & station == station_nr)
        if (day <= trainingDays) {
          ensfc_init[day, , dd] <- unlist(dat[ensemble_members], use.names = FALSE)
          obs_init[day, dd] <- unlist(dat[obs_column], use.names = FALSE)
        } else {
          ensfc[day - trainingDays, , dd] <- unlist(dat[ensemble_members], use.names = FALSE)
          obs[day - trainingDays, dd] <- unlist(dat[obs_column], use.names = FALSE)
        }
      }
    }
  }
  
  if (store_uvpp)
  {
  #################
  ## EMOS output ##
  #################
  
  cat("\nFormatting EMOS output...\n")
  
  uvpp <- load_uvpp(file_name)
  
  dat <- uvpp[c("crps_emos", "station")]
  
  # Create pp_out data structure and fill crps scores
  pp_out      <- array(NA, dim = c(nout, d, 2))
  pp_out_init <- array(NA, dim = c(trainingDays, d, 2))
  
  for (var_idx in 1:length(observation_columns))
  {
    
    # Column names for the ensemble members
    ensemble_members  <- grepl(ensemble_regex[var_idx], names(data))
    obs_column        <- observation_columns[var_idx]
    
    for (station_nr in stations) 
    {
      # The dimension of the variable
      dd <- get_dimension(station_nr, obs_column)
      
      # Diagnostic information
      cat(paste0(format(Sys.time(), "%H:%M:%S"),
                 " | Storing EMOS output for (station, obs) = (", station_nr, ", ", obs_column, ")\n"))
      
      for (day in 1:trainingDays) 
      {
        
        # Subset the data
        temp <- subset(uvpp, td == days[day] & station == station_nr & obs_type == obs_column)
        
        # Check if temp has data (first UVPP window have no values)
        if (nrow(temp) > 0) {
          # If temp is not empty, proceed with the assignment
          pp_out_init[day, dd, ] <- unlist(unname(temp[c("ens_mu", "ens_sd")]))
        } 
        
      }
      
  
      for (day in (trainingDays + 1):length(days)) 
      {
        
          temp <- subset(uvpp, td == days[day] & station == station_nr & obs_type == obs_column)
          # score.env$crps_list$ens[day - trainingDays, index] <- unlist(unname(temp["crps_emos"]))
          pp_out[day - trainingDays, dd, ] <- unlist(unname(temp[c("ens_mu", "ens_sd")]))
          
      }
    }
  }
  } else {
    pp_out <- NULL
    pp_out_init <- NULL
  }
  
  transformed_data <- list(
    "obs"         = obs,
    "obs_init"    = obs_init,
    "ensfc"       = ensfc,
    "ensfc_init"  = ensfc_init,
    "pp_out"      = pp_out,
    "pp_out_init" = pp_out_init
  )
  
  savename <- paste0("Data/MVPP/transformed_data_", file_name, ".Rdata")
  save(transformed_data, file = savename)
  
  return(transformed_data)
}
