library(dplyr)

load_data <- function(file_name = "all_data_kiri", type = "kis") 
{
  
    data <- switch(type,
        "kis"   = load_kis_data(file_name),
        "laef"  = load_laef_data(file_name),
        "toy"   = load_toy_data(file_name),
        "era5"  = load_era5_data(file_name),
        "comb"  = load_comb_data(file_name),
        NULL
    )

    return (data)

}

load_comb_data <- function(file_names)
{
    # Load individual datasets
    kis_data  <- load_kis_data(file_names[1])
    era5_data <- load_era5_data(file_names[2])

    # Compatability of stations 
    kis_stations <- sort(unique(kis_data$station))
    era5_data$station <- kis_stations[as.numeric(era5_data$station)]

    # Merge datasets
    all_data <- merge(era5_data, kis_data, by = c("date", "station"))

    # Remove unused columns
    all_data$X          <- NULL
    all_data$L          <- NULL
    all_data$N          <- NULL
    all_data$W0         <- NULL
    all_data$W1         <- NULL
    all_data$Model      <- NULL
    all_data$Date       <- NULL
    all_data$td.x       <- NULL
    all_data$td.y       <- NULL
    all_data$ytime.x    <- NULL
    all_data$ytime.y    <- NULL

    # Add date indices
    all_data <- all_data %>%
        arrange(date) %>%
        mutate(td = dense_rank(date))
    
    # For compatibility with other functions
    all_data$ytime <- all_data$date

    return(all_data)

}

load_kis_data <- function(file_name)
{
    # Load KNMI observations from KIS (Klimatologisch Informatie Systeem)
    all_data <- read.csv(paste0("Data/Real/IFS/01-20-2025/", file_name, ".csv"))
    
    # Transformations to help R understand the data
    all_data$date <- as.Date(all_data$validTime)
    all_data$station <- factor(all_data$station)
    
    # Remove unused columns
    all_data$X <- NULL 
    all_data$validTime <- NULL
    all_data$runtime <- NULL
    
    # Add date indices
    all_data <- all_data %>%
        arrange(date) %>%
        mutate(td = dense_rank(date))
    
    # For compatibility with other functions
    all_data$ytime <- all_data$date
    
    return(all_data)
}

load_era5_data <- function(file_name)
{
    # Load KNMI observations from KIS (Klimatologisch Informatie Systeem)
    all_data <- read.csv(paste0("Data/Real/AI_ensemble/", file_name, ".csv"))
    
    # Transformations to help R understand the data
    all_data$station <- factor(all_data$station)
    
    # Remove unused columns
    all_data$V <- NULL 
    all_data$H <- NULL 

    # Add date indices
    all_data <- all_data %>%
        arrange(N) %>%
        mutate(td = dense_rank(N))
    
    # For compatibility with other functions
    all_data$ytime <- all_data$N

    all_data$date <- as.Date(all_data$Date)
    
    return(all_data)
}

load_laef_data <- function(file_name = "group1") 
{
  
  # Load datavar variable
  load(paste0("Data/Real/LAEF/", file_name, ".Rdata"))
  all_data <- datavar
  
  # Transformations to help R understand the data
  all_data$date <- as.Date(all_data$time)
  all_data$station <- factor(all_data$stat)
  
  number_of_stations <- length(unique(all_data$station))
  
  # Remove unused columns
  all_data$lt <- NULL
  all_data$initial <- NULL
  all_data$time <- NULL
  all_data$stat <- NULL
  all_data$td <- NULL
  
  all_data <- all_data %>%
    group_by(date) %>%
    filter(n() == number_of_stations) %>%   # retain only dates with exactly enough rows
    ungroup()
  
  # Add date indices
  all_data <- all_data %>%
    arrange(date) %>%
    mutate(td = dense_rank(date))
  
  # For compatibility with other functions
  all_data$ytime <- all_data$date
  
  return(as.data.frame(all_data))
  
}

load_toy_data <- function(file_name)
{
    # Load KNMI observations from KIS (Klimatologisch Informatie Systeem)
    all_data <- read.csv(paste0("Data/Synthetic/AI_ensemble/", file_name, ".csv"))
    
    # Transformations to help R understand the data
    all_data$station <- factor(all_data$station)
    
    # Remove unused columns
    all_data$V <- NULL 
    all_data$H <- NULL 

    # Add date indices
    all_data <- all_data %>%
        arrange(N) %>%
        mutate(td = dense_rank(N))
    
    # For compatibility with other functions
    all_data$ytime <- all_data$N
    
    return(all_data)
}

load_uvpp <-function(file_name = "all_data_kiri") 
{
  
  # Loads a variable called uvpp
  load(paste0("Data/UVPP/uvpp_", file_name, ".Rdata"))
  
  return(uvpp)
  
}

load_sim_matrix <-function(file_name = "all_data_kiri") 
{
  
  # Loads a variable called uvpp
  load(paste0("Data/SimilarityMatrix/simMatrix_", file_name, ".Rdata"))
  
  return(sim_matrix)
  
}

load_boostRVM <-function(file_name = "all_data_kiri") 
{
  
  # Loads a variable called uvpp
  load(paste0("Data/RVineMatrix/rvine_matrix_", file_name, ".Rdata"))
  
  return(boostRVM)
  
}

load_dimension_transform <-function(stations, observation_columns)
{
  
  get_dimension <-function(stat, obs)
  {
    stat_idx  <- match(stat, stations)
    obs_idx   <- match(obs, observation_columns)
    
    return(stat_idx + length(stations) * (obs_idx - 1))
  }
  
  return(get_dimension)
}

load_transformed_data <- function(file_name = "all_data_kiri")
{
  
  # Loads a variable called transformed_data
  load(paste0("Data/MVPP/transformed_data_", file_name, ".Rdata"))
  
  return(transformed_data)
}

load_score_env <- function(file_name = "all_data_kiri", envir = parent.frame())
{
  
  # Loads many score variables
  load(paste0("Data/MVPP/score_env_", file_name, ".RData"), envir = envir)
}

load_scores <- function(res)
{
  input_scores <- c("crps_list", 
                    "es_list", 
                    "vs0_list", 
                    "vs0w_list", 
                    "vs1_list", 
                    "vs1w_list")
  d <- res$d
  
  for (dd in 1:d)
  {
    input_scores <- c(input_scores, paste0("crps_", dd))
  }
  
  return(input_scores)
}

load_dm_statistics <- function(file_name = "all_data_kiri", comparison = FALSE)
{
  
  # Loads a variable
    if (comparison)
    {
        load(paste0("Data/TestStatistic/dm_statistics_comparison_", file_name, ".Rdata"))
    } else 
    {
        load(paste0("Data/TestStatistic/dm_statistics_", file_name, ".Rdata"))
    }
  
  return(all_dfmc)
  
}

pretty_score_name <- function(score_name)
{

    if (grepl("^crps_\\d+$", score_name)) {
        return(sub("^crps_(\\d+)$", "CRPS-\\1", score_name))
    }

  pretty_name <- switch(
    score_name,
    "es_list"           = "ES",
    "vs0_list"          = "VS-0.5",
    "vs0w_list"         = "VS-w-0.5",
    "vs1_list"          = "VS-1",
    "vs1w_list"         = "VS-w-1",
    "???"
  )
  
  return(pretty_name)
}

pretty_model_name <- function(model_name)
{

    if (grepl("-66sh$", model_name))
    {
        return (paste0(pretty_model_name(sub("-66sh$", "", model_name)), "-66-sh"))
    } else if (grepl("sh$", model_name))
    {
        return (paste0(pretty_model_name(sub("sh$", "", model_name)), "-sh"))
    } else if (grepl("^Sim", model_name))
    {
        return (paste0("Sim-", pretty_model_name(sub("^Sim", "", model_name))))
    }

  pretty_name <- switch(
    model_name,
    "ens"           = "Raw Ensemble",
    "CopGCA"        = "GCA",
    "SimSchaake-H"  = "SimSchaake",
    model_name # Not found
  )
  
  return(pretty_name)
}



