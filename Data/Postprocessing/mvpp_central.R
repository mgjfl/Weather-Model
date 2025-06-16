source("Data/Postprocessing/load_data.R")
source("Data/Postprocessing/MVPP/ensfc.R")
source("Data/Postprocessing/scores.R")
source("Data/Postprocessing/MVPP/mvpp_methods.R")

# Data characteristics
file_name           <- c("all_data_kiri", "real_model_AFNONet_StationVGP-4_ens_members_51") # c("all_data_kiri", paste0("real_model_AFNONet_StationVGP-4_ens_members_", ens_member)) # "toy_AFNONet_VariationalGP_m_25" # "data45" # "all_data_kiri" # "toy_AFNONET_VariationalGP"
type                <- "comb" # "kis" # "laef"
trainingDays        <- 365
trainingWindow      <- 30
observation_columns <- c("T_DRYB_10", "T_DEWP_10") #c("obs") #c("T_DRYB_10", "T_DEWP_10") #  c("W0", "W1")
ensemble_regex      <-  c("^Model_W0_|^IFS_x2t_", "^Model_W1_|^IFS_x2d_") # c("^Model_W0_|^IFS_x2t_", "^Model_W1_|^IFS_x2d_") # c("^Model_W0_", "^Model_W1_") # c("^laef") # c("^IFS_x2d_", "^IFS_x2t_") # c("^Model_W0_", "^Model_W1_")
output_dim_standard <- 51



# set random seed
set.seed(1)

# Stations and days from the data
data          <- load_data(file_name, type)
stations      <- unique(data$station)
days          <- sort(unique(data$td))
nout          <- length(days) - trainingDays
d             <- length(stations) * length(observation_columns)
get_dimension <- load_dimension_transform(stations, observation_columns)

# Ensemble members
m <- sum(grepl(ensemble_regex[1], names(data))) # Each observation should have the same number of ens members

# Environment to store scores in
score.env <- new.env()
if (type == "comb")
{
    file_name <- "ifs_and_ai"
    load_score_env(paste0(file_name, "_mout_51"), score.env) # Loads the score env from memory
} else 
{
    load_score_env(file_name, score.env) # Loads the score env from memory
}
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
# boostRVM              <- load_boostRVM(file_name)
score.env$sim_matrix  <- sim_matrix
# score.env$boostRVM    <- boostRVM

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
  
    if ("trainingWindow" %in% names(list(...)))
    {
        return(mvpp(...,
                transformed_data  = transformed_data, 
                score.env         = score.env,
                output_dim        = output_dim_standard,
                addTrainingDays   = TRUE))
    }

    return(mvpp(...,
                transformed_data  = transformed_data, 
                score.env         = score.env,
                trainingWindow    = trainingWindow,
                output_dim        = output_dim_standard))
  
}



##################
## EMOS for ECC ##
##################

emos.q <- mvpp_adjusted(
  method            = "EMOS", 
  variant           = "Q",
  output_dim        = m,
  saveScores        = FALSE)

emos.s <- mvpp_adjusted(
  method            = "EMOS", 
  variant           = "S",
  output_dim        = m,
  saveScores        = FALSE)

emos.r <- mvpp_adjusted(
  method            = "EMOS", 
  variant           = "R",
  output_dim        = m,
  saveScores        = FALSE)

#########
## ECC ##
#########

ecc.q <- mvpp_adjusted(
  method            = "ECC",
  variant           = "Q", # For saved name
  EMOS_sample       = emos.q$mvppout)

ecc.s <- mvpp_adjusted(
  method            = "ECC",
  variant           = "S", # For saved name
  EMOS_sample       = emos.s$mvppout)

ecc.r <- mvpp_adjusted(
  method            = "ECC",
  variant           = "R", # For saved name
  EMOS_sample       = emos.r$mvppout)

decc.q <- mvpp_adjusted(
  method            = "dECC",
  variant           = "Q", # For saved name
  EMOS_sample       = emos.q$mvppout, 
  ECC_out           = ecc.q$mvppout)

##########
## EMOS ##
##########

emos.q <- mvpp_adjusted(
  method            = "EMOS", 
  variant           = "Q")

emos.s <- mvpp_adjusted(
  method            = "EMOS", 
  variant           = "S")

emos.r <- mvpp_adjusted(
  method            = "EMOS", 
  variant           = "R")

#########
## SSH ##
#########

ssh.h <- mvpp_adjusted(
  method            = "SSh-H", 
  EMOS_sample       = emos.q$mvppout)

ssh.i <- mvpp_adjusted(
  method            = "SSh-I14", 
  EMOS_sample       = emos.q$mvppout)

ssh.sim <- mvpp_adjusted(
  method            = "SimSchaake", 
  EMOS_sample       = emos.q$mvppout,
  sim_matrix        = sim_matrix)

ssh.sim.h <- mvpp_adjusted(
  method            = "SimSchaake-H", 
  EMOS_sample       = emos.q$mvppout,
  sim_matrix        = sim_matrix)

##################
## Boost copula ##
##################

## TODO: Make one of the covariates 1 to add an intercept!

# boost_cop <- mvpp_adjusted(
#   method            = "boost_cop",
#   mvsample_fallback = ssh.i)

# boost_cop.sh <- mvpp_adjusted(
#   method            = "boost_cop",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = boost_cop$mvppout)

# boost_cop_fixed <- mvpp_adjusted(
#   method            = "boost_cop_fixed",
#   mvsample_fallback = ssh.i,
#   boostRVM          = boostRVM)

# boost_cop_fixed.sh <- mvpp_adjusted(
#   method            = "boost_cop_fixed",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = boost_cop_fixed$mvppout)


#########
## GCA ##
#########

gca <- mvpp_adjusted(
  method            = "GCA")

sim.gca <- mvpp_adjusted(
  method            = "SimGCA",
  sim_matrix        = sim_matrix)

sim.gca.sh <- mvpp_adjusted(
  method            = "SimGCA",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = sim.gca$mvppout)

gca.sh <- mvpp_adjusted(
  method            = "GCA",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = gca$mvppout)

gca.cop <- mvpp_adjusted(
  method            = "CopGCA")

# gca.cop.ext <- mvpp_adjusted(
#   method            = "CopGCA",
#   trainingWindow    = 66)

# gca.cop.ext.sh <- mvpp_adjusted(
#   method            = "CopGCA-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["CopGCA-66"]])

gca.cop.sh <- mvpp_adjusted(
  method            = "CopGCA",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = gca.cop$mvppout)

sim.gca.cop <- mvpp_adjusted(
  method            = "SimCopGCA",
  sim_matrix        = sim_matrix)

# sim.gca.cop.ext <- mvpp_adjusted(
#   method            = "SimCopGCA",
#   sim_matrix        = sim_matrix,
#   trainingWindow    = 66)

# sim.gca.cop.ext.sh <- mvpp_adjusted(
#   method            = "SimCopGCA-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["SimCopGCA-66"]])

sim.gca.cop.sh <- mvpp_adjusted(
  method            = "SimCopGCA",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = score.env$mvpp_list[["SimCopGCA"]])

#########################
## Archimedean Copulas ##
#########################

clayton <- mvpp_adjusted(
  method            = "Clayton")

gumbel <- mvpp_adjusted(
  method            = "Gumbel")

frank <- mvpp_adjusted(
  method            = "Frank")

# clayton.ext <- mvpp_adjusted(
#   method            = "Clayton",
#   trainingWindow    = 66)

# gumbel.ext <- mvpp_adjusted(
#   method            = "Gumbel",
#   trainingWindow    = 66)

# frank.ext <- mvpp_adjusted(
#   method            = "Frank",
#   trainingWindow    = 66)

# clayton.ext.sh <- mvpp_adjusted(
#   method            = "Clayton-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["Clayton-66"]])

# gumbel.ext.sh <- mvpp_adjusted(
#   method            = "Gumbel-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["Gumbel-66"]])

# frank.ext.sh <- mvpp_adjusted(
#   method            = "Frank-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["Frank-66"]])

sim.clayton <- mvpp_adjusted(
  method            = "SimClayton",
  sim_matrix        = sim_matrix)

sim.gumbel <- mvpp_adjusted(
  method            = "SimGumbel",
  sim_matrix        = sim_matrix)

sim.frank <- mvpp_adjusted(
  method            = "SimFrank",
  sim_matrix        = sim_matrix)

# sim.clayton.ext <- mvpp_adjusted(
#   method            = "SimClayton",
#   sim_matrix        = sim_matrix,
#   trainingWindow    = 66)

# sim.gumbel.ext <- mvpp_adjusted(
#   method            = "SimGumbel",
#   sim_matrix        = sim_matrix,
#   trainingWindow    = 66)

# sim.frank.ext <- mvpp_adjusted(
#   method            = "SimFrank",
#   sim_matrix        = sim_matrix,
#   trainingWindow    = 66)

# sim.clayton.ext.sh <- mvpp_adjusted(
#   method            = "SimClayton-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["SimClayton-66"]])

# sim.gumbel.ext.sh <- mvpp_adjusted(
#   method            = "SimGumbel-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["SimGumbel-66"]])

# sim.frank.ext.sh <- mvpp_adjusted(
#   method            = "SimFrank-66",
#   EMOS_sample       = emos.q$mvppout,
#   shuffle           = TRUE,
#   MVPP_sample       = score.env$mvpp_list[["SimFrank-66"]])

clayton.sh <- mvpp_adjusted(
  method            = "Clayton",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = clayton$mvppout)

gumbel.sh <- mvpp_adjusted(
  method            = "Gumbel",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = gumbel$mvppout)

frank.sh <- mvpp_adjusted(
  method            = "Frank",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = frank$mvppout)

sim.clayton.sh <- mvpp_adjusted(
  method            = "SimClayton",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = score.env$mvpp_list[["SimClayton"]])

sim.gumbel.sh <- mvpp_adjusted(
  method            = "SimGumbel",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = score.env$mvpp_list[["SimGumbel"]])

sim.frank.sh <- mvpp_adjusted(
  method            = "SimFrank",
  EMOS_sample       = emos.q$mvppout,
  shuffle           = TRUE,
  MVPP_sample       = score.env$mvpp_list[["SimFrank"]])


save(list = ls(score.env), 
    file = paste0("Data/MVPP/score_env_", file_name,"_mout_", output_dim_standard, ".RData"), 
    envir = score.env)

