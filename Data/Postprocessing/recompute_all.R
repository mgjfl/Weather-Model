source("Data/Postprocessing/uvpp_util.R")
source("Data/Postprocessing/sim_util.R")
source("Data/Postprocessing/MVPP/ensfc.R")
source("Data/Postprocessing/compute_DM_scores.R")
source("Data/Postprocessing/boost_util.R")

# Data characteristics
file_name           <- "ifs_and_ai_mout_51" # "toy_AFNONet_VariationalGP_m_25_m_51" # "data45" # # "toy_AFNONet_VariationalGP_m_25"
type                <- "kis" # "kis" # "laef"
trainingDays        <- 365
observation_columns <- c("T_DEWP_10", "T_DRYB_10") #c("obs") #c("T_DEWP_10", "T_DRYB_10") #  c("W0", "W1")
ensemble_regex      <- c("^IFS_x2d_", "^IFS_x2t_") # c("^laef") # c("^IFS_x2d_", "^IFS_x2t_") # c("^Model_W0_", "^Model_W1_")
benchmarks          <- c("CopGCA", "Clayton", "Frank", "Gumbel", "SimSchaake-H", "SSh-I14", "Claytonsh", "Franksh", "Gumbelsh", "CopGCAsh", "Clayton-66sh", "Frank-66sh", "Gumbel-66sh", "CopGCA-66sh") # "GCA" # "SimSchaake-H"

# Load the csv data
# data <- load_data(file_name, type)

# Apply UVPP and save
# uvpp <- compute_uvpp_ifs(data, file_name, observation_columns, ensemble_regex)

# Transform the data and save
# transformed_data <- transform_data(data, file_name, trainingDays, observation_columns, ensemble_regex)

# Compute similarity matrix
# compute_sim_matrix(data, file_name)

# Compute the boostRVineMatrix
# compute_boostRVineMatrix(file_name)

# Compute DM statistics
compute_DM_scores(file_name, benchmarks, parallelization = FALSE)


