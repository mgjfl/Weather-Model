source("Data/Postprocessing/uvpp_util.R")
source("Data/Postprocessing/sim_util.R")
source("Data/Postprocessing/MVPP/ensfc.R")
source("Data/Postprocessing/compute_DM_scores.R")
source("Data/Postprocessing/boost_util.R")

settings <- c(1)
ens_members <- c(51)
type                <- "comb" # "kis" # "laef" # "toy"
trainingDays        <- 365
observation_columns <- c("T_DRYB_10", "T_DEWP_10") #c("obs") #c("T_DEWP_10", "T_DRYB_10") #  c("W0", "W1")
full_ensemble_regex <- c("^Model_W0_|^IFS_x2t_", "^Model_W1_|^IFS_x2d_") # c("^laef") # c("^IFS_x2d_", "^IFS_x2t_") # c("^Model_W0_", "^Model_W1_")
ifs_regex <- c("^IFS_x2t_", "^IFS_x2d_") 
ai_regex <- c("^Model_W0_", "^Model_W1_") 
benchmarks          <- c("ens", "CopGCA", "Clayton", "Frank", "Gumbel", "SimSchaake-H", "SSh-I14") # "GCA" # "SimSchaake-H"

for (setting in settings)
{
    for (ens_member in ens_members)
    {

        if (type == "toy")
        {
            file_name <-  paste0("toy_s_", setting, "_model_AFNONet_VariationalGP_ens_members_", ens_member)
        } else if (type == "comb")
        {
            file_name <-  c("all_data_kiri", paste0("real_model_AFNONet_StationVGP-4_ens_members_", ens_member))
        }
        

        # Load the csv data
        data <- load_data(file_name, type)

        if (type == "comb")
        {
            file_name <- "ifs_and_ai"
        }


        # # Apply UVPP and save
        # if (type == "comb")
        # {
        #     uvpp <- compute_uvpp_ifs(data, file_name, observation_columns, ifs_regex, ai_regex)
        # } else 
        # {
        #     uvpp <- compute_uvpp_ifs(data, file_name, observation_columns, full_ensemble_regex)
        # }
        

        # Transform the data and save
        # transformed_data <- transform_data(data, file_name, trainingDays, observation_columns, full_ensemble_regex)

        # Compute similarity matrix
        # compute_sim_matrix(data, file_name)

        # # Compute DM statistics
        # DM_name <- paste0(file_name, "_mout_51")
        # compute_DM_scores(DM_name, benchmarks, parallelization = FALSE)

    }
}



