source("Results/Postprocessing/univariate_pit.R")
source("Results/Postprocessing/dm_plots.R")
source("Results/Postprocessing/skill_scores.R")
source("Results/Postprocessing/score_boxplots.R")

# Define the plot settings
file_name   <- "toy_s_1_model_AFNONet_VariationalGP_ens_members_51_mout_51" # "ifs_and_ai_mout_51" # "toy_s_3_model_AFNONet_VariationalGP_ens_members_51_mout_51" # "all_data_kiri" # "data45"
numBins     <- 10

scores <- c("es_list", "vs1_list")
model_names_MVPP <- c(
    # "ens",
    "SSh-I14", 
    "SimSchaake-H", 
    # "GCA", "SimGCA", 
    # "GCAsh", "SimGCAsh", 
    "CopGCA", 
    "CopGCAsh", 
    # "SimCopGCAsh", 
    "Clayton",  
    # "SimClayton",
    "Claytonsh", 
    # "SimClaytonsh", 
    "Frank",  
    # "SimFrank",
    "Franksh", 
    # "SimFranksh",
    "Gumbel",  
    # "SimGumbel",
    "Gumbelsh", 
    # "SimGumbelsh", 
    "ECC-Q"#, "ECC-R"#,
    # "boost_cop_fixed", "boost_cop_fixedsh"
    )
# model_names_MVPP <- c("ECC-Q", "SSh-I14", "GCA", "GCAsh", "Clayton", "Claytonsh", "Frank", "Franksh", "Gumbel", "Gumbelsh", "SimSchaake-H", "boost_cop_fixed", "boost_cop_fixedsh")
model_names_UVPP_PIT <- c(
    "SSh-I14", "ens",
    "Clayton", "SimClayton", "Frank", "SimFrank", "Gumbel", "SimGumbel")#, "boost_cop_fixed", "boost_cop_fixedsh")

benchmarks <- c("CopGCA", "Clayton", "Frank", "Gumbel")
models_to_compare <- c("CopGCAsh", "Claytonsh", "Franksh", "Gumbelsh")
# model_names_UVPP_PIT <- c("SSh-I14", "GCA", "ECC-Q", "Clayton", "Frank", "Gumbel", "boost_cop_fixed", "boost_cop_fixedsh")

# Create Diebold-Mariano plots
# create_dm_plots_single_benchmark(file_name, "SimSchaake-H", model_names_MVPP, scores)
create_dm_plots_single_benchmark(file_name, "CopGCA", model_names_MVPP, scores)
# create_dm_plots_single_benchmark(file_name, "ens", model_names_MVPP, scores)
# create_dm_plots_mult_benchmarks(file_name, benchmarks, models_to_compare, scores)


# Create univariate PIT plots
# univariate_pit(file_name, numBins, model_names_UVPP_PIT)

# Create skill plots
# create_skill_plots(file_name, model_names_MVPP, scores, benchmark)

# Create score boxplots
# create_boxplots(file_name, model_names_MVPP, scores)

###############################
## Explanation of PP methods ##
###############################

# univariate_pit(file_name, numBins = numBins, model_names = c("ens", "EMOS-Q"), save_name = "ensemble_uncalibrated_vs_calibrated.png")
# univariate_pit(file_name, numBins, TRUE, c("EMOS-R", "EMOS-Q"), savename = "calibration_R_vs_Q.png")




