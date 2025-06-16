source("Results/Postprocessing/univariate_pit.R")
source("Results/Postprocessing/dm_plots.R")
source("Results/Postprocessing/skill_scores.R")
source("Results/Postprocessing/score_boxplots.R")

#####################
## Plots for paper ##
#####################

group_number <- function(file_name)
{
    return(as.numeric(gsub("[^0-9]", "", file_name)))
}

groups <- c("data1", "data2", "data3", "data45")
knmi_data <- "all_data_kiri"
save_folder_prefix <- paste0("Results/Figures/MVPP/Paper/")

# For CRPS
model_names <- c("SSh-I14", "SimSchaake-H", "CopGCA", "Clayton", "Frank", "Gumbel")
model_names_ecc <- c("SSh-I14", "SimSchaake-H", "CopGCA", "Clayton", "Frank", "Gumbel", "ECC-Q")

# For ES and VS
model_names_combined <- c("SSh-I14", "SimSchaake-H", "CopGCA", "CopGCAsh", "Clayton", "Claytonsh", "Frank", "Franksh", "Gumbel", "Gumbelsh")
model_names_combined_ecc <- c("SSh-I14", "SimSchaake-H", "CopGCA", "CopGCAsh", "Clayton", "Claytonsh", "Frank", "Franksh", "Gumbel", "Gumbelsh", "ECC-Q")

# For improvement plots
model_names_unshuffled <- c("CopGCA", "Clayton", "Frank", "Gumbel")
model_names_shuffled <- c("CopGCAsh", "Claytonsh", "Franksh", "Gumbelsh")

# For PIT
model_names_pit <- c("SSh-I14", "CopGCA", "Clayton", "Frank", "Gumbel")
model_names_pit_ecc <- c("SSh-I14", "CopGCA", "Clayton", "Frank", "Gumbel", "ECC-Q")

for (group in groups)
{

    # For 17 member ensemble
    group_17 <- paste0(group, "_m_17")
    group_folder <- paste0("Group_", group_number(group), "/")

    ##########
    ## CRPS ##
    ##########

    # # Misaligned LAEF 51
    # create_dm_plots_single_benchmark(
    #     file_name       = group, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names, 
    #     input_scores    = "crps_list",
    #     save_folder     = paste0(save_folder_prefix, "CRPS/", group_folder),
    #     save_name       = paste0("misaligned_crps_laef_group_", group_number(group), "_m_51.png")
    # )

    # # Misaligned LAEF 17
    # create_dm_plots_single_benchmark(
    #     file_name       = group_17, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_ecc, 
    #     input_scores    = "crps_list",
    #     save_folder     = paste0(save_folder_prefix, "CRPS/", group_folder),
    #     save_name       = paste0("misaligned_laef_crps_group_", group_number(group), "_m_17.png")
    # )

    # for (dd in 1:3)
    # {
    #     create_dm_plots_single_benchmark(
    #     file_name       = group_17, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_combined_ecc, 
    #     input_scores    = paste0("crps_", dd),
    #     save_folder     = paste0(save_folder_prefix, "CRPS/", group_folder),
    #     save_name       = paste0("realigned_laef_crps_group_", group_number(group), "_d_", dd, "_m_17.png")
    # )
    # }

    # # Realigned LAEF 51
    # create_dm_plots_single_benchmark(
    #     file_name       = group, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_combined, 
    #     input_scores    = "crps_list",
    #     save_folder     = paste0(save_folder_prefix, "CRPS/", group_folder),
    #     save_name       = paste0("realigned_crps_laef_group_", group_number(group), "_m_51.png")
    # )

    # # Realigned LAEF 17
    # create_dm_plots_single_benchmark(
    #     file_name       = group_17, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_combined_ecc, 
    #     input_scores    = "crps_list",
    #     save_folder     = paste0(save_folder_prefix, "CRPS/", group_folder),
    #     save_name       = paste0("realigned_laef_crps_group_", group_number(group), "_m_17.png")
    # )

    ##################
    ## Energy Score ##
    ##################

    # # LAEF 51
    # create_dm_plots_single_benchmark(
    #     file_name       = group, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_combined, 
    #     input_scores    = "es_list",
    #     save_folder     = paste0(save_folder_prefix, "ES/", group_folder),
    #     save_name       = paste0("es_laef_group_", group_number(group), "_m_51.png")
    # )

    # # LAEF 17
    # create_dm_plots_single_benchmark(
    #     file_name       = group_17, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_combined_ecc, 
    #     input_scores    = "es_list",
    #     save_folder     = paste0(save_folder_prefix, "ES/", group_folder),
    #     save_name       = paste0("es_laef_group_", group_number(group), "_m_17.png")
    # )

    #####################
    ## Variogram Score ##
    #####################

    # # LAEF 51
    # create_dm_plots_single_benchmark(
    #     file_name       = group, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_combined, 
    #     input_scores    = "vs1_list",
    #     save_folder     = paste0(save_folder_prefix, "VS/", group_folder),
    #     save_name       = paste0("vs_laef_group_", group_number(group), "_m_51.png")
    # )

    # # LAEF 17
    # create_dm_plots_single_benchmark(
    #     file_name       = group_17, 
    #     benchmark_name  = "CopGCA", 
    #     model_names     = model_names_combined_ecc, 
    #     input_scores    = "vs1_list",
    #     save_folder     = paste0(save_folder_prefix, "VS/", group_folder),
    #     save_name       = paste0("vs_laef_group_", group_number(group), "_m_17.png")
    # )

    ####################
    ## Improvement ES ##
    ####################

    # LAEF 51
    create_dm_plots_mult_benchmarks(
        file_name       = group, 
        benchmarks      = model_names_unshuffled, 
        model_names     = model_names_shuffled, 
        input_scores    = "es_list",
        save_folder     = paste0(save_folder_prefix, "ES/", group_folder),
        save_name       = paste0("improvement_es_laef_group_", group_number(group), "_m_51.png")
    )

    # LAEF 17
    create_dm_plots_mult_benchmarks(
        file_name       = group_17, 
        benchmarks      = model_names_unshuffled, 
        model_names     = model_names_shuffled, 
        input_scores    = "es_list",
        save_folder     = paste0(save_folder_prefix, "ES/", group_folder),
        save_name       = paste0("improvement_es_laef_group_", group_number(group), "_m_17.png")
    )

    ####################
    ## Improvement VS ##
    ####################

    # LAEF 51
    create_dm_plots_mult_benchmarks(
        file_name       = group, 
        benchmarks      = model_names_unshuffled, 
        model_names     = model_names_shuffled, 
        input_scores    = "vs1_list",
        save_folder     = paste0(save_folder_prefix, "VS/", group_folder),
        save_name       = paste0("improvement_vs_laef_group_", group_number(group), "_m_51.png")
    )

    # LAEF 17
    create_dm_plots_mult_benchmarks(
        file_name       = group_17, 
        benchmarks      = model_names_unshuffled, 
        model_names     = model_names_shuffled, 
        input_scores    = "vs1_list",
        save_folder     = paste0(save_folder_prefix, "VS/", group_folder),
        save_name       = paste0("improvement_vs_laef_group_", group_number(group), "_m_17.png")
    )

}

################
### KNMI DATA ##
################

# # Misaligned CRPS
# create_dm_plots_single_benchmark(
#     file_name       = knmi_data, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names_ecc, 
#     input_scores    = "crps_list",
#     save_folder     = paste0(save_folder_prefix, "CRPS/"),
#     save_name       = paste0("misaligned_crps_knmi_m_51.png")
# )

# # Realigned KNMI
# create_dm_plots_single_benchmark(
#     file_name       = knmi_data, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names_combined_ecc, 
#     input_scores    = "crps_list",
#     save_folder     = paste0(save_folder_prefix, "CRPS/"),
#     save_name       = paste0("realigned_crps_knmi_m_51.png")
# )

# create_dm_plots_single_benchmark(
#     file_name       = knmi_data, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names_combined_ecc, 
#     input_scores    = "crps_1",
#     save_folder     = paste0(save_folder_prefix, "CRPS/"),
#     save_name       = paste0("realigned_crps_d_1_knmi_m_51.png")
# )

# # ES
# create_dm_plots_single_benchmark(
#     file_name       = knmi_data, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names_combined_ecc, 
#     input_scores    = "es_list",
#     save_folder     = paste0(save_folder_prefix, "ES/"),
#     save_name       = paste0("es_knmi_m_51.png")
# )

# # VS
# create_dm_plots_single_benchmark(
#     file_name       = knmi_data, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names_combined_ecc, 
#     input_scores    = "vs1_list",
#     save_folder     = paste0(save_folder_prefix, "VS/"),
#     save_name       = paste0("vs_knmi_m_51.png")
# )

# Improvement ES
create_dm_plots_mult_benchmarks(
    file_name       = knmi_data, 
    benchmarks      = model_names_unshuffled, 
    model_names     = model_names_shuffled, 
    input_scores    = "es_list",
    save_folder     = paste0(save_folder_prefix, "ES/"),
    save_name       = paste0("improvement_es_knmi_m_51.png")
)

# Improvement VS
create_dm_plots_mult_benchmarks(
    file_name       = knmi_data, 
    benchmarks      = model_names_unshuffled, 
    model_names     = model_names_shuffled, 
    input_scores    = "vs1_list",
    save_folder     = paste0(save_folder_prefix, "VS/"),
    save_name       = paste0("improvement_vs_knmi_m_51.png")
)

###############
## PIT Plots ##
###############

# numBins     <- 10

# # KNMI 51
# univariate_pit(
#     file_name       = knmi_data, 
#     numBins         = numBins, 
#     model_names     = model_names_pit_ecc,
#     save_folder     = save_folder_prefix,
#     save_name       = "PIT_knmi_m_51.png")

# for (group in groups)
# {

#     # LAEF with 17 ensemble members
#     group_17 <- paste0(group, "_m_17")

#     # LAEF 51   
#     univariate_pit(
#         file_name       = group, 
#         numBins         = numBins, 
#         model_names     = model_names_pit,
#         save_folder     = save_folder_prefix,
#         save_name       = paste0("PIT_laef_group_", group_number(group), "_m_51.png")
#     )

#     # LAEF 17   
#     univariate_pit(
#         file_name       = group_17, 
#         numBins         = numBins, 
#         model_names     = model_names_pit_ecc,
#         save_folder     = save_folder_prefix,
#         save_name       = paste0("PIT_laef_group_", group_number(group), "_m_17.png")
#     )

# }
