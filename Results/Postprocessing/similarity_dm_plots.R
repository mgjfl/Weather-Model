source("Results/Postprocessing/univariate_pit.R")
source("Results/Postprocessing/dm_plots.R")
source("Results/Postprocessing/skill_scores.R")
source("Results/Postprocessing/score_boxplots.R")

# Define the plot settings
# file_name   <- "all_data_kiri_mout_51"
file_name   <- "ifs_and_ai_mout_51"

# model_names <- c(
#     # "CopGCA-66", 
#     # "SimCopGCA", 
#     "SimCopGCA-66", 
#     "Clayton", 
#     # "Clayton-66", 
#     # "SimClayton", 
#     "SimClayton-66", 
#     "Frank", 
#     # "Frank-66", 
#     # "SimFrank", 
#     "SimFrank-66",
#     "Gumbel", 
#     # "Gumbel-66", 
#     # "SimGumbel", 
#     "SimGumbel-66"
# )

# model_names_mult <- c(
#     "SimCopGCA", 
#     "SimClayton", 
#     "SimFrank",
#     "SimGumbel"
# )

# benchmarks_mult <- c(
#     "CopGCA",
#     "Clayton",
#     "Frank",
#     "Gumbel"
# )

# benchmarks <- c(
#     "CopGCA", "CopGCA", "CopGCA",
#     "Clayton", "Clayton", "Clayton",
#     "Frank", "Frank", "Frank",
#     "Gumbel", "Gumbel", "Gumbel"
# )


# create_dm_plots_mult_benchmarks(
#     file_name       = file_name, 
#     benchmarks      = benchmarks_mult, 
#     model_names     = model_names_mult, 
#     input_scores    = "es_list",
#     save_name       = "similarity_es.png"
# )

# create_dm_plots_mult_benchmarks(
#     file_name       = file_name, 
#     benchmarks      = benchmarks_mult, 
#     model_names     = model_names_mult, 
#     input_scores    = "vs1_list",
#     save_name       = "similarity_vs1.png"
# )

# create_dm_plots_single_benchmark(
#     file_name       = file_name, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names, 
#     input_scores    = "es_list",
#     save_name       = "gca_similarity_es.png"
# )

# create_dm_plots_single_benchmark(
#     file_name       = file_name, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names, 
#     input_scores    = "vs1_list",
#     save_name       = "gca_similarity_vs1.png"
# )

model_names_sh <- c(
    "CopGCA-66sh", "SimCopGCA-66sh", 
    "Clayton-66sh",  "SimClayton-66sh", 
    "Frank-66sh",  "SimFrank-66sh",
    "Gumbel-66sh",  "SimGumbel-66sh")

benchmarks_sh <- c(
    "CopGCAsh",  "CopGCAsh",
    "Claytonsh",  "Claytonsh",
    "Franksh",  "Franksh",
    "Gumbelsh",  "Gumbelsh"
)


create_dm_plots_mult_benchmarks(
    file_name       = file_name, 
    benchmarks      = benchmarks_sh, 
    model_names     = model_names_sh, 
    input_scores    = "es_list",
    save_name       = "similarity_es_shuffled.png"
)

create_dm_plots_mult_benchmarks(
    file_name       = file_name, 
    benchmarks      = benchmarks_sh, 
    model_names     = model_names_sh, 
    input_scores    = "vs1_list",
    save_name       = "similarity_vs1_shuffled.png"
)

# create_dm_plots_single_benchmark(
#     file_name       = file_name, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names_sh, 
#     input_scores    = "es_list",
#     save_name       = "gca_similarity_es_shuffled.png"
# )

# create_dm_plots_single_benchmark(
#     file_name       = file_name, 
#     benchmark_name  = "CopGCA", 
#     model_names     = model_names_sh, 
#     input_scores    = "vs1_list",
#     save_name       = "gca_similarity_vs1_shuffled.png"
# )
 
 model_names_66_sh <- c(
    "SimCopGCA-66sh", 
    "SimClayton-66sh", 
    "SimFrank-66sh",
    "SimGumbel-66sh")

benchmarks_66_sh <- c(
    "CopGCA-66sh", 
    "Clayton-66sh", 
    "Frank-66sh", 
    "Gumbel-66sh" 
)

create_dm_plots_mult_benchmarks(
    file_name       = file_name, 
    benchmarks      = benchmarks_66_sh, 
    model_names     = model_names_66_sh, 
    input_scores    = "es_list",
    save_name       = "similarity_66_es_shuffled.png"
)
create_dm_plots_mult_benchmarks(
    file_name       = file_name, 
    benchmarks      = benchmarks_66_sh, 
    model_names     = model_names_66_sh, 
    input_scores    = "vs1_list",
    save_name       = "similarity_66_vs1_shuffled.png"
)
