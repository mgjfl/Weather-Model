source("Data/compute_DM_scores.R")
source("Results/univariate_pit.R")
source("Results/multivariate_pit.R")
source("Results/skill_scores.R")

# The dataset to plot
file_name <- "all_data_kiri"

# Compute the DM scores
compute_DM_scores(file_name)

# Compute rank histograms
univariate_pit(file_name)
multivariate_pit(file_name)

# Compute skill score boxplots
create_skill_plots(file_name)

