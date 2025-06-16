source("Data/Postprocessing/compute_DM_scores.R")
source("Data/Postprocessing/load_data.R")
source("Data/Postprocessing/MVPP/ensfc.R")
source("Data/Postprocessing/scores.R")

# # Put ai ensemble in correct format
# ifs_and_ai_data <- load_data(
#     c("all_data_kiri", "real_model_AFNONet_StationVGP-4_ens_members_51")
#     , "comb")

# transformed_data <- transform_data(
#     ifs_and_ai_data, 
#     "ai_data", 
#     365,
#     c("T_DRYB_10", "T_DEWP_10"), 
#     c("^Model_W0_", "^Model_W1_"),
#     store_uvpp = FALSE
# )

# ai_data <- load_transformed_data("ai_data")
# ai_res <- list(mvppout = ai_data$ensfc)
# score.env <- new.env()
# score.env$obs <- ai_data$obs
# add_scores(score.env, ai_res, "ai_ens", Sys.time())
# score.env$obs           <- ai_data$obs
# score.env$obs_init      <- ai_data$obs
# score.env$ensfc         <- ai_data$ensfc
# score.env$ensfc_init    <- ai_data$ensfc_init
# score.env$mvpp_list$ens <- ai_data$ensfc
# score.env$file_name           <- "ai_data"
# score.env$d                   <- dim(ai_data$ensfc)[3]
# score.env$nout                <- dim(ai_data$ensfc)[1]
# score.env$trainingDays        <- 365
# score.env$trainingWindow      <- 30
# score.env$observation_columns <- c("T_DRYB_10", "T_DEWP_10")
# score.env$ensemble_regex      <- c("^Model_W0_", "^Model_W1_")
# save(list = ls(score.env), 
#     file = paste0("Data/MVPP/score_env_ai_data_mout_51.RData"), 
#     envir = score.env)

# Load stat only
ifs_res <- new.env()
load_score_env("all_data_kiri", ifs_res)

# Load stat and AI
ifs_and_ai_res <- new.env()
load_score_env("ifs_and_ai_mout_51", ifs_and_ai_res)

# Load AI
ai_res <- new.env()
load_score_env("ai_data_mout_51", ai_res)

# Compute DM statistics
all_dfmc <- compute_DM_scores_two_forecasts(
    res_benchmark   = ifs_res,
    res_comparison  = ifs_and_ai_res,
    savename        = "ifs_vs_ifs_and_ai",
    parallelization = FALSE
)

all_dfmc <- compute_DM_scores_two_forecasts(
    res_benchmark   = ifs_res,
    res_comparison  = ai_res,
    savename        = "ifs_vs_ai",
    parallelization = FALSE
)
