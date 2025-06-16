source("Data/Postprocessing/load_data.R")
library(MCS)

file_name <- "ifs_and_ai_mout_51" 

res <- new.env()
load_score_env(file_name, res)

score <- "es_list"

# Build the loss matrix: rows = forecast cases, columns = models
loss_mat <- do.call(cbind, res[[score]])
colnames(loss_mat) <- names(res[[score]])
loss_mat <- loss_mat[,colnames(loss_mat) != "ens"]
loss_mat <- loss_mat[,!grepl("^EMOS", colnames(loss_mat))]

# Run the MCS procedure
set.seed(123)  # For reproducibility
mcs_out <- MCSprocedure(
  Loss      = loss_mat,
  alpha     = 0.05,   # 5 % confidence interval
  B         = 1000,   # Number of bootstrap samples
  statistic = "TR"    # Test statistic: "Tmax" or "TR"
)

save(mcs_out, file = paste0("Data/TestStatistic/mcs_", score, ".RData"))
