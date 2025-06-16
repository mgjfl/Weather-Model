source("Data/Postprocessing/load_data.R")
library(ggplot2)

file_name <- "all_data_kiri"

res <- new.env()
load_score_env(file_name, res)

obs <- res$obs
models <- names(res$mvpp_list)
models <- models[!(models %in% c("GCA", "SimGCA"))]

# Initialize empty data frame
mae_df <- data.frame(model = character(), mae = numeric(), stringsAsFactors = FALSE)

# Loop through all models in the list
for (model in models) {
    mvpp_ensemble <- res$mvpp_list[[model]]  # Shape: [1045, 51, 12]
  
    # Compute the ensemble mean over the 51 members for each day-location
    ensemble_mean <- apply(mvpp_ensemble, c(1, 3), mean)  # Resulting shape: [1045, 12]

    # Compute the absolute error
    abs_error <- abs(ensemble_mean - obs)

    # Compute the mean absolute error over all days and locations
    mae <- mean(abs_error)
  
    # Store results
    mae_df <- rbind(mae_df, data.frame(model = model, mae = mae))
}

# Plot MAE per model
ggplot(mae_df, aes(x = reorder(model, mae), y = mae)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Mean Absolute Error per MVPP Model",
       x = "Model",
       y = "Mean Absolute Error") +
  theme_minimal() +
  coord_flip()
