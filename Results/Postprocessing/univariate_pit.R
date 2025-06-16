library(ggpubr)

save_pit_plot <- function(path, plot, dims) {
  ggsave(path, plot = plot, width = dims$width, height = dims$height, dpi = dims$dpi)
}


univariate_pit <- function(
    file_name = "all_data_kiri", 
    numBins = 10, 
    model_names = NULL, 
    save_folder = NULL, 
    save_name = NULL) {
  # Load and prepare data
  res <- new.env()
  load_score_env(file_name, res)

  if (is.null(save_folder))
  {
    save_folder <- paste0("Results/Figures/MVPP/", file_name, "/")
  }
  if (!dir.exists(save_folder)) dir.create(save_folder, recursive = TRUE)

  # Determine models
  mvpp_approaches <- model_names

  # Bin rank values
  res$bin_list <- compute_all_bin_assignments(res, mvpp_approaches)

  # Compute and plot boxplots
  maxHeight <- max(sapply(mvpp_approaches, \(model) max(construct_df_long(res$bin_list[[model]], numBins, dim(res$obs)[1])$value, na.rm = TRUE)))
  
  plot_vec <- lapply(mvpp_approaches, \(model) create_boxplot_pit(model, res, numBins, maxHeight))

  # Arrange and save grid plot
  cols <- ceiling(sqrt(length(mvpp_approaches)))
  rows <- ceiling(length(mvpp_approaches) / cols)
  grid_plot <- arrange_plot_grid(plot_vec, mvpp_approaches, rows, cols)
  final_name <- resolve_output_name(file_name, save_name)

  plot_dims <- list(width = 5 * cols, height = 3 * rows, dpi = 400)
  save_pit_plot(paste0(save_folder, final_name), grid_plot, plot_dims)
}

compute_bin_number <- function(values, obs) {
  sorted <- sort(values)
  equalBins <- which(sorted == obs)
  lastBin <- sum(sorted <= obs)
  if (length(equalBins) > 0) sample(equalBins, 1) else lastBin + 1
}

compute_all_bin_assignments <- function(res, models) {
  n <- dim(res$mvpp_list[[models[1]]])[1]
  d <- dim(res$mvpp_list[[models[1]]])[3]
  bin_list <- list()

  for (model in models) {
    bin_list[[model]] <- array(NA, dim = c(n, d))
    for (day in seq_len(n)) {
      for (i.d in seq_len(d)) {
        if (!model %in% c("Clayton", "Frank", "Gumbel") || !any(is.na(res$mvpp_list[[model]][day,,i.d]))) {
          bin_list[[model]][day, i.d] <- compute_bin_number(res$mvpp_list[[model]][day,,i.d], res$obs[day, i.d])
        }
      }
    }
  }
  return(bin_list)
}

construct_df_long <- function(bin_array, numBins, n) {
  d <- dim(bin_array)[2]
  maxRank <- max(bin_array, na.rm = TRUE)
  rankCounts <- table(factor(bin_array, levels = 1:maxRank), rep(1:d, each = n))

  newBinsRankCounts <- matrix(0, nrow = numBins, ncol = d)
  for (rank in 1:numBins) {
    a <- maxRank / numBins * (rank - 1)
    b <- maxRank / numBins * rank
    a_ceil <- ceiling(a)
    b_floor <- floor(b)

    for (i.d in 1:d) {
      count <- 0
      if (a_ceil > 1) count <- count + rankCounts[a_ceil, i.d] * (a_ceil - a)
      if (a_ceil < b_floor) count <- count + sum(rankCounts[(a_ceil + 1):b_floor, i.d])
      if (b_floor + 1 <= maxRank) count <- count + rankCounts[b_floor + 1, i.d] * (b - b_floor)
      newBinsRankCounts[rank, i.d] <- count
    }
  }

  df_long <- as.data.frame(newBinsRankCounts)
  df_long$row <- factor(1:numBins)
  df_long <- pivot_longer(df_long, cols = -row, names_to = "column", values_to = "value")
  df_long$value <- df_long$value / n
  return(df_long)
}

create_boxplot_pit <- function(model, res, numBins, height) {
  df_long <- construct_df_long(res$bin_list[[model]], numBins, dim(res$obs)[1])
  bin_edges <- seq(0, 1, length.out = numBins + 1)
  bin_centers <- head(bin_edges, -1) + diff(bin_edges)/2
  bin_width <- 1 / numBins * 0.9 

  df_long$bin <- as.factor(df_long$row)
  df_long$x <- bin_centers[as.integer(df_long$row)]
  vertical_breaks <- 5

  ggplot(df_long, aes(x = x, y = value, group = bin)) +
    geom_boxplot(outlier.shape = NA, width = bin_width) +
    geom_hline(yintercept = 1 / numBins, col = "steelblue", linetype = "dashed") +
    scale_x_continuous(breaks = seq(0, 1, by = 1 / numBins), limits = c(0, 1)) +
    scale_y_continuous(breaks = seq(0, height, height / vertical_breaks), 
                       labels=round(seq(0, height, height / vertical_breaks), 3),
                       limits = c(0, height)) +
    labs(y = "Relative Frequency", x = "PIT value") +
    ggtitle(pretty_model_name(model)) +
    theme(plot.title = element_text(hjust = 0.5))
}

arrange_plot_grid <- function(plot_vec, models, rows, cols) {
  ggarrange(plotlist = plot_vec, nrow = rows, ncol = cols)
}

resolve_output_name <- function(file_name, savename) {
  if (!is.null(savename)) return(savename)
  return(paste0("Boxplot_PIT_", file_name, ".png"))
}

