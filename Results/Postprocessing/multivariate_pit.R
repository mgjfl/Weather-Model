# rm(list=ls())
library(ggplot2)
library(ggpubr)
library(purrr)
library(qqplotr)
library(fields)
library(vegan)

source("Data/load_data.R")
source("Results/multivariate_pit_util.R")

multivariate_pit <- function(file_name)
{

# Load all variables
res <- new.env()
load_score_env(file_name, res)
dat   <- res$mvpp_list
obs   <- res$obs
days  <- dim(dat$ECC)[1]
m     <- dim(dat$ECC)[2]

# Save settings
# Create and save the Multivariate histogram plots
plot_folder <- paste0("Results/Figures/MVPP/")
plot_vec <- c()
plotWidth <- 8
plotHeight <- 8
resolution <- 250


# mvtypes <- c("multivariate", "average", "bandDepth", "tree")
mvtypes <- c("multivariate")

for (histType in mvtypes) {
  plotWidth <- 8
  plotHeight <- 8
  plot_vec <- c()
  plot_vec2 <- c()
  for (model in names(dat)) {
    if (model != "EMOS" && model != "GOF") {
      print(paste0("Creating histogram for ", model))
      p <- mvr.histogram(dat, obs, days, modelName = model, histType = histType)
      plot_vec <- c(plot_vec, list(p))
      # savePlots(plot_folder, 
      #           paste0(histType, "_histogram_", model, "_", file_name, ".png"), 
      #           p, plotWidth, plotHeight, resolution)
      
      if (model %in% c("ens", "Clayton", "Frank", "Gumbel")) {
        plot_vec2 <- c(plot_vec2, list(p))
      }
    }
  }
  
  cols <- 6
  rows <- 3
  
  
  plotWidth <- 5 * cols
  plotHeight <- 5 * rows
  savePlots(plot_folder, 
            paste0(histType, "_histogram_grid_", file_name, ".png"), 
            ggarrange(plotlist = plot_vec, nrow = rows, ncol = cols), plotWidth, plotHeight, resolution)
  
  cols <- 2
  rows <- 2
  
  
  plotWidth <- 5 * cols
  plotHeight <- 5 * rows
  savePlots(plot_folder, 
            paste0(histType, "_histogram_subset_grid_", file_name, ".png"), 
            ggarrange(plotlist = plot_vec2, nrow = rows, ncol = cols), plotWidth, plotHeight, resolution)
}
}

