library(ggplot2)

source("Data/Postprocessing/load_data.R")

computeSkillScore <- function(val, refval) {
  return (1 - val / refval)
}

create_skill_plots <- function(file_name, model_names = NULL, input_scores = NULL, benchmark = "SSh-I14")
{

# Load all variables
res <- new.env()
load_score_env(file_name, res)


# Save settings
plot_folder   <- paste0("Results/Figures/MVPP/", file_name, "/")
plotWidth     <- 8
plotHeight    <- 8
resolution    <- 400

# Function to save the plots
savePlots <- function(fileName, plot) {
  ggsave(
    paste0(plot_folder, fileName),
    plot = plot,
    width = plotWidth,
    height = plotHeight,
    dpi = resolution
  )
}
  
# Create the palette
if (is.null(model_names))
{
  model_names <- names(res$es_list)
}
mypal       <- colorspace::rainbow_hcl(length(model_names))
mypal_use   <- setNames(mypal[seq_along(model_names)], model_names)

# Set the variables to display
input_models <- model_names

if (is.null(input_scores))
{
  input_scores <- load_scores(res)
}



plotScore <- function(this_score) {
  
  dfplot <- data.frame(matrix(ncol=2, nrow=0))
  
  names(dfplot) <- c("input", "value")
  
  for (input_name in input_models) {
    if(grepl("[0-9]+$", this_score))
    {
      n <- as.numeric(gsub("[^0-9]", "", this_score))
      val <- c(res[["crps_list"]][[input_name]][,n])
    } else 
    {
      val <- c(res[[this_score]][[input_name]])
    }
    newdf <- data.frame(input = input_name, value = val)
    dfplot <- rbind(dfplot, newdf)
  }
  
  dfplot$input <- factor(dfplot$input, levels = input_models)
  
  
  p1 <- ggplot(dfplot, aes(x = input, y = value, colour = input))
  p1 <- p1 + geom_boxplot() + geom_hline(yintercept = 0, linetype = "dashed", color = "gray25") 
  
  # +
  #   scale_y_continuous(limits = quantile(dfplot$value, c(0.05, 0.95)))
  
  
  p1 <- p1 + ggtitle(this_score) 
  
  p1 <- p1 + theme_bw() + theme(legend.position = "bottom")
  p1 <- p1 + xlab("Model") + ylab("Score value") 
  p1 <- p1 + scale_color_manual(values = mypal_use, name = "Model", label = model_names) 
  p1 <- p1 + scale_x_discrete(label = model_names)+
    theme(plot.title = element_text(hjust = 0.5))
  
  
  plotWidth <- 12
  plotHeight <- 6
  resolution <- 400
  fileName <- paste0("scores_", this_score, "_", file_name, ".png")
  
  ggsave(
    paste0(plot_folder, fileName),
    p1,
    width = plotWidth,
    height = plotHeight,
    dpi = resolution
  )
  
}

skillScore <- function(this_score) {
  # Plot data
  # lst <- 
  # n <- length(lst)
  # 
  # dfplot <- data.frame(
  #   model = names(lst),
  #   score = rep(this_score, n),
  #   value = unlist(lst)
  # )
  # 
  # dfplot <- subset(dfplot, model != "emos.q" & model != "GOF" & model != "ens")
  
  dfplot <- data.frame(matrix(ncol=2, nrow=0))
  
  names(dfplot) <- c("input", "value")
  
  ref.values <- c(res[[this_score]][[benchmark]])
  
  
  for (input_name in input_models) {
    vals <- c(res[[this_score]][[input_name]])
    skill.vals <- c()
    
    for (i in 1:length(vals)) {
      ref.val     <- ref.values[i]
      val         <- vals[i]
      skill.vals  <- c(skill.vals, computeSkillScore(val, ref.val))
    }

    # cat("Score: ", this_score, " Model: ", input_name, "\n")
    
    newdf <- data.frame(input = input_name, value = skill.vals)
    dfplot <- rbind(dfplot, newdf)
  }
  
  dfplot$input <- factor(dfplot$input, levels = input_models)
  
  
  p1 <- ggplot(dfplot, aes(x = input, y = value, colour = input))
  p1 <- p1 + geom_boxplot(outlier.shape = NA) + 
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray25") 
  
  # +
  #   scale_y_continuous(limits = quantile(dfplot$value, c(0.05, 0.95)))
  
  
  p1 <- p1 + ggtitle(pretty_score_name(this_score))
  
  p1 <- p1 + theme_bw() + theme(legend.position = "bottom")
  p1 <- p1 + xlab("Model") + ylab("Score value") 
  p1 <- p1 + scale_color_manual(values = mypal_use, name = "Model", label = model_names) 
  p1 <- p1 + scale_x_discrete(label = model_names)+
    theme(plot.title = element_text(hjust = 0.5))
  
  
  plotWidth <- 12
  plotHeight <- 6
  resolution <- 400
  fileName <- paste0("skillScores_", this_score, "_", file_name, ".png")
  
  ggsave(
    paste0(plot_folder, fileName),
    p1,
    width = plotWidth,
    height = plotHeight,
    dpi = resolution
  )
}


for (this_score in input_scores) {
  # plotScore(this_score)
  
    skillScore(this_score)
}

}