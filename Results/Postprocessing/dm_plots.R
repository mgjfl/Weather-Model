library(ggplot2)
library(gridExtra)
library(tidyr)
library(dplyr)
library(colorspace)

source("Data/Postprocessing/load_data.R")

# --------------------- Helper Functions ---------------------

load_and_prepare_data <- function(file_name, benchmark_name) {
	res <- new.env()
	load_score_env(file_name, res)
	dfmc <- load_dm_statistics(file_name)
	
	df <- subset(dfmc, benchmark == benchmark_name)
	bootstrap_columns <- grep("bootstrap_[0-9]+$", names(df), value = TRUE)
	df[bootstrap_columns] <- (-1) * df[bootstrap_columns]
	
	return(list(df = df, res = res))
}

load_and_prepare_data_comparisons <- function(file_name, benchmarks, models) {
	if (length(benchmarks) != length(models)) {
		stop("Length of 'benchmarks' must match length of 'models'")
	}

	# Load environment and statistics
	res <- new.env()
	load_score_env(file_name, res)
	dfmc <- load_dm_statistics(file_name)

	# Create a filter data.frame of (benchmark, model) pairs
	filter_pairs <- data.frame(benchmark = benchmarks, model = models, stringsAsFactors = FALSE)

	# Keep only rows in dfmc that match any (benchmark, model) pair
	df <- dplyr::semi_join(dfmc, filter_pairs, by = c("benchmark", "model"))

	# Flip bootstrap signs
	bootstrap_columns <- grep("bootstrap_[0-9]+$", names(df), value = TRUE)
	df[bootstrap_columns] <- (-1) * df[bootstrap_columns]

	return(list(df = df, res = res))
}

filter_models <- function(df, model_names, input_scores) {
	# df <- subset(df, model != "ens")

	if (!is.null(model_names)) {
		df <- subset(df, model %in% model_names)
	} else {
		model_names <- unique(df$model)
	}

	df <- subset(df, !(model %in% c("EMOS-Q", "GOF")))

	return(list(df = df, model_names = model_names, input_scores = input_scores))
}

# define_palette <- function(model_names) {
  # pal <- rainbow_hcl(length(model_names))
  # setNames(pal, model_names)
# }

define_palette <- function(model_names) {
	colors <- ifelse(grepl("sh", model_names), "darkred", "steelblue")
	setNames(colors, model_names)
}


plot_scores <- function(dfplot, score, palette, model_names, benchmarks) {
	alpha <- 0.25
	quants <- quantile(dfplot$value, c(0.01, 0.99))
	ylimits <- 1.5 * range(quants, qnorm(c(alpha, 1 - alpha)))

    pretty_model_names <- vapply(model_names, pretty_model_name, character(1))
    pretty_benchmark_names <- vapply(benchmarks, pretty_model_name, character(1))
	
	# Apply pretty_model_name to the model column (vectorized)
    dfplot$model <- vapply(as.character(dfplot$model), pretty_model_name, character(1))
    
	# Determine the benchmark label per model
	if (length(pretty_benchmark_names) > 1) {
		benchmark_labels <- setNames(pretty_benchmark_names, pretty_model_names)
	} else {
		benchmark_labels <- setNames(rep(pretty_benchmark_names[1], length(pretty_model_names)), pretty_model_names)
	}

	# Create data frame for annotation near y = 0
	annot_df <- data.frame(
		x = factor(pretty_model_names, levels = pretty_model_names),
		y = 0,
		label = paste(benchmark_labels[pretty_model_names])
	)

    # Ensure factor levels match the actual transformed values
    dfplot$model <- factor(dfplot$model, levels = pretty_model_names, ordered = TRUE)
	dfplot$method_type <- ifelse(grepl("sh$", dfplot$model), "Shuffled variant", "Standard Methods")


	p <- ggplot(dfplot, aes(model, value, colour = method_type)) +
			geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = qnorm(alpha), ymax = qnorm(1 - alpha)),
					fill = "gray75", color = "gray75", alpha = alpha) +
			geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.7) +
			geom_hline(yintercept = 0, linetype = "dashed", color = "gray25") +
			theme_bw() + 
            theme(
                legend.position = "bottom",
                text=element_text(size=20),
                legend.box.margin = margin(t = 0, b = 0),
                legend.margin = margin(t = -10, b = 0),
                axis.text.x = element_text(angle = 30, hjust = 1, size = 19),
                axis.title.x = element_text(vjust =5),
                plot.margin = margin(t = 5, r = 33, b = 0, l = 30)
            ) +
			xlab("Model") + ylab("DM test statistic") +
			scale_color_manual(values = c("Shuffled variant" = "darkred", "Standard Methods" = "steelblue"),
					name = "Method type")

					
	if (length(benchmarks) > 1) {
		p <- p + geom_vline(xintercept = seq(1.5, length(model_names) - 0.5, by = 1), 
				color = "gray60", linetype = "dashed") +
				geom_text(data = annot_df, aes(x = x, y = y, label = label), 
					colour = "black", vjust = -0.5, size = 3, inherit.aes = FALSE) +
				ggtitle(pretty_score_name(score)) 
	} else 
	{
		p <- p + ggtitle(paste0(pretty_score_name(score), " vs ", benchmark_labels[1]))
	}

	return(p)
}

save_plot <- function(plot_folder, file_name, fig, width, height, dpi = 100) {
	ggsave(filename = paste0(plot_folder, file_name), plot = fig,
			width = width, height = height, dpi = dpi, limitsize = FALSE)
}

create_and_save_plot <- function(score_name, df2, model_names, palette, plot_folder, file_name_prefix, benchmarks, save_name) {
    dfplot <- subset(df2, score == score_name)

	dfplot <- dfplot %>%
		pivot_longer(cols = starts_with("bootstrap_"), names_to = "bootstrap", values_to = "value")


	plot_width <- 2.5 + 0.75 * length(unique(dfplot$model))
	plot_height <- 4.5

	fig <- plot_scores(dfplot, score_name, palette, model_names, benchmarks)

	if (is.null(save_name))
	{
		save_name <- paste0(score_name, "_", file_name_prefix, ".png")

		if (length(benchmarks) == 1)
		{
		save_name <- paste0(benchmarks[1], "_benchmark_", save_name)
		} else {
		save_name <- paste0("multiple_benchmark_", save_name)
		}
	}

	save_plot(plot_folder, save_name, fig, plot_width, plot_height)
}

# --------------------- Main Function ---------------------

create_dm_plots <- function(file_name, data_env,
                            model_names = NULL, input_scores = NULL, benchmarks = NULL, save_folder = NULL, save_name = NULL) {

	# Load and transform data
	df <- data_env$df
	res <- data_env$res

	# Extract score names if not provided
	if (is.null(input_scores)) {
		input_scores <- unique(df$score)
	}

	# Filter models and scores
	filter_result <- filter_models(df, model_names, input_scores)
	df2 <- filter_result$df
	model_names <- filter_result$model_names
	input_scores <- filter_result$input_scores

	if(is.null(save_folder))
	{
		save_folder <- paste0("Results/Figures/MVPP/", file_name, "/")
	}

	# Colour palette
	palette <- define_palette(model_names)

	# Save and plot for each score
	for (score in input_scores) {
		create_and_save_plot(score, df2, model_names, palette, save_folder, file_name, benchmarks, save_name)
	}
}

create_dm_plots_single_benchmark <- function(file_name, benchmark_name, 
                                              model_names = NULL, input_scores = NULL, save_folder = NULL, save_name = NULL) {

	# Load and transform data
	data_env <- load_and_prepare_data(file_name, benchmark_name)
	create_dm_plots(
		file_name     	= file_name, 
		data_env      	= data_env, 
		model_names   	= model_names, 
		input_scores  	= input_scores, 
		benchmarks 		= c(benchmark_name), 
		save_folder 	= save_folder,
		save_name 		= save_name)
}

create_dm_plots_mult_benchmarks <- function(file_name, benchmarks, model_names, input_scores = NULL, save_folder = NULL, save_name = NULL) {

	# Load and transform data
	data_env <- load_and_prepare_data_comparisons(file_name, benchmarks, model_names)
	create_dm_plots(
		file_name 		= file_name, 
		data_env 		= data_env, 
		model_names 	= model_names, 
		input_scores 	= input_scores, 
		benchmarks 		= benchmarks, 
		save_folder 	= save_folder,
		save_name 		= save_name)
  
}
