source("Results/Postprocessing/dm_plots.R")

threefold_dm <- function(all_dfmc, model_names, score_name, dodge = "identity")
{
    dfplot <- subset(all_dfmc, score == score_name)

    dfplot <- dfplot %>%
        pivot_longer(cols = starts_with("bootstrap_"), names_to = "bootstrap", values_to = "value")


    alpha <- 0.25
    quants <- quantile(dfplot$value, c(0.01, 0.99))
    ylimits <- 1.5 * range(quants, qnorm(c(alpha, 1 - alpha)))

    pretty_model_names <- vapply(model_names, pretty_model_name, character(1))
    # pretty_benchmark_names <- vapply(benchmarks, pretty_model_name, character(1))

    # Apply pretty_model_name to the model column
    dfplot <- dfplot[dfplot$model %in% model_names,]
    dfplot$model <- vapply(as.character(dfplot$model), pretty_model_name, character(1))

    # Determine the benchmark label per model
    benchmark_labels <- setNames(pretty_model_names, pretty_model_names)


    # Create data frame for annotation near y = 0
    annot_df <- data.frame(
        x = factor(pretty_model_names, levels = pretty_model_names),
        y = 0,
        label = paste(benchmark_labels[pretty_model_names])
    )

    # Ensure factor levels match the actual transformed values
    dfplot$model <- factor(dfplot$model, levels = pretty_model_names, ordered = TRUE)
    dfplot$method_type <- ifelse(grepl("sh$", dfplot$model), "Shuffled variant", "Standard Methods")
    ylim1 <- boxplot.stats(dfplot$value)$stats[c(1, 5)]


    p <- ggplot(dfplot, aes(model, value, colour = method_type)) +
            geom_rect(aes(xmin = -Inf, xmax = Inf, ymin = qnorm(alpha), ymax = qnorm(1 - alpha)),
                    fill = "gray75", color = "gray75", alpha = alpha) +
            geom_boxplot(aes(fill = name, group = interaction(model, name)),
                outlier.shape = NA, alpha = 0.6, width = 0.5, position = dodge) +
            geom_hline(yintercept = 0, linetype = "dashed", color = "gray25") +
            theme_bw() +
            xlab("Model") + ylab("DM test statistic") +
            theme(
                legend.position = "bottom",
                legend.box.margin = margin(t = 0, b = 0),
                legend.margin = margin(t = -10, b = 0),
                text = element_text(size = 18),
                axis.text.x = element_text(angle = 30, hjust = 1, size = 17),
                axis.title.x = element_text(vjust = 5),
                plot.margin = margin(t = 5, r = 33, b = 0, l = 30),
            ) +
            scale_color_manual(values = c("Shuffled variant" = "darkred", "Standard Methods" = "steelblue"),
                    name = "Method type") + 
            geom_vline(xintercept = seq(1.5, length(model_names) - 0.5, by = 1), 
                    color = "gray60", linetype = "dashed") +
            scale_fill_manual(values = c("AI" = "lightblue", "AI + Stat" = "lightcoral"),
                    name = "Group") + 
            coord_cartesian(ylim = ylim1*1.05) +
            scale_y_continuous() +
            ggtitle(pretty_score_name(score_name)) 

    plot_width <- 1 + 0.8 * length(unique(dfplot$model))
    plot_height <- 5

    ggsave(filename = paste0("Results/Figures/MVPP/Threefold Comparison/dm_", score_name, ".png"), plot = p,
                width = plot_width, height = plot_height, dpi = 100, limitsize = FALSE)

    return (p)
}

get_all_dfmc <- function()
{

    ifs_and_ai_dfmc <- load_dm_statistics("ifs_vs_ifs_and_ai", comparison =  TRUE)
    ifs_and_ai_dfmc$name <- "AI + Stat"
    ai_dfmc <- load_dm_statistics("ifs_vs_ai", comparison =  TRUE)
    ai_dfmc$name <- "AI"

    all_dfmc <- rbind(ifs_and_ai_dfmc, ai_dfmc)
    bootstrap_columns <- grep("bootstrap_[0-9]+$", names(all_dfmc), value = TRUE)
    all_dfmc[bootstrap_columns] <- (-1) * all_dfmc[bootstrap_columns]

    return (all_dfmc)
}


all_dfmc <- get_all_dfmc()

model_names <- c(
    "ens",
    "SSh-I14", "SimSchaake-H", 
    # "GCA", "SimGCA", 
    # "GCAsh", "SimGCAsh", 
    # "CopGCA", "CopGCAsh", 
    # "SimCopGCAsh", 
    "Clayton",  
    # "SimClayton",
    "Claytonsh", 
    # "SimClaytonsh", 
    "Frank",  
    # "SimFrank",
    "Franksh", 
    # "SimFranksh",
    "Gumbel",  
    # "SimGumbel",
    "Gumbelsh", 
    # "SimGumbelsh", 
    "ECC-Q"#, "ECC-R"#,
    # "boost_cop_fixed", "boost_cop_fixedsh"
)

threefold_dm(all_dfmc, model_names, "crps_1")
threefold_dm(all_dfmc, model_names, "es_list")
threefold_dm(all_dfmc, model_names, "vs1_list", "dodge")
# threefold_dm(all_dfmc, model_names, "vs0_list")
