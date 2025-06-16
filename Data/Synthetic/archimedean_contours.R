# Load necessary packages
library(copula)
library(ggplot2)
library(reshape2)
library(mvtnorm)

plot_folder   <- paste0("Data/Figures/")

res <- 200

# # Define grid
u <- seq(-3, 3, length.out = res)
v <- seq(-3, 3, length.out = res)
# grid_matrix <- as.matrix(grid_df)

# # Function to compute density values and return a data frame
get_copula_df <- function(copula_obj, name) {
  dens <- c()
  for (x in u)
  {
    for (y in v)
    {
      dens<- c(dens, dMvdc(c(x, y), copula_obj))
    }
  }
  df <- expand.grid(u = u, v = v)
  df$density <- dens
  df$copula <- name
  return(df)
}

# Define copulas with moderate dependence (Kendall's tau ≈ 0.5)
theta_clayton <- iTau(claytonCopula(), 0.5)
theta_frank   <- iTau(frankCopula(),   0.5)
theta_gumbel  <- iTau(gumbelCopula(),  0.5)
theta_norm    <- iTau(normalCopula(),  0.5)

cop_clayton  <- mvdc(claytonCopula(param = theta_clayton), margins=c("norm","norm"),
                   paramMargins=list(list(mean=0, sd=1),list(mean=0, sd=1)))
cop_frank    <- mvdc(frankCopula(param = theta_frank), margins=c("norm","norm"),
                   paramMargins=list(list(mean=0, sd=1),list(mean=0, sd=1)))
cop_gumbel   <- mvdc(gumbelCopula(param = theta_gumbel), margins=c("norm","norm"),
                   paramMargins=list(list(mean=0, sd=1),list(mean=0, sd=1)))
cop_norm     <- mvdc(normalCopula(param = theta_norm), margins=c("norm","norm"),
                   paramMargins=list(list(mean=0, sd=1),list(mean=0, sd=1)))


# # Get densities
df_clayton <- get_copula_df(cop_clayton, "Clayton")
df_frank   <- get_copula_df(cop_frank,   "Frank")
df_gumbel  <- get_copula_df(cop_gumbel,  "Gumbel")
df_normal  <- get_copula_df(cop_norm,  "Normal")

# Combine all data
df_all <- rbind(df_clayton, df_frank, df_gumbel, df_normal)


p <- ggplot(df_all, aes(x = u, y = v, z = density)) +
  geom_contour_filled() +
  facet_wrap(~ copula, nrow = 1, scales = "fixed") +
  coord_cartesian(xlim = c(-3, 3), ylim = c(-3, 3), expand = FALSE) +
  scale_x_continuous(expand = c(-3, 3), limits = c(-3, 3)) +
  scale_y_continuous(expand = c(-3, 3), limits = c(-3, 3)) +
  theme_minimal(base_size = 13) +
  theme(
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
    strip.background = element_rect(fill = "gray90", colour = NA),
    strip.text = element_text(size = 18, face = "bold"),   # Enlarged facet titles
    axis.title = element_blank(),
    axis.ticks = element_line(),
    axis.text = element_text(size = 16),
    plot.margin = margin(5, 5, 5, 5),
    panel.spacing = unit(2, "lines")  # Increased spacing between facets
  ) + 
  theme(legend.position="none")

print(p)

ggsave(
    paste0(plot_folder, "contour_plots_copulas.png"),
    plot = p,
    width = 20,
    height = 3,
    dpi = 150
  )
