library(tidyverse)
library(ggplot2)
library(GGally)
library(psych)    # For describe()

# Directory to save plots
save_dir <- "./Data/Figures/"

# Load KNMI observations from KIS (Klimatologisch Informatie Systeem)
all_data <- read.csv("./Data/Real/IFS/01-20-2025/all_data_kiri.csv")

# Transformations to help R understand the data
all_data$date <- as.Date(all_data$validTime)
all_data$station <- factor(all_data$station)

# Remove unused columns
all_data$X <- NULL 
all_data$validTime <- NULL
all_data$runtime <- NULL

# Focus on the De Bilt station and their observations
de_bilt_station_number <- 260
de_bilt_all <- all_data[all_data$station == de_bilt_station_number,c("T_DEWP_10", "T_DRYB_10", "date")]

# Separate summer and winter
de_bilt <- de_bilt_all %>% 
  filter(year(date) == 2022) %>%
  mutate(season = 
           ifelse(month(date)>=3 & month(date)<=5, "spring",(
           ifelse(month(date)>=6 & month(date)<=8, "summer",
           ifelse(month(date)>=9 & month(date)<=11, "autumn", 
           "winter")))))
de_bilt$season <- factor(de_bilt$season)

# Summary
d_summary <-
  de_bilt %>% 
  group_by(season) %>% 
  summarise(Tdryb_m = mean(T_DRYB_10, na.rm = T),
            Tdryb_sd = sd(T_DRYB_10, na.rm = T),
            Tdewp_m = mean(T_DEWP_10, na.rm = T),
            Tdewp_sd = sd(T_DEWP_10, na.rm = T))

# Perform the Kolmogorov-Smirnov test for each season and extract p-values, removing ties
normality_results_dryb <- de_bilt %>%
  group_by(season) %>%
  summarise(p_value = ks.test(unique(T_DRYB_10), "pnorm", 
                              mean(T_DRYB_10, na.rm = TRUE), 
                              sd(T_DRYB_10, na.rm = TRUE))$p.value)
normality_results_dewp <- de_bilt %>%
  group_by(season) %>%
  summarise(p_value = ks.test(unique(T_DEWP_10), "pnorm", 
                              mean(T_DEWP_10, na.rm = TRUE), 
                              sd(T_DRYB_10, na.rm = TRUE))$p.value)

# Draw the fitted normal curves in ggplot
get_stat_function_dryb <- function(season, d_summary) {
  season_summary <- d_summary %>% filter(season == !!season)
  return(
    stat_function(
      data = season_summary,
      fun = dnorm,
      args = list(
        mean = season_summary$Tdryb_m,
        sd = season_summary$Tdryb_sd
      ),
      aes(color = season),
      size = 1
    )
  )
}

get_stat_function_dewp <- function(season, d_summary) {
  season_summary <- d_summary %>% filter(season == !!season)
  return(
    stat_function(
      data = season_summary,
      fun = dnorm,
      args = list(
        mean = season_summary$Tdewp_m,
        sd = season_summary$Tdewp_sd
      ),
      aes(color = season),
      size = 1
    )
  )
}

# Plot Dry bulb temperature
ggplot(data = de_bilt) +
  geom_histogram(aes(x = T_DRYB_10, y = ..density.., color = season), 
                 fill = "white", position = "stack") +
  # Apply the stat_function for each season
  get_stat_function_dryb("spring", d_summary) +
  get_stat_function_dryb("summer", d_summary) +
  get_stat_function_dryb("autumn", d_summary) +
  get_stat_function_dryb("winter", d_summary) +
  # Add p-value annotations for each season at the top-right corner of the facet
  geom_text(data = normality_results_dryb, 
            aes(x = Inf, y = Inf, 
                label = paste("p =", round(p_value, 4))),
            inherit.aes = FALSE, 
            hjust = 1.1, vjust = 1.5, 
            color = "black", size = 7) +
  # Add facets for each season
  facet_wrap(~ season, ncol = 2) +
  theme(strip.text = element_text(size = 23)) + # Adjust facet labels
  labs(title = "Air temperature in De Bilt for 2022", 
       x ="Air Temperature (K)",
       y = "Density") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 26))

ggsave(width = 10, height = 5, paste0(save_dir, "drybulb_normality_de_bilt.pdf"))

# Plot dew point temperature
ggplot(data = de_bilt) +
  geom_histogram(aes(x = T_DEWP_10, y = ..density.., color = season), 
                 fill = "white", position = "stack") +
  # Apply the stat_function for each season
  get_stat_function_dewp("spring", d_summary) +
  get_stat_function_dewp("summer", d_summary) +
  get_stat_function_dewp("autumn", d_summary) +
  get_stat_function_dewp("winter", d_summary) +
  # Add p-value annotations for each season at the top-right corner of the facet
  geom_text(data = normality_results_dewp, 
            aes(x = Inf, y = Inf, 
                label = paste("p =", round(p_value, 4))),
            inherit.aes = FALSE, 
            hjust = 1.1, vjust = 1.5, 
            color = "black", size = 7) +
  # Add facets for each season
  facet_wrap(~ season, ncol = 2) +
  theme(strip.text = element_text(size = 23)) + # Adjust facet labels
  labs(title = "Dew point temperature in De Bilt for 2022", 
       x ="Dew point Temperature (K)",
       y = "Density") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 26))

ggsave(width = 10, height = 5, paste0(save_dir, "dew_point_normality_de_bilt.pdf"))

# Plot seasonality

# Reshape data into long format
de_bilt_long <- de_bilt_all %>%
  pivot_longer(cols = c(T_DEWP_10, T_DRYB_10), 
               names_to = "Temperature_Type", 
               values_to = "Temperature")

# Custom facet labels
facet_labels <- c(T_DEWP_10 = "Dew Point Temperature", 
                  T_DRYB_10 = "Air Temperature")

library(scales)
# Create the plot with facets
ggplot(de_bilt_long, aes(x = date, y = Temperature)) +
  geom_line() +
  labs(x = "Date", 
       y = "Temperature", 
       title = "Temperature Over Time") +
  facet_wrap(~ Temperature_Type, ncol = 1, scales = "free_y", labeller = as_labeller(facet_labels)) +
  scale_y_continuous(breaks = c(270, 285, 300)) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(size = 21)) +
  theme(plot.margin = margin(10, 20, 10, 10))


ggsave(paste0(save_dir, "seasonality_de_bilt.pdf"), width = 10, height = 4)


