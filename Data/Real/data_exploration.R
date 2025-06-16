
library(tidyverse)
library(ggplot2)
library(GGally)
library(psych)    # For describe()

# Directory to save plots
save_dir <- "./Data/Figures/"

# Load KNMI observations from KIS (Klimatologisch Informatie Systeem)
all_data <- read.csv("./Data/Real/IFS/01-20-2025/all_data_kiri.csv")

# Transformations to help R understand the data
# all_data$date <- as.Date(all_data$validTime)
# all_data$station <- factor(all_data$station)

# Shows the structure of the data
str(all_data)

# Brief summary & description
summary(all_data)
describe(all_data)

# Select sensors in De Bilt
debilt <- 260

##################
## De Bilt Data ##
##################

debilt_data <- all_data[all_data$station == debilt,]

# Dry bulb
debilt_dryb <- debilt_data %>%
  group_by(validTime, station) %>%
  summarize(across(c(T_DRYB_10), mean, na.rm = TRUE)) %>%
  pivot_wider(names_from = station, values_from = c(T_DRYB_10))

# Station and data for seasonality
debilt_summer <- debilt_data %>% 
  filter(month(validTime)>=7 & month(validTime)<=9)
debilt_rest <- debilt_data %>% 
  filter(month(validTime)<7 | month(validTime)>9)



## Compare stations
p <- ggpairs(debilt_dryb[,names(debilt_dryb)[names(debilt_dryb) != "validTime"]], 
             aes(alpha = 0.2))  + 
  theme(
    axis.text = element_text(size = 5),      # Reduce axis label text size
    strip.text = element_text(size = 5),    # Reduce facet label size
    legend.text = element_text(size = 5),    # Reduce legend text size
    legend.title = element_text(size = 5)    # Reduce legend title size
  ) +
  labs(title = "Dry Bulb temperature in De Bilt")
ggsave(paste0(save_dir,"dryb_deBilt.png"), plot = p, width = 10, height = 8, dpi = 300)

## Compare seasons

### Summer
hist(debilt_summer$T_DRYB_10)

### Non-Summer
hist(debilt_rest$T_DRYB_10)

# Dew point
debilt_dewp <- debilt_data %>%
  group_by(validTime, station) %>%
  summarize(across(c(T_DEWP_10), mean, na.rm = TRUE)) %>%
  pivot_wider(names_from = station, values_from = c(T_DEWP_10))

## Compare stations
p <- ggpairs(debilt_dewp[,names(debilt_dewp)[names(debilt_dewp) != "validTime"]], 
             aes(alpha = 0.2))  + 
  theme(
    axis.text = element_text(size = 5),      # Reduce axis label text size
    strip.text = element_text(size = 5),    # Reduce facet label size
    legend.text = element_text(size = 5),    # Reduce legend text size
    legend.title = element_text(size = 5)    # Reduce legend title size
  ) +
  labs(title = "Dew Point temperature in De Bilt")
ggsave(paste0(save_dir,"dewp_debilt.png"), plot = p, width = 10, height = 8, dpi = 300)

## Compare seasons

### Summer
hist(debilt_summer$T_DEWP_10)

### Non-Summer
hist(debilt_rest$T_DEWP_10)

##############
## KIS Data ##
##############

# kis_data <- all_data[all_data$station %in% c(240, 260),]
kis_data <- all_data #[all_data$station == debilt,]
kis_data_subset <- subset(kis_data, sapply(kis_data["station"], function(x) return(x %in% c(235, 240))))

create_pair_plot <- function(data, savename = "ggpairs_plot.png", width = 24, height = 16)
{
  df_wide <- data %>%
    group_by(validTime, station) %>%
    summarize(across(c(T_DEWP_10, T_DRYB_10), mean, na.rm = TRUE)) %>%
    pivot_wider(names_from = station, values_from = c(T_DEWP_10, T_DRYB_10))
  df_wide %>% mutate(across(where(is.factor), as.character)) -> df_wide

  df_plot <- df_wide[,names(df_wide) != "validTime"]
  .GlobalEnv$df_plot <- df_plot
  # df_plot2 <- df_plot
  df_plot2 <- df_plot[rowSums(df_plot[, -1] < 245) == 0, ]

  format_names <- function(name) {
    # Split the string into parts
    parts <- unlist(strsplit(name, "_"))
    
    # Construct the formatted string
    formatted_name <- paste0("T[", tolower(parts[2]), "] - ", parts[4])
    
    return(formatted_name)
  }


  # Apply the format_names function to the column names
  colnames(df_plot2) <- sapply(colnames(df_plot), function(name) format_names(name))

  p <- ggpairs(df_plot2, aes(alpha = 0.2), columnLabels = names(df_plot2), labeller = label_parsed,
    upper = list(continuous = wrap("cor", size = 9)))  + 
    theme(
        strip.text.x = element_text(size = 23),         # horizontal (top) variable labels
        strip.text.y = element_text(size = 15),         # vertical variable labels
        legend.text = element_text(size = 20),    # Reduce legend text size
        legend.title = element_text(size = 20),    # Reduce legend title size
        axis.text = element_text(size=23)
    ) +
    scale_x_continuous(breaks=(seq(270, 300, 15))) + # 3 ticks for x-axis
    scale_y_continuous(breaks=(seq(270, 300, 15)))  # 3 ticks for y-axis


  ggsave(paste0(save_dir,savename), plot = p, width = width, height = height, dpi = 100)
}

# create_pair_plot(kis_data)
create_pair_plot(kis_data_subset, "correlation_plot_subset.png", width = 18, height = 7)
