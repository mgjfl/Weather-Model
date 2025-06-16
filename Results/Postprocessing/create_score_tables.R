source("Data/Postprocessing/load_data.R")
library(tidyr)
library(xtable)

print_table_full <- function(model_names)
{

table_preamble <- sprintf("
\\begin{table}[H]
\\centering
\\makebox[\\textwidth]{
\\begin{tabular}{|ll%s|}
\\hline
\\multicolumn{2}{|l|}{}  %s\\\\\\hline\n", 
strrep("|l", length(model_names)),
paste(paste0(sapply(model_names, function(x) paste0(" & \\textbf{", x, "}"))), collapse = ""))

# Scores for the table
scores <- c("crps_list", "es_list", "vs1_list")
score_names <- c("CRPS", "ES", "VS")

for (i in 1:length(scores))
{
  score <- scores[i]
  score_name <- score_names[i]

  scores_per_model <- res[[score]]
  
  ref_model <- model_names[2]
  
  if (length(dim(scores_per_model[[ref_model]])) == 2)
  { # Average over the first dimension
    d <- dim(scores_per_model[[ref_model]])[2]
    
    table_preamble <- paste0(table_preamble, "\\multicolumn{1}{|c|}{\\multirow{", toString(d), "}{*}{", score_name, "}}\n")
    
    for (dd in 1:d)
    {
      if (dd != 1)
      {
        table_preamble <- paste0(table_preamble, "\\multicolumn{1}{|l|}{} \n")
      }
      table_preamble <- paste0(table_preamble, " & ", toString(dd))
      for (model in model_names)
      {
        val <- mean(scores_per_model[[model]][,dd])
        sd_val <- sd(scores_per_model[[model]][,dd])
        
        table_preamble <- paste0(table_preamble, "& $", toString(round(val, 2)), "\\pm ", toString(round(sd_val, 2)), "$")
      }
      if (dd != d)
      {
        table_preamble <- paste0(table_preamble, "\\\\ \\cline{2-", toString(length(model_names) + 2), "}\n")
      } else
      {
        table_preamble <- paste0(table_preamble, "\\\\ \\hline\n")
      }
      
    }
    
  } else 
  {
    table_preamble <- paste0(table_preamble, "\\multicolumn{2}{|c|}{", score_name, "}\n")
    
    for (model in model_names)
    {
      val <- mean(scores_per_model[[model]])
      sd_val <- sd(scores_per_model[[model]])
      table_preamble <- paste0(table_preamble, " & $", toString(round(val, 2)), "\\pm ", toString(round(sd_val, 2)), "$")
    }
    table_preamble <- paste0(table_preamble, "\\\\\\hline \n")
    
  }
}

table_preamble <- paste0(table_preamble, sprintf("\\end{tabular}
}
\\end{table}"))

cat(table_preamble)
}

model_names <- names(res$mvpp_list)
model_names <- model_names[model_names != "ens"]

# maxModelsPerTable <- 6

# split_vector <- function(vec, chunk_size) {
#   split(vec, ceiling(seq_along(vec) / chunk_size))
# }

# model_groups <- split_vector(model_names, maxModelsPerTable)

# for (model_group in model_groups)
# {
#   print_table(model_group)
#   cat("\n\n")
# }

# https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/prcp_temp_tables/library.txt
# https://www.ogimet.com/display_synopsc2.php?lang=en&estado=Austri&tipo=ALL&ord=REV&nil=SI&fmt=txt&ano=2017&mes=03&day=23&hora=10&anof=2017&mesf=03&dayf=31&horaf=10&send=send
# https://www.ogimet.com/indicativos.phtml.en
# https://gcos.wmo.int/sites/default/files/2024-12/Austrian_GCOS_Report_2020.pdf
# https://catalogue.ceda.ac.uk/uuid/2de7440eec04492aba1147fe83c717d5/

station_to_name_df <- as.data.frame(
  matrix(c(
    c(11343, "Sonnblick"),
    c(11344, "Kolm-Saigurn"),
    c(11346, "Rauris"),
    c(11126, "Patscherkofel"),
    c(11316, "Pitztaler Gletscher"),
    c(11035, "Wien Hohe Warte"),
    c(11290, "Graz Universitaet"),
    c(11320, "Innsbruck Universitaet"),
    c(11022, "Retz"),
    c(11024, "Jauerling"),
    c(11077, "Brunn am Gebirge"),
    c(11007, "Kollerschlag"),
    c(11036, "Wien Schwechat"),
    c(11056, "Vöcklabruck")), ncol = 2, byrow = TRUE
    
  ))
names(station_to_name_df) <- c("stat_nr", "stat_name")

get_station_to_name_map <- function(station_to_name_df)
{

  library(r2r)
  m <- hashmap()

  for (row in 1:nrow(station_to_name_df))
  {
    m[[station_to_name_df$stat_nr[row]]] <- station_to_name_df$stat_name[row]
  }

  return(m)
}


print(xtable(station_to_name_df), include.rownames=FALSE)

station_to_name_map <- get_station_to_name_map(station_to_name_df)

model_names <- c("ens", "EMOS-Q", "GCA", "Gumbel", "Clayton", "Frank")
print_model_names <- c("Raw", "EMOS-Q", "GCA", "Gumbel", "Clayton", "Frank")
all_scores <- list()

file_names <- c("data1", "data2", "data3", "data45")

for (file_name in file_names)
{

  # Settings
  isLAEF              <- TRUE
  observation_columns <- c("obs") 
  
  
  
  
  # Retrieve the scores
  res <- new.env()
  load_score_env(file_name, res)
  
  # Retrieve the data
  data <- load_data(file_name, isLAEF)
  stations <- unique(data$station)
  
  # Load the transformation from (station, obs_type) -> dimension
  get_dimension <- load_dimension_transform(stations, observation_columns)
  
  # For each model
  for (model in model_names)
  {
    scores <- res$crps_list[[model]]
    
    score_df <- as.data.frame(scores)
    names(score_df) <- stations
    
    # Pivot the table
    
    
    pivotted_scores <- score_df %>%
      pivot_longer(
        cols = all_of(stations),
        names_to = "station"
      )
    
    pivotted_scores$station_name <- station_to_name_map[pivotted_scores$station]
    
    if (is.null(all_scores[[model]]))
    {
      all_scores[[model]] <- pivotted_scores
    } else
    {
      all_scores[[model]] <- bind_rows(all_scores[[model]], pivotted_scores)
    }
  }
}

scores_summaries <- data.frame()
for (model in model_names)
{
   
  temp <- all_scores[[model]] %>%
    group_by(station_name) %>%
    summarise_at(vars(value), mean)
  temp$model <- model
  scores_summaries <- bind_rows(scores_summaries, temp)
}

print_crps_table <- function(model_names, scores_summaries)
{

  table_preamble <- sprintf("
  \\begin{table}[H]
  \\centering
  \\makebox[\\textwidth]{
  \\begin{tabular}{l%s}
  \\hline \\hline
  Station  %s\\\\\\hline\n",
  strrep("l", length(model_names)),
  paste(paste0(sapply(print_model_names, function(x) paste0(" & ", x))), collapse = ""))
  
  # Scores for the table
  for(stat_name in unique(scores_summaries$station_name))
  {
    table_preamble <- paste0(table_preamble, unlist(stat_name))
    
    for(mod in model_names)
    {
      value <- unlist(subset(scores_summaries, (station_name == unlist(stat_name)) & (model == mod))$value)[1]
      table_preamble <- paste0(table_preamble, " & ", round(value, 3))
    }
    
    table_preamble <- paste0(table_preamble, "\\\\\n")
  }
  
  
  
  table_preamble <- paste0(table_preamble, sprintf("\\hline\\hline
  \\end{tabular}}
  \\end{table}"))
  
  cat(table_preamble)
}

print_crps_table(model_names, scores_summaries)

reference_model <- "SSh-I14"
models <- names(es_list)
for (model in models)
{
  values <- es_list[[model]] - es_list[[reference_model]]
  if (mean(values) < sd(values))
  {
    x <- "True"
  } else
  {
    x <- "False"
  }
  cat(format(model, width = 13, justify = "left"),  round(mean(values), 4), "+/-", round(sd(values), 4), x, "\n")
}

