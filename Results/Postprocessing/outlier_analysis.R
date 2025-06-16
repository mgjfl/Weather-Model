source("Results/Postprocessing/skill_scores.R")

file_name <- "data1"

data <- load_data(file_name, T)
transformed_data <- load_transformed_data(file_name)
n_init <- dim(transformed_data$ensfc_init)[1]
n_out <- dim(transformed_data$ensfc)[1]

model_names <- c("SSh-I14", "GCA", "GCAsh", "Clayton", "Claytonsh", "Frank", "Franksh", "Gumbel", "Gumbelsh", "SimSchaake-H")
benchmark <- "GCA"
scores <- c("es_list", "vs1_list")


# Load all variables
res <- new.env()
load_score_env(file_name, res)

##########################
## Compute Skill scores ##
##########################

df_total <- data.frame(matrix(ncol=4, nrow=0))
colnames(df_total) <- c("input", "value", "day", "type")

for (this_score in scores)
{

dfplot <- data.frame(matrix(ncol=2, nrow=0))
names(dfplot) <- c("input", "value")

ref.values <- c(res[[this_score]][[benchmark]])
days <- (n_init + 1):(n_init + length(vals))

df_scores <- data.frame(matrix(ncol = 2, nrow = 0))
for (input_name in model_names) {
  vals <- c(res[[this_score]][[input_name]])
  skill.vals <- c()
  
  for (i in 1:length(vals)) {
    ref.val     <- ref.values[i]
    val         <- vals[i]
    skill.vals  <- c(skill.vals, computeSkillScore(val, ref.val))
  }
  
  
  
  new_skill_df <- data.frame(input = input_name, value = skill.vals, day = days)
  new_scores_df <- data.frame(input = input_name, value = vals, day = days)
  dfplot <- rbind(dfplot, new_skill_df)
  df_scores <- rbind(df_scores, new_scores_df)
}

# For proper ordering
dfplot$input <- factor(dfplot$input, levels = model_names)
df_scores$input <- factor(df_scores$input, levels = model_names)

# Get the quantiles
skill_quants <- quantile(dfplot$value, c(0.01, 0.99))
score_quants <- quantile(df_scores$value, c(0.01, 0.99))

# Compute the outliers
df_skill_outliers <- subset(dfplot, (dfplot$value < skill_quants[["1%"]]) | (dfplot$value > skill_quants[["99%"]]))
df_score_outliers <- subset(df_scores, (df_scores$value < score_quants[["1%"]]) | (df_scores$value > score_quants[["99%"]]))

df_skill_outliers$type <- paste0(this_score, " skill outlier")
df_score_outliers$type <- paste0(this_score, " score outlier")


df_total <- rbind(df_total, df_skill_outliers, df_score_outliers)
}


ggplot(subset(df_total, grepl("skill", type)), aes(x = day, colour = type)) +
  geom_histogram(breaks = days, fill  = "blue", alpha = 0.5, position = "dodge")

ggplot(subset(df_total, grepl("score", type)), aes(x = day, colour = type)) +
  geom_histogram(breaks = days, fill  = "blue", alpha = 0.5, position = "dodge")


# for(i in 1:nrow(df_score_outliers)) 
# {
#   model <- df_score_outliers$model[[i]]
#   day <- df_score_outliers$day[[i]]
#   mvppout <-  mvpp_list[[model]][]
# }
# 
# all_recomputed_scores <- list()
# 
# for (model in unique(df_score_outliers$input))
# {
#   tmp <- eval_all_mult(mvpp_out = mvpp_list[[model]], 
#                        obs = obs)
#   all_recomputed_scores[[model]] <- tmp$vs1
#     
# }
# 
# all(all_recomputed_scores[["GCA"]] == vs1_list[["GCA"]])

  
