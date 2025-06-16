source("Data/Postprocessing/load_data.R")

# data_era5 <- load_data("real_model_AFNONet_StationVGP-4_ens_members_51", type = "era5")
# data_kis <- load_data(file_name = "all_data_kiri", type = "kis")
# kis_stations <- sort(unique(data_kis$station))
# data_era5$station <- kis_stations[as.numeric(data_era5$station)]

# data_all <- merge(data_era5, data_kis, by = c("date", "station"))
data_all <- load_data(
    c("all_data_kiri", "real_model_AFNONet_StationVGP-4_ens_members_51"), 
    type = "comb")

data_all <- load_data(
    "all_data_kiri", 
    type = "kis")

data_all_0 <- subset(data_all, data_all$station == 235)

end <- 60
shift <- 3
plot(data_all_0$date[1:end], data_all_0$Model_W0_7[1:end])
lines(data_all_0$date[1:end], data_all_0$T_DRYB_10[1:end])


mae_vals <- c()
shifts <- 1:14
len <- nrow(data_all_0)
for (shift in shifts)
{
mae_vals<-c(mae_vals, mean(abs(data_all_0$Model_W1_1[1:(len-shift + 1)] - data_all_0$T_DEWP_10[shift:len])))
}
plot(shifts, mae_vals)

mae_vals <- c()
shifts <- 1:7
for (shift in shifts)
{
mae_vals<-c(mae_vals, mean(abs(data_all_0$Model_W0_1[1:(len-shift + 1)] - data_all_0$T_DRYB_10[shift:len])))
}
plot(shifts, mae_vals)

plot(data_all$Model_W0_0, data_all$T_DRYB_10)
lines(data_all$T_DRYB_10, data_all$T_DRYB_10)

plot(data_all$Model_W1_0, data_all$T_DEWP_10)
lines(data_all$T_DEWP_10, data_all$T_DEWP_10)


plot(data_all$Model_W1_49, data_all$Model_W0_30)
lines(data_all$Model_W1_49, data_all$Model_W1_49)

w1_names <- names(data_all)[grepl("^Model_W1", names(data_all))]
w1_means <- rowMeans(data_all[w1_names])

w0_names <- names(data_all)[grepl("^Model_W0", names(data_all))]
w0_means <- rowMeans(data_all[w0_names])
plot(w1_means, w0_means, col = rgb(0, 0, 1, alpha = 0.1), pch = 16)
lines(w0_means, w0_means)

names(data_all)[grepl("^Model_W1_|^IFS_x2t_", names(data_all))]

plot(w1_means, data_all$T_DEWP_10)
lines(w1_means, w1_means)

len <- length(w1_means)
mae_vals <- c()
shifts <- 1:7
for (shift in shifts)
{
mae_vals<-c(mae_vals, mean(abs(w1_means[1:(len-shift + 1)] - data_all$T_DEWP_10[shift:len])))
}
plot(shifts, mae_vals)

mae_vals <- c()
shifts <- 1:7
for (shift in shifts)
{
mae_vals<-c(mae_vals, mean(abs(w1_means[shift:len] - data_all$T_DEWP_10[1:(len-shift + 1)])))
}
plot(shifts, mae_vals)

mae_vals <- c()
shifts <- 1:7
for (shift in shifts)
{
mae_vals<-c(mae_vals, mean(abs(w0_means[shift:len] - data_all_0$T_DRYB_10[1:(len-shift + 1)])))
}
plot(shifts, mae_vals)



length(w1_means)
