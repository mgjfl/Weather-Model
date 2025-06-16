library(ggplot2)
library(ggpubr)
library(purrr)
library(qqplotr)
library(fields)
library(vegan)

# Function to save the plots
savePlots <- function(plot_folder, fileName, plot, plotWidth, plotHeight, resolution) {
  ggsave(
    paste0(plot_folder, fileName),
    plot,
    width = plotWidth,
    height = plotHeight,
    dpi = resolution
  )
}

set.seed(1)

reformat <- function(dat, obs, days, modelName)
{
  x <- dat[[modelName]]
  B <- list()
  
  for(i in 1:days){
    ## first column of matrix is the observation
    B[[i]] <- cbind(obs[i,], t(matrix(x[i,,], nrow = dim(x)[2], ncol = dim(x)[3])))
    
  }
  
  return(B)
}

## Univariate rank
uv.rank <- function(x, d = 1)
{
  x.rank <- apply(x,1,rank)[,d]
  return(x.rank)
}

## Minimum spanning tree ranks 
mst.rank <- function (x) {
  l.mst <- NULL
  for(f in 1:(dim(x)[2])) {
    euc.dist <- rdist(t(x[,-f]))
    l.mst <- c(l.mst,sum(spantree(euc.dist)$dist))
  }
  x.rank <- rank(l.mst,ties="random")
  return(x.rank)
}

## Multivariate ranks 
mv.rank <- function(x)
{
  d <- dim(x)
  x.prerank <- numeric(d[2])
  for(i in 1:d[2]) {
    x.prerank[i] <- sum(apply(x<=x[,i],2,all))
  }
  x.rank <- rank(x.prerank,ties="random")
  return(x.rank)
}

## Average ranks
avg.rank <- function(x)  {
  x.ranks <- apply(x,1,rank)
  x.preranks <- apply(x.ranks,1,mean)
  x.rank <- rank(x.preranks,ties="random")
  return(x.rank)
}

## Band depth ranks
bd.rank <- function(x)
{
  d <- dim(x)
  x.prerank <- array(NA,dim=d)
  for(i in 1:d[1]) {
    tmp.ranks <- rank(x[i,])
    x.prerank[i,] <- (d[2] - tmp.ranks) * (tmp.ranks - 1)
  }
  x.rank <- apply(x.prerank,2,mean) + d[2] - 1
  x.rank <- rank(x.rank,ties="random")
  return(x.rank)
} 

## Multivariate rank histograms
mvr.histogram <- function(dat, obs, days, modelName, histType, d = 1)
{
  B <- reformat(dat, obs, days, modelName)
  x <- c() 
  for(i in 1:days){
    
    if (histType == "multivariate") {
      x <- c(x, mv.rank(B[[i]])[1])
    } else if (histType == "average") {
      # B.ranks <- apply(B[[i]],2,rank)
      # B.preranks <- apply(B.ranks,1,mean)
      # x <- c(x, rank(B.preranks,ties="random")[1])
      x <- c(x, avg.rank(B[[i]])[1])
    } else if (histType == "bandDepth") {
      x <- c(x, bd.rank(B[[i]])[1])
    } else if (histType == "tree") {
      x <- c(x, mst.rank(B[[i]])[1])
    } else if (histType == "uv") {
      x <- c(x, uv.rank(B[[i]], d)[1])
    }
  }
  
  # .GlobalEnv$x <- x
  
  # hist(x,breaks=seq(0,20,by=1),main="",xlab=hist_xlab,ylab=hist_ylab,axes=FALSE,col="gray40",border="white",ylim=hist_ylim)
  # ggplot(dfplot, aes(model, value, colour = model))
  m <- dim(dat[[modelName]])[2]
  x <- x / (m + 1)
  highestValue <- max(sapply(1:(m+1), FUN = function(s) sum(x == s)))
  
  numberOfBins <- 10
  intercept <- days/numberOfBins
  
  p <- ggplot() + aes(x) + geom_histogram(breaks = seq(0, 1, 1.0 / numberOfBins)) +
    geom_hline(yintercept = intercept, col="steelblue", linetype = "dashed") + 
    scale_x_continuous(breaks = seq(0, 1, 1.0 / numberOfBins), labels=round(seq(0, 1.0, 1.0 / numberOfBins), 2)) + 
    scale_y_continuous(breaks = seq(0, intercept, intercept), labels = seq(0, 1, 1)) +
    labs(y = "Frequency ratio", x = "Normalized rank value")
  
  
  # if (histType == "multivariate") {
  #   p <- p + ggtitle(paste("Multivariate rank histogram for", modelName))
  # } else if (histType == "average") {
  #   p <- p + ggtitle(paste("Average rank histogram for", modelName))
  # } else if (histType == "bandDepth") {
  #   p <- p + ggtitle(paste("Band depth rank histogram for", modelName))
  # } else if (histType == "tree") {
  #   p <- p + ggtitle(paste("Minimum spanning tree rank histogram for", modelName))
  # }
  p <- p + ggtitle(modelName)
  
  plotTheme <- theme(
    axis.text = element_text(size = 7),
    axis.title = element_text(size = 18, face = "bold"),
    plot.title = element_text(size = 22, hjust = 0.5)
  )
  
  p <- p + plotTheme
  
  return(p)
  
}