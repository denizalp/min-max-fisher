library(tidyverse)
library(reshape2)

obj_hist_mogd_linear <- read_csv("data/obj/obj_hist_mogd_linear.csv" )
obj_hist_ngd_linear <- read_csv("data/obj/obj_hist_ngd_linear.csv")
obj_hist_mogd_cd <- read_csv("data/obj/obj_hist_mogd_cd.csv")
obj_hist_ngd_cd <- read_csv("data/obj/obj_hist_ngd_cd.csv")
obj_hist_mogd_leontief <- read_csv("data/obj/obj_hist_mogd_leontief.csv")
obj_hist_ngd_leontief  <- read_csv("data/obj/obj_hist_ngd_leontief.csv")

prices_mogd_linear <- read_csv("data/prices/prices_mogd_linear") 
prices_ngd_linear <- read_csv("data/prices/prices_ngd_linear")
prices_mogd_cd <- read_csv("data/prices/prices_mogd_cd")
prices_ngd_cd <- read_csv("data/prices/prices_ngd_cd")
prices_mogd_leontief <- read_csv("data/prices/prices_mogd_leontief")
prices_ngd_leontief <- read_csv("data/prices/prices_ngd_leontief")

iter_num <- length(obj_hist_mogd_linear[1,-1])
exper_num <- 2
obj_hist_mogd_linear <- t(rbind(1:iter_num, obj_hist_mogd_linear[,-1]))
colnames(obj_hist_mogd_linear) <- c("Iteration", paste("Experiment", 1:exper_num))
data1 <- melt(obj_hist_mogd_linear, varnames =  paste("Experiment", 1:exper_num))
head(data1)
ggplot(obj_hist_mogd_linear) +  geom_line(aes(x = Iteration, y = `Experiment 1`))