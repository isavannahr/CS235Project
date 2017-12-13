library(dtw)
library(readr)
library(TSclust)
library(reshape2)
library(ggplot)
library(directlabels)

# load the data and cast it in cross tab form
numGunLawsByState <- read_csv("num_gun_laws_by_state_per_year.csv", 
                              col_types = cols_only(lawtotal = col_guess(), 
                                                    state = col_guess(), year = col_character()))
gunLawsTabs <- acast(numGunLawsByState, state~year, value.var="lawtotal")


massShootings14to17 <- read_csv("mass_shootings_2014-2017.csv", 
                                col_types = cols(Year = col_character()))
massShootingsTabs <- acast(massShootings14to17, State~Year, value.var="Num of Mass Shootings")
colnames(massShootings14to17) <- c("year","state","MassShootings")

# plot a time series of all the states by gun laws
ggplot(numGunLawsByState, aes(x = year, y = lawtotal, group = state, colour = state)) + 
  geom_line() +
  scale_colour_discrete(guide = 'none') +
  scale_x_discrete(expand=c(0, 1)) +
  geom_dl(aes(label = state), method = list(dl.trans(x = x - 1.4), "last.points", cex = 0.8)) +
  geom_dl(aes(label = state), method = list(dl.trans(x = x + 1.4), "first.points", cex = 0.8)) 

# plot a time series of all the states by mass shootings
ggplot(massShootings14to17, aes(x = year, y = MassShootings, group = state, colour = state)) + 
  geom_line() +
  scale_colour_discrete(guide = 'none') +
  scale_x_discrete(expand=c(0, 1)) +
  geom_dl(aes(label = state), method = list(dl.trans(x = x + .5), "last.points", cex = 0.8)) +
  geom_dl(aes(label = state), method = list(dl.trans(x = x - .5), "first.points", cex = 0.8)) 

# calculate the dissimilarity matrix based on DTW
dGunLaws <- diss(gunLawsTabs, "DTWARP")
# build a hierarchical cluster and plot
cGunLaws <- hclust(dGunLaws)
plot(cGunLaws, main = "Gun Laws Time Series Clusters 1991-2017", xlab = "States")

# calculate dissimilarity matrix based on DTW
dMassShootings <- diss(massShootingsTabs, "DTWARP")
# create a hierarchical cluster and plot
cMassShootings <- hclust(dMassShootings)
plot(cMassShootings, main="Mass Shootings Time Series Clusters 2014-2017", xlab="States")

