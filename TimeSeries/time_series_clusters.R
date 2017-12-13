library(dtw)
library(readr)
library(TSclust)
library(reshape2)


# load the data and cast it in cross tab form
numGunLawsByState <- read_csv("../Datasets/num_gun_laws_by_state_per_year.csv", 
                              col_types = cols_only(lawtotal = col_guess(), 
                                                    state = col_guess(), year = col_character()))
gunLawsTabs <- acast(numGunLawsByState, state~year, value.var="lawtotal")


massShootings14to17 <- read_csv("../Datasets/mass_shootings_2014-2017.csv", 
                                col_types = cols(Year = col_character()))
massShootingsTabs <- acast(massShootings14to17, State~Year, value.var="Num of Mass Shootings")

# calculate the dissimilarity matrix based on DTW
dGunLaws <- diss(gunLawsTabs, "DTWARP")
# build a hierarchical cluster and plot
cGunLaws <- hclust(dGunLaws)
plot(cGunLaws, main = "Gun Laws Time Series Clusters 1981-2017", xlab = "States")

# calculate dissimilarity matrix based on DTW
dMassShootings <- diss(massShootingsTabs, "DTWARP")
# create a hierarchical cluster and plot
cMassShootings <- hclust(dMassShootings)
plot(cMassShootings, main="Mass Shootings Time Series Clusters 2014-2017", xlab="States")

