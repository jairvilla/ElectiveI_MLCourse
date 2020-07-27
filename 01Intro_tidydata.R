# Exploratory data Analysis (EDA)
# Dataviz

# Packages  ................................
pkgs <- c("ggplot2", "unvotes", "DT","dplyr", "tidyr", "caret","xtable","data.table",
          "lubridate","klaR","randomForest","qwraps2","corrplot", "nnet","DT","rio",
          "tidyverse", "psych", "MASS", "ggplot2", "kernlab", "knitr")
# Install packages .........................
sapply(pkgs, require, character.only = T) 

## Save & Load ............................
save.image(file="feature.RData")
load      (file="feature.RData")  

# Libraries  ..............................
library(ggplot2); library(Hmisc);library(pastecs) 
library(unvotes); library(pastecs)
library(tidyverse); library(psych) 
library(lubridate); library(summarytools)
library(DT)

# Load data .............................
# data = read.csv("iris",header = T, sep = ";")  #
load(file=starwars)
?starwars
# Describe Information Dataset 

# Summary   ............................... 
summary(iris)
str(iris)
names(iris)
stat.desc(iris)
str(breast_cancer)
names(breast_cancer)
glimpse(starwars)
names(starwars)
str (starwars)
?starwars      # help 


# Data Viz  (ggplot2)....................
# Scatterplot
ggplot(data = starwars, mapping = aes(x = height, y = mass, color = gender, 
                                      shape = gender)) +
  # Hand NA
  
  sum(is.na(starwars))

#facet_grid(. ~ gender) +     
geom_point(size = 3) +
  labs(title = "Mass vs. height of Starwars characters",
       subtitle = "Faceted by gender",
       x = "Height (cm)", y = "Weight (kg)")

# Tidy data & data wranling  
## Histogram ---------------------------------

ggplot(data = starwars, mapping = aes(x = height)) +
  geom_histogram(binwidth = 10)

## Density plots ------------------------------
ggplot(data = starwars, mapping = aes(x = height)) +
  geom_density()

## Box Plots  --------------------------------
ggplot(data = starwars, mapping = aes(y = height, x = skin_color)) +
  geom_boxplot()

# Categorical data ---------------------------
ggplot(data = starwars, mapping = aes(x = eye_color)) +
  geom_bar()
names(starwars)


# tidy variable s
# %>% (pipes) is a structure for writing more natural

starwars %>%
  filter(species == "Human") %>%
  lm(mass ~ height, data = .)

# wrangling (data manipulation )-------------------------------------
install.packages("dsbox")
library(dsbox)# no esta disponible para esta version 
ncbikecrash

## dplyr  ---------------------------------------------  

names(starwars)
glimpse(starwars)
starwars %>%
  filter (species =="Human", gender=="male")


# Getting data from a dataset on web

if (!file.exists("./data")){dir.create("./data")}
fileUrl   <- "https://data.baltimorecity.gov/resource/k5ry-ef3g.csv"
download.file(fileUrl, destfile = "./data/restaurants.csv", method = "curl")
restData <- read.csv("./data/restaurants.csv")
names(restData)

# EXTRAER Nombres de una Ds


if (!file.exists("./data")){dir.create("./data")}
filenames <- "https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data"
download.file(filenames, destfile = "./data/cancerDetection.csv", method = "curl")
restData <- read.csv("./data/cancerDetection.csv")
names(restData)
str(restData)
summary(restData)