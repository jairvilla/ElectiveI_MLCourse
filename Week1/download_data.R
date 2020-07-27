# Week01: cleaning data 
# Title: Get/set working directory 

# Commands-----------------
#getwd
-#setwd("C:\Users\jvill\Desktop\ElectivaML_Course\2-Electiva_MLHC\ElectiveI_MLCourse")


# 

# Descargar data desde internet ------ 
if (!file.exists("./data")){dir.create("./data")}
fileurl    <- "https://data.baltimorecity.gov/resource/aqgr-xx9h.csv"
download.file(fileurl, destfile ="./data/cameras.csv", method ="curl")
list.files ("./data")
cameradata <- read.csv("./data/cameras.csv")
str(cameradata)

# escribir /guardar achivos en un folder 
write.csv(cameradata, file="cameradata.csv", row.names = F)   # Formato csv
write.table(cameradata, file="cameradata.csv", row.names = F) # Formato excel xsls

# explore data 
str(cameradata)
names(cameradata)


# Download Excel file  ------------------







