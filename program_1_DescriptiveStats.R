# Contents of the program
# This code file is a descriptive statistics of the paper entitled:
# "Early prediction of the duration of protests using probabilistic
#  Latent Dirichlet Allocation and Decision Trees"
# The paper has been accepted for publication at the Advances in Intelligent Systems and Computing - Springer.

# The data can be downloaded from 
# https://data.code4sa.org/dataset/Protest-Data/7y3u-atvk

#==================================================================
#==================================================================

closeAllConnections()
rm(list = ls())

#Data fetching
# master data is denoted by m.data
setwd("C:\\Users\\~\sa_new_protest")

m.data <- read.csv("Protest_Data.csv",
                   header = T,
                   sep = ",",
                   stringsAsFactors = T,
                   na.strings = "")

dim(m.data)

names(m.data)

#Extracting complete data rows
#complete data is denoted by c.data

c.data <- m.data[complete.cases(m.data),]

dim(c.data)

names(c.data)

#Descriptive statistics

provinces <- gsub(" .*$", 
                  "", 
                  c.data$Police_Station)

provinces.sorted <- sort(prop.table(table(provinces))*100,
                         decreasing = T)

names(provinces.sorted) <- c("Gauteng",
                             "Western Cape",
                             "Kwazulu Natal",
                             "Eastern Cape",
                             "North West",
                             "Limpopo",
                             "Mpumalanga",
                             "Free State",
                             "Northern Cape")

#==================================================================
#==================================================================
#Table 2
print(round(provinces.sorted))

#Fig. for Table 2
barplot(round(provinces.sorted),
            col = rainbow(length(table(provinces))),
            xlab = "Provinces",
            ylab = "% in total",
            main = "Percentage of protests vis-a-vis provinces",
            cex.main = 1.5,
            cex.lab = 1.3,
            cex.axis = 1.2)
#==================================================================
#==================================================================
#Table 3:

issue.sorted <- sort(prop.table(table(c.data$type))*100,
                         decreasing = T)

print(round(issue.sorted))


#Fig for table 3:
barplot(issue.sorted,
        col = rainbow(length(table(issue.sorted))),
        xlab = "Issue",
        ylab = "% in total",
        main = "Percentage of protests vis-a-vis issues",
        cex.main = 1.5,
        cex.lab = 1.3,
        cex.axis = 1.2)
#==================================================================
#==================================================================
#Table 4

violent.sorted <- sort(prop.table(table(c.data$Violent_or_violent))*100,
                       decreasing = T)

print(round(violent.sorted))

#Fig for table 4
barplot(violent.sorted,
        col = rainbow(length(table(violent.sorted))),
        xlab = "State",
        ylab = "% in total",
        main = "Percentage of protests vis-a-vis state",
        cex.main = 1.5,
        cex.lab = 1.3,
        cex.axis = 1.2)
#==================================================================
#==================================================================
#Working with dates to find the duration of protest
c.data$Start_Date <- as.character(c.data$Start_Date)

c.data$Start_Date <- gsub("12:00:00 AM", 
                          "", 
                          as.factor(c.data$Start_Date))

c.data$Start_Date <- as.Date(c.data$Start_Date,
                             "%m/%d/%Y")



c.data$End_Date <- as.character(c.data$End_Date)

c.data$End_Date <- gsub("12:00:00 AM", 
                        "", 
                        as.factor(c.data$End_Date))

c.data$End_Date <- as.Date(c.data$End_Date,
                           "%m/%d/%Y")

#Duration of protest days 
duration.protest.days <- c.data$End_Date - c.data$Start_Date

duration.sorted <- sort(prop.table(table(duration.protest.days))*100,
                        decreasing = T)

#Table 5
print(duration.sorted)

#Fig. for table 5
barplot(duration.sorted,
        col = rainbow(length(table(duration.sorted))),
        xlab = "Days",
        ylab = "% in total",
        main = "Percentage of protests vis-a-vis duration in days",
        cex.main = 1.5,
        cex.lab = 1.3,
        cex.axis = 1.2)
#==================================================================
#==================================================================
