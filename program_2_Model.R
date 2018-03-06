# Contents of the program
# This code file is a descriptive statistics of the paper entitled:
# "Early prediction of the duration of protests using probabilistic
#  Latent Dirichlet Allocation and Decision Trees"
# The paper has been accepted for publication at the Advances in Intelligent Systems and Computing - Springer.

# Objective: To predict the duration of a protest based on only texts as predictors
# Response variable: duration(days) of protests broken into two classes:
#                   less than one day
#                   one_or_more_days 

# Technique used for text mining: Unsupervised learning - LDA

# Techniques used for classification: Supervised learning - C5.0, Treebag, RF

# The data can be downloaded from 
# https://data.code4sa.org/dataset/Protest-Data/7y3u-atvk
#==================================================================
#==================================================================



closeAllConnections()
rm(list = ls())

#Data fetching
# master data is denoted by m.data
setwd("C:\\Users\\~\\sa_new_protest")

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

#Removing unimportant variables
#reduced dataset is denoted by r.data

r.data <- subset(c.data,
                 select = c(Start_Date,
                            End_Date,
                            Reasonforprotest))

#Note: Metro and Rural are binary opposite of one another.
#      So only one variable is chosen

dim(r.data)
names(r.data)


#Working with dates to find the duration of protest
r.data$Start_Date <- as.character(r.data$Start_Date)

r.data$Start_Date <- gsub("12:00:00 AM", 
                          "", 
                          as.factor(r.data$Start_Date))

r.data$Start_Date <- as.Date(r.data$Start_Date,
                             "%m/%d/%Y")



r.data$End_Date <- as.character(r.data$End_Date)

r.data$End_Date <- gsub("12:00:00 AM", 
                        "", 
                        as.factor(r.data$End_Date))

r.data$End_Date <- as.Date(r.data$End_Date,
                           "%m/%d/%Y")

#Duration of protest days 
duration.protest.days <- r.data$End_Date - r.data$Start_Date

r.data <- cbind(r.data,
                duration.protest.days)

names(r.data)

#Working with the variable: reason for protest

#Remove special character

print(require(stringi))
print(require(stringr))
reason.RemoveSpecialCharacter = NULL

for(i in 1:length(r.data$Reasonforprotest)){
     
     reason.RemoveSpecialCharacter[i] <- str_replace_all(r.data$Reasonforprotest[i], 
                                                         "[^[:alnum:]]", 
                                                         " ")
}

stopwords = c("protest",
              "Protest",
              "demand",
              "Demand")

#removing specific words

reason.RemoveSpWo <- NULL

for(i in 1:length(reason.RemoveSpecialCharacter)){
     
     reason.RemoveSpWo[i] <- gsub(reason.RemoveSpecialCharacter[i],
                                   pattern = paste(stopwords, 
                                            collapse = "|"),
                                   replacement = "")
     
}



print(require(RTextTools))
print(require(topicmodels))
print(require(tm))
print(require(wordcloud))
print(require(RColorBrewer))
print(require(wordcloud))
print(require(plyr))
print(require(ggplot2))
print(require(ldatuning))
print(require(parallel))
print(require(doParallel))

source <- VectorSource(reason.RemoveSpWo)

corpus <- Corpus(source)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus,
                 removeWords, 
                 stopwords('english'))



corpus <- tm_map(corpus, stemDocument)

#==================================================================
#==================================================================
#Fig. 1:

wordcloud(corpus, 
          max.words = 75,
          min.freq  = 25,
          colors=brewer.pal(8, "Dark2"))
#==================================================================
#==================================================================


mat <- DocumentTermMatrix(corpus)

#Finding the optimum number of topics (value of k)
#Code inspired from http://ellisp.github.io/blog/2017/01/05/topic-model-cv
#----------------10-fold cross-validation, different numbers of topics----------------
# Leaving one core spare

#Run this entire block of code in one go

# START HERE
cluster <- makeCluster(detectCores(logical = TRUE) - 1) 
registerDoParallel(cluster)

clusterEvalQ(cluster, {
     library(topicmodels)
})


folds <- 10

n <- nrow(mat)

splitfolds <- sample(1:folds, 
                     n, 
                     replace = TRUE)

#candidates for how many topics i.e. various values of k

candidate_k <- c(2,10, 20, 23,24, 25, 26, 30) 



burnin = 1000
iter = 1000
keep = 50

full_data <- mat
clusterExport(cluster, 
              c("full_data", 
                "burnin", 
                "iter", 
                "keep", 
                "splitfolds", 
                "folds", 
                "candidate_k"))


# we parallelize by the different number of topics.  
#A processor is allocated a value
# of k, and does the cross-validation serially.  
#This is because it is assumed there
# are more candidate values of k than there are 
#cross-validation folds, hence it
# will be more efficient to parallelise

system.time({
     results <- foreach(j = 1:length(candidate_k), 
                        .combine = rbind) %dopar%{
          k <- candidate_k[j]
          results_1k <- matrix(0, 
                               nrow = folds, 
                               ncol = 2)
          colnames(results_1k) <- c("k", "perplexity")
          for(i in 1:folds){
               train_set <- full_data[splitfolds != i , ]
               valid_set <- full_data[splitfolds == i, ]
               
               fitted <- LDA(train_set, 
                             k = k, 
                             method = "Gibbs",
                             control = list(burnin = burnin,
                                            iter = iter,
                                            keep = keep) )
               results_1k[i,] <- c(k, 
                                   perplexity(fitted,
                                              newdata = valid_set))
          }
          return(results_1k)
     }
})

stopCluster(cluster)

results_df <- as.data.frame(results)

# END HERE

save(results_df,
     file = "results_df.RData")

load("results_df.RData")


smoothingSpline = smooth.spline(results_df$k, 
                                results_df$perplexity, 
                                spar=0)

par(mar=c(5,6,4,1)+.1)

plot(results_df$k, 
     results_df$perplexity,
     xlab = "Number of topics",
     ylab = "Perplexity when fitting the trained model to the test set",
     main = "Ten fold cross-validation to find the optimal value of k",
     pch = 16,
     cex.axis = 2.5,
     cex.main=3,
     cex.lab = 2.5)

lines(smoothingSpline,
      col = "blue",
      lwd = 5)


#Latent Dirichlet Allocation


a <- which(results_df$perplexity == min(results_df$perplexity), arr.ind = T)

b <- print(results_df[a,])

optimal.k <- b$k
lda <- LDA(mat, optimal.k)

save(lda,
     file = "lda.RData")

load("lda.RData")


terms(lda)
#topics(lda)


gammaDF <- as.data.frame(lda@gamma) 
names(gammaDF) <- c(1:optimal.k )

# View(gammaDF)



toptopics <- as.data.frame(cbind(document = row.names(gammaDF), 
                                 topic = apply(gammaDF,
                                               1,
                                               function(x) 
                                                    names(gammaDF)
                                               [which(x==max(x))])))

#Finding the index of the highest, second highest... probabilities.

maxn <- function(n) function(x) order(x, decreasing = TRUE)[n]

#Topic associated with max probability
index.Largestprob <- apply(gammaDF, 1, maxn(1))
#See the gammaDF when u the analyzing this
#print(as.matrix(index.Largestprob))  

#Topic associated with 2nd max probability
index.Secondlargestprob <- apply(gammaDF, 1, maxn(2))
#print(as.matrix(index.Secondlargestprob))  

#Topic associated with 3rd max probability
index.Thirdlargestprob <- apply(gammaDF, 1, maxn(3))
#print(as.matrix(index.Thirdlargestprob))

#Topic associated with 4th max probability
index.Fourthlargestprob <- apply(gammaDF, 1, maxn(4))
#print(as.matrix(index.Fourthlargestprob))

index.Largestprob       <- as.matrix(index.Largestprob)
index.Secondlargestprob <- as.matrix(index.Secondlargestprob)
index.Thirdlargestprob  <- as.matrix(index.Thirdlargestprob)
index.Fourthlargestprob <- as.matrix(index.Fourthlargestprob)

topic.Matrixtable <- data.frame(index.Largestprob,
                                index.Secondlargestprob,
                                index.Thirdlargestprob,
                                index.Fourthlargestprob)


r.data <- cbind(r.data,
                topic.Matrixtable)

names(r.data)

#Removing the Start_Date,End_Date and Reasonforprotest further from r.data


#Removing unimportant variables from m.data to create the reduced dataset
# further reduced dataset is denoted by fr.data

fr.data <- subset(r.data,
                  select = -c(Start_Date,
                              End_Date,
                              Reasonforprotest))

names(fr.data)

# Building class for the duration.protest.days variable

#Fig.2
par(mfrow = c(1,2))

par(mar=c(5,6,4,1)+.1)

barplot(prop.table(table(fr.data$duration.protest.days))*100,
        xlab = "Number of days",
        ylab = "% in total",
        col = rainbow(length(table(fr.data$duration.protest.days))),
        cex.names = 3, #class labels
        cex.axis = 3,
        cex.main=3,
        cex.lab = 3)

grid(nx = NULL, 
     ny = NULL, 
     col = "gray", 
     lty = 5,
     lwd = 1, 
     equilogs = TRUE)

print(require(Hmisc))
minor.tick(ny=10, tick.ratio=0.5)

class.days = NULL

for(i in 1:length(fr.data$duration.protest.days)){
     
     if (fr.data$duration.protest.days[i] ==0){
          class.days[i] = "less than one day"
     }
     
     if (fr.data$duration.protest.days[i] >=1){
          class.days[i] = "one or more days"
     }
}

barplot(prop.table(table(class.days))*100,
        xlab = "Classes of number of days",
        ylab = "% in total",
        col = rainbow(length(table(class.days))),
        cex.names = 3, #class labels
        cex.axis = 3,
        cex.main=3,
        cex.lab = 3)

grid(nx = NULL, 
     ny = NULL, 
     col = "gray", 
     lty = 5,
     lwd = 1, 
     equilogs = TRUE)
minor.tick(ny=10, tick.ratio=0.5)

par(mfrow = c(1,1))



fr.data <- cbind(fr.data,
                 class.days)

names(fr.data)

#Final modeling data

modeling.data <- subset(fr.data,
                        select = -c(duration.protest.days))



names(modeling.data)


convert.f <- c(1:dim(modeling.data)[2])

modeling.data[,convert.f] <- data.frame(apply(modeling.data[convert.f], 
                                        2, 
                                        as.factor))


print(require(caret))
print(require(klaR))

print(require(ROSE))

data.balanced.both <- ovun.sample(class.days  ~ ., 
                                  data = modeling.data, 
                                  method = "both", 
                                  p=0.5,      
                                  N=1298, seed = 1)$data



#Random forest with  Caret

inTrain <- createDataPartition(y = data.balanced.both$class.days,
                               p = 0.7,
                               list = F)

training.data <- data.balanced.both[inTrain,]

dim(training.data)

training.x <- subset(training.data,
                     select = -c(class.days))

training.y <- training.data$class.days

testing.data <- data.balanced.both[-inTrain,]

dim(testing.data)

testing.x <- subset(testing.data,
                    select = -c(class.days))

testing.y <- testing.data$class.days
#==================================================================
#==================================================================
# C5.OTree. 
seed <- 10
control <- trainControl(method="cv", 
                        number=10, 
                        repeats=5)

c50_model<-train(class.days~.,
                 data = training.data,
                 method="C5.0Tree",
                 trControl = control)

test.pred.c50 <- predict(c50_model,
                         testing.data)

c50CFM <- confusionMatrix(test.pred.c50,
                          testing.y,
                          dnn = c("Predicted",
                                  "Actual"),
                          positive = 'one or more days')

print(c50CFM)
#==================================================================

## Treebag model
#cont<-trainControl(method="cv",number=5,returnResamp = "none")
seed <- 10



tb_model<-train(class.days~.,
                data = training.data,
                method = "treebag",
                trControl = control)

test.pred.tbag <- predict(tb_model,
                          testing.data)

treebagCFM <- confusionMatrix(test.pred.tbag,
                              testing.y,
                              dnn = c("Predicted",
                                      "Actual"),
                              positive = 'one or more days')

print(treebagCFM)


#==================================================================
#Random forest
rf_tuning_model <-train(class.days~.,
                 data = training.data,
                 method="rf",
                 trControl=trainControl(method="cv",
                                        number=10,
                                        repeats = 5),
                 prox=TRUE,
                 allowParallel=TRUE)


 
print(rf_tuning_model)

#Confusion matrix for the test data with tuned parameters
test.pred.rf <- predict(rf_tuning_model,
                     testing.data)

rfCFM <- confusionMatrix(test.pred.rf,
                         testing.y,
                          dnn = c("Predicted",
                                   "Actual"),
                         positive = "one or more days")

print(rfCFM)
#==================================================================

#Overall accurary comparision

c50CFM$overall
treebagCFM$overall
rfCFM$overall

#==================================================================
#==================================================================
#Result: 

#Random forest gives the best accuracy

#==================================================================
#==================================================================



