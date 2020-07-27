....................................................
# Title:  Classification Models to predict OSA treatment  
# Goal:   Evaluate the performance of predictive models with a dataset  
# Date:   July, 2019
# Author: J Villanueva
#...................................................

# Packages required -------------------------------------------------------

pkgs <- c("dplyr", "tidyr", "caret","xtable","data.table", "lubridate", 
          "klaR","randomForest","qwraps2","corrplot", "nnet","DT","rio",
          "tidyverse", "psych", "MASS", "ggplot2", "kernlab", "knitr") 
# Install packages ..................
sapply(pkgs, require, character.only = T) 

# Save & Load .......................
save.image(file="feature.RData")
load      (file="feature.RData")        


# Pre-procesing data dataset ----------------------------------------------

names(cpap1)   # variables names 
str(cpap1)

# Creating dataset ..................
df1 <- dplyr::select(cpap1,glucosa, ta_media_24, ta_media_diurna, 
                     ta_media_nocturna, ta_sistolica_24, ta_sistolica_diurna, 
                     tas1,ta_sistolica_nocturna, ta_diastolica_nocturna, tas_pico,
                     doi, sato2, iah, arousal, class)
df1 <- cpap_sig 

str(df1)# 15 features

# Split dataset...............................
set.seed(899)
inTrain <- createDataPartition(y=df1$class, p = 0.70, list =FALSE)
train   <- df1[inTrain,]
test    <- df1[-inTrain,]
x_train <- subset(train, select= -c(class))   # only predictors from train 
x_test  <- subset(test, select= -c(class))    # on?y predictors from test
y_train <- subset(train, select= c(class)) # outcome from train data 1x160
y_test  <- subset(test, select= c(class))   # outcome from test data 1x68
control <- trainControl(method = 'repeatedcv',number = 10,repeats = 3) # control

# Building Models...............................
# Logistic Model -----------------------------------
set.seed(1156)
logFit <- train(class ~.,
                data = train,
                method = 'glm',
                preProc = c("center", "scale"),
                trControl = control)
#....................................................
#  KNN ----
set.seed(445)
knnFit <- train(class~.,
                data = train,
                method = "knn",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = 1:10),
                trControl = control)

#  SVM -----
set.seed(523)
svmFit <- train(class~.,
                data = train,
                method = 'svmLinear',
                preProc = c('center','scale'),
                tuneLength = 7,
                trControl = control)

# Bayes Naives ------
set.seed(456)
nbFit  <- train(class~ .,
                data=train,
                method="nb",
                preProc = c('center','scale'),
                trControl=control)

# Random forest -----
set.seed(123)
rfFit <- train(class~., data = train, 
               method = "rf",
               preProc = c('center','scale'),
               trControl = control,
               skip = TRUE)
rfFit$finalModel
rfFit$bestTune
# Neural Network  -----------
set.seed(123)
nnFit <- train(class~., data = train, 
               method = "nnet",
               preProc = c('center','scale'),
               trControl = control,
               skip = TRUE)


#**************************************************************
# Predicted values -----
# Logistic pred ---- 
log_pred <- predict(logFit,newdata = test[ ,-which(names(test) == "class")], 
                    type="prob")[,2]  # Testing Prediction 


print(aucglm <- auc(test$class, predictionglm))
print(cisglm  <- ci.auc(test$class, predictionglm))

predictionglm01 <- predict(logFit, 
                           newdata = test[ , - which(names(test) == "class")])

confusionmatrix       <- table(predictionglm01, test[ , which(names(test) == "class")])

misclassificationsvmlinear <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationsvmlinear

confusionMatrix(data= predictionglm01, reference = test$class, positive = "1") 


# Compute of Threshold and AUC -------------------------------------
plot.roc(test$class, predictionglm,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

# Calculate optimal threshold with Youden's index ------------------
rocsglm <- roc(test$class, predictionglm)
bestglm <- coords(rocglm, "b", ret = "threshold", best.method = "youden")
bestglm

## Prediction based on misclassification error
predictionglm <- ifelse(predictionsvmlinear > 0.35, 1, 0)
predictionsvmlinearyouden <- as.factor(predictionglm)
confusionmatrix <- table(predictionsvmlinearyouden, 
                         test[ , which(names(test) == "class")])

misclassificationsvmlinear <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationsvmlinear

confusionMatrix(data= predictionsvmlinearyouden,reference = test$class, positive = "1"))


# coef(glm.model) 
# summary(glm.model)

# Predictions full model ---- 
cl.glm  <- predict(glm.model, test[,-14], type="response")  # Probabilidad de que sea 1(no cumple)
#pr.glm  <- predict(glm.model, test[,-41], type="link")      # output logit function 
class.pre
# Note: to find cutoff with function
class.pre <- ifelse(log_pred > 0.5, 1, 0)
class.pre <- factor(class.pre)
class.pre

y1      <- confusionMatrix(data= class.pre, reference = test$class, positive = "1") 
y1
y2      <- as.matrix(y1, what = "overall")
y3      <- as.matrix(y1, what = "classes")
res_log <- rbind(y2, y3)

# KNN pred ---- 
knn_pred  <- predict(knnFit,x_test, type="raw")  # Testing Prediction
confusionMatrix(data= knn_pred,reference = test$class, positive = "1")
y1      <-confusionMatrix(data= knn_pred,reference = y_test, positive = "1")
y2      <- as.matrix(y1, what = "overall")
y3      <- as.matrix(y1, what = "classes")
res_knn <- rbind(y2, y3)

# svm pred ---- 
svm_pred <- predict(svmFit,x_test, type="raw")  # Testing Prediction 
confusionMatrix(data= svm_pred,reference = test$class,positive = "1")
y1      <- confusionMatrix(data= svm_pred,reference = y_test,positive = "1")
y2      <- as.matrix(y1, what = "overall")
y3      <- as.matrix(y1, what = "classes")
res_svm <- rbind(y2, y3)

# Naives Bayes  pred----
nb_pred  <- predict(nbFit,x_test, type="raw")# Testing Prediction 
confusionMatrix(data= nb_pred, reference = test$class, positive = "1")
y1     <- confusionMatrix(data= nb_pred, reference = y_test, positive = "1")
y2     <- as.matrix(y1, what = "overall")
y3     <- as.matrix(y1, what = "classes")
res_nb <- rbind(y2, y3)

# Random forest pred------
rf_pred  <- predict(rfFit,x_test, type="raw")  #Testing Prediction 
confusionMatrix(data= rf_pred, reference = test$class, positive = "1")
y1     <- confusionMatrix(data= rf_pred, reference = y_test, positive = "1")
y2     <- as.matrix(y1, what = "overall")
y3     <- as.matrix(y1, what = "classes")#
res_rf <- rbind(y2, y3)

# neural net pred ----
nn_pred  <- predict(nnFit,x_test, type="raw")   # Testing Prediction 
confusionMatrix(data= nn_pred, reference = test$class, positive = "1")
y1    <- confusionMatrix(data= nn_pred, reference = y_test, positive = "1")
y2    <- as.matrix(y1, what = "overall")
y3    <- as.matrix(y1, what = "classes")
res_nn<- rbind(y2, y3)

# summary result prediction 
result_pred= NULL              
result_pred = rbind(result_pred, data.frame( res_log,res_knn,res_svm, res_nb, res_rf,res_nn))
colnames(result_pred) <- c("Logistic", "KNN","SVM","NAIVES","RF","NN")
print(result_pred)

# export results
write.csv(result_pred,"result_pred_res6_4h.csv")

#************************************************************

# Predict w train
log_train <- predict(logFit,x_train, type="raw")    # Training Prediction 
confusionMatrix(data= log_train,reference = y_train, positive = "1") # Train 
fm_log1 <- confusionMatrix(data= log_train, reference = y_train, mode = "prec_recall") # f-score
fm_log2$byClass["F1"]





fm_knn2 <-confusionMatrix(data= knn_pred, reference = y_test, mode = "prec_recall")
fm_knn1$byClass["F1"]
# Predict training
knn_train <- predict(knnFit,x_train, type="raw")    # Training Prediction 
confusionMatrix(data= knn_train,reference = y_train, positive = "1")
fm_knn1 <-confusionMatrix(data= knn_train, reference = y_train, mode = "prec_recall") # f-score
fm_knn2$byClass["F1"] 

# Predict 
# Test Prediction 
svm_pred  <- predict(svmFit,x_test, type="raw")  # Testing Prediction 
confusionMatrix(data= svm_pred,reference = y_test,positive = "1")
y7 <- confusionMatrix(data= svm_pred,reference = y_test,positive = "1")
y8 <- as.matrix(y7, what = "overall")
y9 <- as.matrix(y7, what = "classes")
res_log <- merge(y8, y9, by = "row.names", all = TRUE)
write.csv(res_log,"svm.csv")

fm_svm2   <-confusionMatrix(data= svm_pred, reference = y_test, mode = "prec_recall")
fm_svm2$byClass["F1"]

# Training 
svm_train <- predict(svmFit,x_train, type="raw") 
confusionMatrix(data= svm_train,reference = y_train,positive = "1")
fm_svm1   <-confusionMatrix(data= svm_train, reference = y_train, mode = "prec_recall")
fm_svm1$byClass["F1"]

# ------------------------------------

# Predict 
# Testing  

res_log <- merge(y11, y12, by = "row.names", all = TRUE)
write.csv(res_log,"nb.csv")


fm_nb2 <-confusionMatrix(data= nb_pred, reference = y_test, mode = "prec_recall") # f-score
fm_nb2$byClass["F1"]

# Training 
nb_train <- predict(nbFit,x_train, type="raw")    # Training Prediction 
confusionMatrix(data= nb_train,reference = y_train,positive = "1")
fm_nb1 <-confusionMatrix(data= nb_train, reference = y_train, mode = "prec_recall") # f-score
fm_nb1$byClass["F1"]


# Predict 
# Testing 

res_log <- merge(y14, y15, by = "row.names", all = TRUE)
write.csv(res_log,"knn.csv")

fm_knn2 <-confusionMatrix(data= knn_pred, reference = y_test, mode = "prec_recall") # f-score
fm_knn2$byClass["F1"]

# Training 
knn_train <- predict(knnFit,x_train, type="raw") #Training Prediction 
confusionMatrix(data= knn_train,reference =y_train,positive = "1")
fm_knn1 <-confusionMatrix(data= knn_train, reference = y_train, mode = "prec_recall")
fm_knn1$byClass["F1"]
#----------------------------


# Training
nn_train <- predict(nnFit,x_train, type="raw")  # Training Prediction 
confusionMatrix(data= nn_train,reference = y_train,positive = "1")
fm_nn1 <-confusionMatrix(data=nn_train, reference = y_train, mode = "prec_recall") # f-score
fm_nn1$byClass["F1"]

# Resume -------------------------------------
results <- resamples(list(Logistic = logFit, SVM = svmFit, Naives=nbFit, KNN = knnFit, RF= rfFit,NNs=nnFit))
summary(results)


# results list
res_test <- list(Logistic = cm_test_log$byClass["Precision"],
                 SVM = cm_test_svm$byClass["Precision"], 
                 KNN =cm_test_knn$byClass["Precision"]
)

summary(res_test)
datatable(filter = 'top', options = list(
  pageLength = 15, autoWidth = F))

# Compute p-value
diffs <- diff(results)
# summarize p-values for pair-wise comparisons
summary(diffs)

#-----------------------
# Table to summary statistical 

tmp <- do.call(data.frame, 
               list(mean = apply(df1[,-8], 2, mean),
                    sd = apply(df1[,-8], 2, sd),
                    median = apply(df1[,-8], 2, median),
                    min = apply(df1[,-8], 2, min),
                    max = apply(df1[,-8], 2, max),
                    n = apply(df1[,-8], 2, length)))
data.frame(t(tmp))# transpuesta 

#------------------------------------


results2 <- resamples(list(Logistic = 60, SVM =61 , Naives=56, KNN = 47, RF= 54,NNs=58))
summary(results)
