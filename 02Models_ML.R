# Data set: CPAP treatment
# Aims:
# To identify potential variables to improve CPAP treatment 
# Predictive Model: SVM, Random forest, KNN. 
#...........................................................

# Load Packages ----
pkgs <-   c("dplyr", "tidyr", "caret","xtable","data.table", "lubridate","ROCR","broom", 
            "klaR","randomForest","corrplot", "nnet","DT","rio","pROC","tidyverse","pscl",
            "tidyverse", "psych", "MASS", "ggplot2", "kernlab", "knitr", "gmodels", 
            "compareGroups","ggpubr") 

sapply(pkgs, require, character.only = T)
#............................................................

# Load and save data----
load(file= "update_analysisCPAP.RData" )
save.image(file="update_analysisCPAP.RData")

#..................................................

# Summary data
dim(cpap3)
names(cpap3)
summary(cpap3)


# Load data ------------
cpap3      <- cpap3[,2:54]
names(cpap3)
str(cpap3)
summary(cpap3)
#.............................................

# Pre-processing data ------
## Convert chr varibles to factor
cpap3       <- as.data.frame(unclass(cpap3))
cpap3$class <- as.factor(ifelse(cpap3$cpap1>=4, 1, 0))  # response variable
cpap4  <- dplyr::select(cpap3, 1:50)
cpap3 <- cbind(cpap4, class=cpap3$class)
cpap3$class
# Relevel variables 
cpap3$class <- relevel(cpap3$class, ref="1")  # Reorder levels of factor 

#.....................................................

# Remove reduntant variable  -----
# Correlation Matrix 
correlationMatrix <- cor(cpap3[,1:35])                              # Select only numerical Variables
hcorrelated       <- findCorrelation(correlationMatrix, cutoff=0.6) # Threshold >0.6, Find Features that are highly corrected 
print(hcorrelated)                                                  # print indexes of highly correlated attributes
highly_cor_var    <- colnames(cpap1[hcorrelated])                   # displaying highly correlated variables
data.frame(highly_cor_var)

# Removing the highly correlated variables
print(cpap_no_hcor <- cpap3[,-hcorrelated])
#....................................................................

# Standarized Numeric variables ----
names(cpap3)
num <- dplyr::select(cpap3,2:36)# Select categorical variables .
cat <- dplyr::select(cpap3,37:55) # Select Numeric var

# standarize Function ---
standarize <- function(x) {
  num   <- x - mean(x)
  denom <- sd(x)
  return (num / denom)
}

# Applying function  
num   <- as.data.frame(lapply(num, standarize)) # apply function 
cpap3 <-cbind(num, cat)                         # merge dataframes 
summary(cpap3)

# complete case analysis
cpap3 <- cpap3[complete.cases(cpap3),] # omit NA values from data set
str(cpap3)
names(cpap3)

# ...............................................



# Preparing for Models  ------ 

# Split dataset
summary(cpap3)
set.seed(2019)
inTrain <- createDataPartition(y=cpap3$class, p = 0.70, list =FALSE)
train   <- cpap3[inTrain,]
test    <- cpap3[-inTrain,]
x_train <- subset(train, select= -c(class))   # only predictors from train 
x_test  <- subset(test, select= -c(class))    # only predictors from test
y_train <- subset(train, select= c(class))    # outcome from train data 1x160
y_test  <- subset(test, select= c(class))    # outcome from test data 1x68

#......................................................

# Building Predictive Models ----
## Model 1: GLM full model  ----- 
set.seed(2019)
glmFit <- glm(class ~.,
              data= train, 
              family=binomial(link="logit"), maxit=100)

# Summary model 
head(tidy(glmFit),20)
sink(file= "logistic2.doc")
summary(glmFit)
exp(coef(glmFit))      # coef values
caret::varImp(glmFit)  
sink()
exp(coef(glmFit))      # coef values
caret::varImp(glmFit)  # variable is the most influential in predicting

#**************************************************************
# Model 2: BACKWARD 
set.seed(2019) 
backward <- glmFit %>%   # We uses this methods for backward
  stepAIC (glmFit, direction=c("backward"))

sink(file = "back1.doc")
summary(backward)
sink()
formula(backward)  # variables include in model

#........................................................

# MODEL 3: FORWARD  ----
# Null model , we used for forward meth 
set.seed(2019)
null_model <- glm(class ~1 ,data=train,family=binomial(link="logit"))
null_model$aic
# full model 
full_model <- glm(class ~.,data=train,family=binomial(link="logit"))
full_model$aic

## forward 
forward   <- step(null_model,scope=list(lower=formula(null_model),upper=formula(full_model)), direction="forward")

sink(file="forward.doc")
summary(forward)
sink()

# For backward
#backward  <- step(null_model,scope=list(lower=formula(null_model),upper=formula(full_model)), direction="backward")
#summary(backward)

#..................................................
# Model 4: Both  ----  
set.seed(2019)
both <-step(null_model,scope=list(lower=formula(null_model),upper=formula(full_model)), direction="both")
sink(file="both.doc")
summary(both)
sink()
#..................................................

# Model 5: Lasso ------
control <- trainControl(method="repeatedcv",  # model specific training parameter
                        number=10,
                        repeats=3,
                        verboseIter=FALSE)

set.seed(2019)                         # for reproducibility
x <- model.matrix(class~., train)[,-1] # features variable as matrix
y <- ifelse(train$class == 1, 1, 0)    # Outcome variable
y <- train$class
head(y, 10)

# Grid for lambda 
lambda <- 10^seq(-3, 3, length = 100)

# Tuning lasso parameters with caret package.---- 
set.seed(2019)  # for reproducibility
cv.lasso  <- train(class ~., 
                   data = train, method = "glmnet",
                   standardize=FALSE, # TRUE si no
                   trControl = control,
                   tuneGrid = expand.grid(alpha = 1, lambda = lambda))

coef(cv.lasso$finalModel, cv.lasso$bestTune$lambda)# coefficient
cv.lasso$bestTune  # best tune, lambda value 

# Tuning glmnet function   ----- 
set.seed(123) 
library(glmnet)
cv.lasso2 <- cv.glmnet(x = x,
                       y = y,
                       family="binomial", 
                       standardize = FALSE,
                       alpha = 1 # LASSO 
)

coef(cv.lasso2) # coefficients selected  

plot (cv.lasso2, xvar="lambda", xlab = "lambda",         # Show coefficients, lambda value and number of features selected 
      ylab = "Value of the coefficients", label =TRUE)  # xvar="dev"

# Print the minimum lambda - regularization factor
cv.lasso2$lambda.min  # lambda for this min MSE
cv.lasso2$lambda.1se  # lambda for this MSE

## Fit the Lasso model with the lambda value that minimizes error (deviance)
library(glmnet)
lasso <- glmnet(x, y, 
                family = "binomial",
                alpha = 1,
                lambda = bestlambda, 
                standardize = FALSE)
# Coefficients
coef(lasso)
# Plot 
plot(lasso, xlab=" Log. of Lambda", ylab="Binomial deviance")

## Lasso model
lassocoefficients <- predict (lasso, type = "coefficients", s = bestlambda)
lassocoefficients

## Sparse LASSO model  (cvlasso$lambda.1se)
lassosparse <- glmnet(x, y, family = "binomial", alpha = 1, lambda = lasso$lambda.1se, standardize = FALSE)
lassosparse                      

## Lasso model
lassosparsecoefficients <- predict (lassosparse, type = "coefficients", s = lasso$lambda.1se)
lassosparsecoefficients

# plot the most relevant variables -----
library(broom)
coef(lasso, s = "lambda.min") %>%
  tidy() %>%
  filter(row != "(Intercept)") %>%
  ggplot(aes(value, reorder(row, value), color = value > 0)) +
  geom_point(show.legend = FALSE) +
  ggtitle("Influential variables") +
  xlab("Coefficient") +
  ylab(NULL)

# extract coefficients for the best performing model
coef <- data.frame(coef.name = dimnames(coef(lasso$finalModel,s=lasso$bestTune$lambda))[[1]], 
                   coef.value = matrix(coef(lasso$finalModel,s=lasso$bestTune$lambda)))

print(coef <- coef[-1,])     # exclude the (Intercept) term

picked_features <- nrow(filter(coef,coef.value!=0))  #   

not_picked_features <- nrow(filter(coef,coef.value==0))

Output
cat("Lasso picked",picked_features,"variables and eliminated the other",
    not_picked_features,"variables\n")

coef <- arrange(coef,-coef.value) # Coeficients 
coef

# extract the top 10 and bottom 10 features
imp_coef <- rbind(head(coef,10),
                  tail(coef,10))

ggplot(imp_coef) +
  geom_bar(aes(x=reorder(coef.name,coef.value),y=coef.value),
           stat="identity") +
  ylim(-1.5,0.6)   +
  coord_flip()     +
  ggtitle("Coefficents in the Lasso Model") +
  theme(axis.title=element_blank())

#*************************************************************

# x <- model.matrix(class ~ ., data = train)[,-which(names(train) %in% "class")]# 
# # Dumy code categorical predictor variables
# x <- model.matrix(diabetes ~., train.data)[,-1]  # Remove intercept
# y <- train$class     # Create the y vector (as numeric)
# 
# # Building a grid of lambda values 
# grid <- 10 ^ seq(0.01,-2, length = 100)

# ## create a grid with unkwon lambda
# model_lasso1  <- glmnet(x, y, family = "binomial",  # by default glmnet applies 100 of ?? 
#                        alpha = 1, lambda = grid)  # standarised=TRUE

# # coeficientes
# coef(model_lasso1)   # Model`s Coefficients 
# 
# # plot 
# plot (model_lasso1, xvar="lambda", xlab = "lambda", 
#               ylab = "Value of the coefficients", label =TRUE)  # xvar="dev"

# ## Calculate the best value for lambda using cross validation
# set.seed(2019)
# cv.lasso <- cv.glmnet(x, y, family="binomial",alpha = 1)
# 
# # Plot 
# plot(cv.lasso, xlab = "Logarithm of lambda", ylab = "Binomial deviance", label =TRUE)


# Coeficients 
coef(cv.lasso, cv.lasso$lambda.min)

## Determine the best lambda value
print(bestlambda <- cv.lasso$lambda.min)

#..........................................................
names(cpap3)
# RANDOM FOREST MODEL 6 -----
# Tunning Random Forest 
set.seed (2019)
## Divide the training set into 3 folds
folds   <- sample(rep(1:3, length = nrow(train)))
## Make sure folds have more or less the same size
table (folds)

mtryvalues <- 1:(length(train)-1) # Replace for total number of features of dataset 

set.seed(2019)     #  an arbitrary number
## Set an empty data frame with the proportion misclassified 
##(rows will be different mtry, and columns values of k) 
proportionerror <- data.frame(matrix(0, ncol = 3, nrow = 50)) # change for number total of  features
colnames(proportionerror) <- c("k = 1", "k = 2", "k = 3")

## Calculate the best mtry by cross-validation
## Create a loop with 3 folds
for (k in 1:3) {
  ## Create a loop with 18 values of mtry
  for (j in 1:50) {
    ## Calculate the random forest model based on 2 of the folds
    forest <- randomForest(class ~ ., data = train[folds != k,], 
                           mtry = mtryvalues[j], ntree = 1000, importance = TRUE)
    ## Predict performance on the remaining fold not used for fitting the model
    prediction <- predict(forest, newdata = train[folds == k,-which(names(train) == "class")]) 
    ## Confusion matrix for the prediction
    confusionmatrix <- table(prediction, train[folds == k, which(names(train) == "class")]) 
    ## Misclassification error
    proportionerror[j, k] <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
      (confusionmatrix[1, 1] + confusionmatrix[1, 2] + 
         confusionmatrix[2, 1] + confusionmatrix[2, 2])
  }
}

## Calculate the mean prediction error for each mtry
proportionerror$meanerror <- rowMeans(proportionerror)
proportionerror$`k = 1`    # Expresa la proporcion de error en cada fold
mean(proportionerror$`k = 3`) # mean of error 

## Plot the proportion of errors in relation with the number of variables in mtry
plot(mtryvalues, proportionerror$meanerror, pch = 16, cex = 2, 
     col = "red", type = "o", lwd = 2,
     xlab = "Number of variables used at each node", 
     ylab = "Misclassification in the fold not used for developing the model")

## Identify the lowest mtry within 1 standard error of the mtry with minimum error
mtryminimumerror <- which(proportionerror$meanerror == min(proportionerror$meanerror))[1]
mtryminimumerror

# Fit forest model with mtry selected
forest <- randomForest(class ~ ., data = train, 
                       mtry = mtryminimumerror, ntree = 1000, importance = TRUE)
forest$importance

# # fit Forest2 with tunning mtry= 7.69
forest2 <- randomForest(class ~ ., data = train, 
                        mtry = (length(cpap3)/3), ntree = 1000, importance = TRUE))

# Plots relevant varaibles 
varImpPlot(forest, main ="Variables sorted by importance based on mean decrease accuracy and on mean decreased Gini")

# Regarding to mtry2
# varImpPlot(forest3, main ="Variables sorted by importance based on mean decrease accuracy and on mean decreased Gini")

#forest2$importance
#............................................................
# Approach 6  SUPPORT VECTOR MACHINE
# Tunning Linear- SVM - ---------------------------- 
library(e1071)
svmparameterslinear <- tune(svm, class ~ ., data = train, 
                            kernel = "linear", 
                            ranges = list(cost = c(0.001,0.05,0.01, 0.1, 1,2, 5, 10, 100)),
                            tunecontrol = tune.control(cross = 10), 
                            scale = TRUE, probability = TRUE)

sink(file="svmlinear1.doc")
summary(svmparameterslinear)
svmlinear <- svmparameterslinear$best.model## Parameters for the best model
print(svmlinear)
sink()


#.................................................

# Approach 6: poly svm   
#Polynomial SVM ----- 
svmparameterspoly <- tune(svm, class ~ ., data= train, 
                          kernel = "polynomial", 
                          ranges = list(cost = list(cost = c(0.001,0.05,0.01, 0.1, 1,2, 5, 10, 100)),
                                        degree = seq(1, 6, 1)),
                          tunecontrol = tune.control(cross = 10), 
                          scale = TRUE, probability = TRUE)

svmparameterspoly
svmparameterspoly$best.parameters # Parameters for the best model

## Select the best model
svmpolynomial  <- svmparameterspoly$best.model

sink(file="svmpolinomial.doc")
summary(svmpolynomial)
sink()
#.....................................................


# ASSESSMENT OF THE MODEL'S PERFORMANCE

## Fit full Model 1------------------------------------------------------
predictionglm <- predict(glmFit, newdata = x_test, type="response")  # Probabilidad de que sea 1(no cumple)
predglm01     <- ifelse(predictionglm> 0.5, 1,0)
predglm01     <- as.factor(predglm01)

# AUC VALUE GLM FULL --
print(aucglm   <- auc(test$class,predictionglm))
print(ciaucglm <- ci.auc(test$class, predictionglm))

# Compute error ---
confusionmatrix  <- table(predglm01, 
                          test[,which(names(test) == "class")])
## Missclasification --
misclassificationglm <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])

misclassificationglm


## Matrix confusion
print(cm.glm <- confusionMatrix(data=predglm01,reference = test$class, positive = "0")) 

# Threshold and AUC -------------------------------------
plot.roc(test$class, predictionglm,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE,ci.auc=TRUE) # a

## Calculate optimal threshold with Youden's index
rocglm  <- roc(test$class, predictionglm)
print(bestglm <- coords(rocglm, "b", ret = "threshold", best.method = "youden"))


# Threshold Adj. fit Model ------
predglm01  <- ifelse( predictionglm > bestglm, 1, 0)
predglm01  <- as.factor(predglm01)

# Matrix Confusion 
print(cm.glm01 <- confusionMatrix(data=predglm01, reference = test$class, positive = "1")) 

# Compute error --------------------------
confusionmatrix  <- table(predglm01, 
                          test[,which(names(test) == "class")])
## Missclasification --
misclassificationglm <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationglm


# values cm.glm ---- 
sink(file="gml_performance.doc")
print(cm.glm01 <- confusionMatrix(data=predglm01, reference = test$class, positive = "1")) 
acc.glm <- cm.glm01$overall["Accuracy"]
sen.glm <- cm.glm01$byClass["Sensitivity"]
spe.glm <- cm.glm01$byClass["Specificity"]
ppv.glm <- cm.glm01$byClass["Pos Pred Value"]
npv.glm <- cm.glm01$byClass["Neg Pred Value"]
print(f1.glm  <- cm.glm01$byClass["F1"])
sink()



#****************************************************************************
## Model 2: Backward prediction---- 
library(pROC); library(ROCR)
predictionbackward <- predict(backward, newdata= x_test, type ="response") 

# AUC BACKWARD ---- 
print(aucbackward <- auc(test$class, predictionbackward))
print(cibackward  <- ci.auc(test$class, predictionbackward))

# Prediction based on misclassification error
predictionbackward01 <- ifelse(predictionbackward > 0.5, 1, 0)
predictionbackward01 <- as.factor(predictionbackward01)
confusionmatrix      <- table(predictionbackward01, 
                              test[ , which(names(test) == "class")])
misclassificationbackward <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationbackward
sink(file="backward.doc")
print(cm.backward<- confusionMatrix(data=predictionbackward01 ,reference = test$class, positive = "0"))
sink()
# Threshold and AUC -------------------------------------
plot.roc(test$class, predictionbackward,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) 

## Calculate optimal threshold with Youden's index
rocbackward  <- roc(test$class, predictionbackward)
bestbackward <- coords(rocbackward, "b", ret = "threshold", best.method = "youden")
bestbackward

## Prediction based on misclassification error -------
predictionbackward01 <- ifelse(predictionbackward > bestbackward, 1, 0)
predictionbackward01 <- as.factor(predictionbackward01)
confusionmatrix      <- table(predictionbackward01, 
                              test[,which(names(test) == "class")])
## Missclasification --
misclassificationbackward <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationbackward

## Prediction based on sensitivity, specificity, positive predictive value, 
print(cm.backward<- confusionMatrix(data=predictionbackward01 ,reference = test$class, positive = "0"))

# values cm.glm ---- 
acc.backward <- cm.backward$byclass["Accuracy"]
sen.backward <- cm.backward$byClass["Sensitivity"]
spe.backward <- cm.backward$byClass["Specificity"]
ppv.backward <- cm.backward$byClass["Pos Pred Value"]
npv.backward <- cm.backward$byClass["Neg Pred Value"]
print(f1.backward  <- cm.backward$byClass["F1"])

#*****************************************************************

## Model3: FORWARD stepwise ---- 
predictionforward <- predict(forward, type = "response", newdata = test[,-which(names(test)=="class")]) 
# AUC FORWARD ----
print(aucforward  <- auc(test$class, predictionforward))
print(ciforward   <- ci.auc(test$class, predictionforward))

## Prediction based on misclassification error
predictionforward01  <- ifelse(predictionforward > 0.5, 1, 0)
predictionforward01  <- as.factor(predictionforward01)
confusionmatrix       <- table(predictionforward01, 
                               test[ , which(names(test) == "class")])
misclassificationforward <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationforward

sink(file="forward.doc")
print(cm.forward<- confusionMatrix(data=predictionforward01 ,reference = test$class, positive = "1"))
sink()
# Threshold and AUC -------------------------------------
plot.roc(test$class, predictionforward,
         main="Comparison of AUC backward_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a
#-------
## Calculate optimal threshold with Youden's index
rocforward        <- roc(test$class, predictionforward)
print(bestforward <- coords(rocforward, "b", ret = "threshold", best.method = "youden"))

## Prediction based on misclassification error -----
predictionforward01 <- ifelse(predictionforward > bestforward, 0, 1)
predictionforward01 <- as.factor(predictionforward01)

## Missclasification -----
confusionmatrix          <- table(predictionforward01,test[,which(names(test) == "class")]) 
misclassificationforward <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationforward

## Prediction based on sensitivity, specificity, positive predictive value, 
print(cm.forward <-confusionMatrix(data=predictionforward01,reference=test$class,positive="1"))
# values cm.glm ---- 
acc.forward <- cm.forward$byclass["Accuracy"]
sen.forward <- cm.forward$byClass["Sensitivity"]
spe.forward <- cm.forward$byClass["Specificity"]
ppv.forward <- cm.forward$byClass["Pos Pred Value"]
npv.forward <- cm.forward$byClass["Neg Pred Value"]
print(f1.forward  <- cm.forward$byClass["F1"])                   

#**********************************************************
## Both Method -----------------------------------
predictionboth <- predict(both, type = "response", newdata = test[,-which(names(test)=="class")]) 

# AUC Both 
print(aucboth <- auc(test$class, predictionboth))
print(ciboth  <- ci.auc(test$class, predictionboth))

## Prediction based on misclassification error 
predictionboth01 <- ifelse(predictionboth > 0.5, 1, 0)
predictionboth01 <- as.factor(predictionboth01)

## Misclassification  
confusionmatrix       <- table(predictionboth01,test[,which(names(test) == "class")]) 
misclassificationboth <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationboth

# Matrix confusion
sink(file="both.doc")
print(cm.both <-confusionMatrix(data=predictionboth01,reference=test$class,positive="1"))
sink()
## auc, threshold 
plot.roc(test$class, predictionboth,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

## Calculate optimal threshold with Youden's index
rocboth  <- roc(test$class, predictionboth)
print(bestboth <- coords(rocboth, "b", ret = "threshold", best.method = "youden"))

## Prediction based on misclassification error 
predictionboth01 <- ifelse(predictionboth > bestboth, 1, 0)
predictionboth01 <- as.factor(predictionboth01)

## Missclasification 
confusionmatrix  <- table(predictionboth01,test[,which(names(test) == "class")]) 
misclassificationboth <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationboth

## Prediction based on sensitivity, specificity, positive predictive value, 
print(cm.both <-confusionMatrix(data=predictionboth01,reference=test$class,positive="1"))
acc.both <- cm.both$byclass["Accuracy"]
sen.both <- cm.both$byClass["Sensitivity"]
spe.both <- cm.both$byClass["Specificity"]
ppv.both <- cm.both$byClass["Pos Pred Value"]
npv.both <- cm.both$byClass["Neg Pred Value"]
print(f1.both  <- cm.both$byClass["F1"])


# ************************************************************
# Prediction LASSO 
## Create the x matrix
x <- model.matrix(class~., data = test)[ , -which(names(test) %in% "class")]
## Create the y vector
y <- as.numeric(test$class)

# Lasso predictions
predictionlasso <- predict(lasso, newx = x, type = "response")# lasso, · prob
predictionlasso <- as.numeric(predictionlasso)

# Area under the curve
print(auclasso <- auc(y, predictionlasso))
print(cilasso <- ci.auc(y, predictionlasso))

## Prediction based on misclassification error
predictionlasso01 <- ifelse(predictionlasso > 0.5, 1, 0)
predictionlasso01 <- as.factor(predictionlasso01)

confusionmatrix   <- table(predictionlasso01, 
                           test[ , which(names(test) == "class")])

misclassificationlasso <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationlasso

# Matrix confusion
sink(file="lasso.doc")
print(cm.lasso <-confusionMatrix(data=predictionlasso01, reference= test$class, positive="1"))

## Calculate optimal threshold with Youden's index
roclasso <- roc(test$class, predictionlasso)
bestlasso <- coords(roclasso, "b", ret = "threshold", best.method = "youden")
bestlasso

# Threshold and AUC -------------------------------------
plot.roc(test$class, predictionlasso,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a


## Prediction based on sensitivity, specificity, positive predictive value, 
print(cm.lasso <-confusionMatrix(data=predictionlasso01,reference=test$class,positive="1"))

## Prediction based on misclassification error
predictionlasso01 <- ifelse(predictionlasso > bestlasso, 1, 0)
predictionlasso01 <- as.factor(predictionlasso01)
confusionmatrix   <- table(predictionlasso01, 
                           test[ , which(names(test) == "class")])

misclassificationlasso <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationlasso
y_test <- as.factor(test.data$class)
## Prediction based on sensitivity, specificity, positive predictive value, 
print(cm.lasso <-confusionMatrix(data=predictionlasso01,reference=y,positive="1"))
acc.lasso <- cm.lasso$byclass["Accuracy"]
sen.lasso <- cm.lasso$byClass["Sensitivity"]
spe.lasso <- cm.lasso$byClass["Specificity"]
ppv.lasso <- cm.lasso$byClass["Pos Pred Value"]
npv.lasso <- cm.lasso$byClass["Neg Pred Value"]
f1.lasso  <- cm.lasso$byClass["F1"] 

#**************************************************************
## Prediction Random Forest 

predictionrf <- predict(forest, newdata=x_test, type ='prob')[,2] # para succesful
print(aucrf  <- auc(test$class, predictionrf))
print(cirf   <- ci.auc(test$class, predictionrf))

## Prediction based on sensitivity, specificity, positive predictive value, 
predictionrf01  <- ifelse(predictionrf > 0.5, 1, 0)
predictionrf01  <- as.factor(predictionrf01)

sink("file=random.doc")
print(cm.rf <- confusionMatrix(data=predictionrf01 ,reference = test$class, positive = "1"))
sink()

## Missclasification -----
confusionmatrix      <- table(predictionrf01,test[,which(names(test)=="class")]) 
misclassificationrf  <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationrf    # Error value

# Compute of Threshold and AUC -------------------------------------
plot.roc(test$class, predictionrf,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

#---------------------------------------
## Option 2: Calculate optimal threshold with Youden's index
rocrf  <- roc(test$class, predictionrf)
bestrf <- coords(rocrf, "b", ret = "threshold", best.method = "youden")
print(bestrf)

#------------------
## Prediction based on misclassification error -------------------
predictionrf01  <- ifelse(predictionrf > bestrf, 1, 0)
predictionrf01  <- as.factor(predictionrf01)
confusionmatrix <- table(predictionrf01,test[,which(names(test)=="class")]) 

## Missclasification -----
misclassificationrf <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationrf

## Prediction based on sensitivity, specificity, positive predictive value, 
print(cm.rf01 <- confusionMatrix(data=predictionrf01 ,reference = test$class, positive = "1"))
acc.rf <- cm.rf01$overall["Accuracy"]
sen.rf <- cm.rf01$byClass["Sensitivity"]
spe.rf <- cm.rf01$byClass["Specificity"]
ppv.rf <- cm.rf01$byClass["Pos Pred Value"]
npv.rf <- cm.rf01$byClass["Neg Pred Value"]
f1.rf  <- cm.rf01$byClass["F1"]
# -----------------------------------------------------------
# Approach 1: ----
predictionsvmlinear <- predict(svmlinear, probability = TRUE,
                               newdata = test[ ,- which(names(test) == "class")])
predictionsvmlinear <- attr(predictionsvmlinear, "probabilities")[ , 2]  # probabilidad que sea cero 

# AUC and interval
print(aucsvmlinear <- auc(test$class, predictionsvmlinear))
print(cisvmlinear  <- ci.auc(test$class, predictionsvmlinear))

## Prediction based on misclassification error
predictionsvmlinear01 <- predict(svmlinear, 
                                 newdata = test[ , - which(names(test) == "class")])

confusionmatrix       <- table(predictionsvmlinear01, test[ , which(names(test) == "class")])

misclassificationsvmlinear <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationsvmlinear

# Confusion Matrix
sink(file="svmlinear.doc")
confusionMatrix(data= predictionsvmlinear01,reference = test$class, positive = "1")
sink()

# Compute of Threshold and AUC -------------------------------------
plot.roc(test$class, predictionsvmlinear,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a

# Calculate optimal threshold with Youden's index ------------------
rocsvmlinear <- roc(test$class, predictionsvmlinear)
bestsvmlinear <- coords(rocsvmlinear, "b", ret = "threshold", best.method = "youden")
bestsvmlinear

## Prediction based on misclassification error
predictionsvmlinearyouden <- ifelse(predictionsvmlinear > bestsvmlinear, 1, 0)
predictionsvmlinearyouden <- as.factor(predictionsvmlinearyouden)
confusionmatrix <- table(predictionsvmlinearyouden, 
                         test[ , which(names(test) == "class")])

misclassificationsvmlinear <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationsvmlinear

# Matrix confusion
print(cm.svmlinear <- confusionMatrix(data= predictionsvmlinearyouden,reference = test$class, positive = "1"))
acc.svmlinear <- cm.svmlinear$overall["Accuracy"]
sen.svmlinear <- cm.svmlinear$byClass["Sensitivity"]
spe.svmlinear <- cm.svmlinear$byClass["Specificity"]
ppv.svmlinear <- cm.svmlinear$byClass["Pos Pred Value"]
npv.svmlinear <- cm.svmlinear$byClass["Neg Pred Value"]
f1.svmlinear  <- cm.svmlinear$byClass["F1"]


#**************************************************************
## Evaluation of  Poly SVM -------
## Prediction based on area under the curve
predictionsvmpolynomial <- predict(svmpolynomial, 
                                   probability = TRUE, 
                                   newdata = test[ , - which(names(test) == "class")]) 
predictionsvmpolynomial <- attr(predictionsvmpolynomial, "probabilities")[ , 2]

print(aucsvmpolynomial  <- auc(test$class, predictionsvmpolynomial))
print(cisvmpolynomial   <- ci.auc(test$class, predictionsvmpolynomial))


## Prediction based on misclassification error -------------------
predictionsvmpolynomial01 <- predict(svmpolynomial, 
                                     newdata = test[ , - which(names(test) == "class")])
confusionmatrix <- table(predictionsvmpolynomial01, 
                         test[ , which(names(test) == "class")])

misclasvmpolynomial <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclasvmpolynomial

sink(file="polynomial.doc")
confusionMatrix(data= predictionsvmpolynomial01,reference = test$class, positive = "1")
sink()

# Compute of Threshold and AUC -------------------------------------
plot.roc(test$class, predictionsvmpolynomial,
         main="Comparison of AUC GLM_FULL MODEL ", percent=TRUE,
         ci=TRUE, of="thresholds", # compute  the threshold of prediction 
         thresholds="best", # select the (best) threshold
         print.thres="best", 
         print.auc=TRUE, ci.auc=TRUE) # a


## Calculate optimal threshold with Youden's index
rocsvmpolynomial <- roc(test$class, predictionsvmpolynomial)
bestsvmpolynomial <- coords(rocsvmpolynomial, "b", ret = "threshold", best.method = "youden")
bestsvmpolynomial

## Prediction based on misclassification error
predictionsvmpolynomialyouden01 <- ifelse(predictionsvmpolynomial > bestsvmpolynomial, 1, 0)
predictionsvmpolynomialyouden01 <- as.factor(predictionsvmpolynomialyouden01)
confusionmatrix <- table(predictionsvmpolynomialyouden01, 
                         test[ , which(names(test) == "class")])

misclassificationsvmpolynomialyouden <- (confusionmatrix[2, 1] + confusionmatrix[1, 2]) / 
  (confusionmatrix[1, 1] + confusionmatrix[1, 2] + confusionmatrix[2, 1] + confusionmatrix[2, 2])
misclassificationsvmpolynomialyouden


# confusion Matrix
sink(file="poly2.doc")
print(cm.svmpoly <-confusionMatrix(data=predictionsvmpolynomialyouden01,
                                   reference = test$class, positive = "1"))
sink()

acc.svmpoly <- cm.svmpoly$overall["Accuracy"]
sen.svmpoly <- cm.svmpoly$byClass["Sensitivity"]
spe.svmpoly <- cm.svmpoly$byClass["Specificity"]
ppv.svmpoly <- cm.svmpoly$byClass["Pos Pred Value"]
npv.svmpoly <- cm.svmpoly$byClass["Neg Pred Value"]
f1.svmpoly  <- cm.svmpoly$byClass["F1"]


##  Summarize results 

## Model names
modelnames <- c("Forward","Backward", "Both", "Lasso", "Random Forest", 
                "SVM linear", "SVM polynomial")
## Outcome names
outcomenames <- c("Acurracy", "AUC 95%CI:L","Sensitivity", "Specificity",
                  "PPV", "NPV", "Misclassification", "Kappa")

## Model area under the curve
modelauc <- c(aucforward, 
              aucbackward, aucboth, auclasso, aucfr, aucsvmlinear, 
              aucsvmpoly)
## Model confidence interval for area under the curve: lower
modelcilower <- c(ciforward[1], cibackward[1], ciboth[1], cilasso[1], cirf[1], 
                  cisvmlinear[1], cisvmpolynomial[1])
## Model confidence interval for area under the curve: upper
modelciupper <- c(ciforward[3], cibackward[3], ciboth[3], cilasso[3], cirf[3], 
                  cisvmlinear[3], cisvmpolynomial[3])

## Model misclassification error
modelmisclassification <- c(misclassificationforward, misclassificationbackward, 
                            misclassificationboth, misclassificationlasso, 
                            misclassificationrf, misclassificationsvmlinear, 
                            misclassificationsvmpolynomial)
## Model sensitivity
modelsensitivity <- c(sen.forward, sen.backward, sen.both, 
                      sens.lasso, sen.rf, sensitivitysvmlinear, 
                      sen.svmpoly)
## Model specificity

modelspecificity <- c(spe.forward, spe.backward, spe.both, specificitylasso, spe.rf, 
                      spec.svmlinear, spec.svmpoly)

## Model positive predictive value
modelppv <- c(ppv.forward, ppv.backward, ppv.both, ppv.lasso, 
              ppv.rf, ppv.svmlinear, ppv.svmpoly)

## Negative predictive value
modelnpv <- c(npv.forward, npv.backward, npv.both, npv.lasso, 
              npv.rf, npv.svmlinear, npv.svmpoly)

# Model f1
modelf1 <- c(f1.forward, f1.backward, f1.both, f1.lasso, 
             f1.rf, f1.svmlinear, f1.svmpoly)

## Final table
results <- data.frame(modelauc, modelcilower, modelciupper,  
                      modelmisclassification, modelsensitivity, modelspecificity, modelppv,
                      modelnpv,modelf1 )
rownames(results) <- modelnames
colnames(results) <- outcomenames
results
