
#######################################################################################################                                                                                              #
#              #Breast Cancer Prediction using machine learning methodologies.                        #
#                               #Research Project                                                     #
#                               #MSc Data Analytics                                                   #
#                             #Pranav Kiran Rajhans                                                   #                   #
#                                  # x16149645                                                        #
#######################################################################################################
#Installation of packages
#######################################################################################################
install.packages("corrplot")
install.packages("Boruta")
install.packages("REdaS")
install.packages("randomForest")
install.packages("caret")
install.packages("class")
install.packages("e1071")
install.packages("gbm")

#######################################################################################################
#Data Preparation  
#######################################################################################################
#Set the directory path 
setwd("/Users/pranav/Desktop/CANCER") 
#######################################################################################################
#Data loading 
breastcancer <- read.csv("data.csv",stringsAsFactors = FALSE)  
#######################################################################################################
#structure of the data 
str(breastcancer)
#######################################################################################################
#Unnecessary column are removed from the data set  
breastcancer<- breastcancer[-1]
breastcancer<- breastcancer[-32]

#######################################################################################################
#Data Manipulation and conversion in to the requrired fromat
breastcancer$diagnosis [breastcancer$diagnosis =="M"] <- 1
breastcancer$diagnosis [breastcancer$diagnosis =="B"] <- 0
breastcancer$diagnosis <- as.character(breastcancer$diagnosis)
breastcancer$diagnosis <- as.numeric(breastcancer$diagnosis)
#######################################################################################################
#Exploratory Data analytics 
#######################################################################################################

#Histograms of the variables and their frequency

hist(breastcancer$diagnosis,col="red")
hist(breastcancer$concavity_mean,col = "blue")
hist(breastcancer$radius_mean,col = "red")
hist(breastcancer$perimeter_mean,col="blue")
hist(breastcancer$area_mean,col="red")
hist(breastcancer$compactness_mean,col="blue")
hist(breastcancer$concave.points_mean,col="red")
hist(breastcancer$perimeter_worst,col="blue")
hist(breastcancer$radius_worst,col = "red")
hist(breastcancer$area_worst,col = "blue")
#######################################################################################################
#Outliers from the data 

boxplot(breastcancer$radius_mean)
boxplot(breastcancer$texture_mean)
boxplot(breastcancer$perimeter_mean)
boxplot(breastcancer$area_mean)
boxplot(breastcancer$smoothness_mean)
boxplot(breastcancer$compactness_mean)
boxplot(breastcancer$concavity_mean)
boxplot(breastcancer$concave.points_mean)
boxplot(breastcancer$symmetry_mean)
boxplot(breastcancer$fractal_dimension_mean)
boxplot(breastcancer$radius_se,breastcancer$texture_se,breastcancer$perimeter_se)
boxplot(breastcancer$area_se)
boxplot(breastcancer$smoothness_se)
boxplot(breastcancer$compactness_se)
#library(car)
plot(breastcancer$radius_mean,breastcancer$radius_worst)
plot(breastcancer$area_mean,breastcancer$perimeter_mean,xlim = c(-2,2), ylim = c(-2,2), panel.first = grid())

#ggplot(data = breastcancer, mapping = aes(x =diagnosis, y =area_mean )) +
  geom_boxplot()

ggplot(data = breastcancer) +
  geom_count(mapping = aes(x = diagnosis, y = texture_worst ))

ggplot(data = breastcancer) +
  geom_count(mapping = aes(x = diagnosis, y = area_mean))
#######################################################################################################
#######################################################################################################

#The number of attributes which are highly correlated with each other  
corr_mat <- cor(breastcancer[,c(2,3,5,7,8,9,22,24,25,28)]) # Selected attributes which are highly correlated with each other 
corrplot(corr_mat,method = "color", outline = T, addgrid.col = "green",
         order="hclust", addrect = 4, rect.col = "black", rect.lwd = 5,cl.pos = "b", 
         tl.col = "indianred4", tl.cex = 0.5, cl.cex = 0.5, addCoef.col = "white", 
         number.digits = 2, number.cex = 0.75, col = colorRampPalette(c("darkred","white","midnightblue"))(100))
#######################################################################################################
#######################################################################################################


# One of the Feature selection methodology which indicated which varibale is done more impact on the prediction varibale 
#######################################################################################################

# required library installation 
library(Boruta) 
library(ranger)
# Changing the name of the column 
names(breastcancer) <- gsub("_", "", names(breastcancer)) # gsub is used for the replacement 

convert<- c(1) #the column 1 is tored in the varibale convert 
breastcancer[,convert] <- data.frame(apply(breastcancer[convert], 1, as.factor)) # data is converted in to the factor and save into the data frame

summary(breastcancer)
set.seed(123)
boruta.train <- Boruta(diagnosis~., data = breastcancer, doTrace = 2)
print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n") # plot the boruta graph with out any lables 
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median)) 
                                                      # code is used for the attributes labelling 
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
final.boruta <- TentativeRoughFix(boruta.train) #final boruta graph is printed 
print(final.boruta)
#######################################################################################################
# The data preparation for model building. Data is divided in to two section Training and Testing 
#######################################################################################################


#Data Division
str(breastcancer)
table(breastcancer$diagnosis)
Train <- breastcancer[1:400,]
str(Train)
Test<- breastcancer[401:569,]

# adata conversion as per requirement of the model 
Test$diagnosis<- as.character(Test$diagnosis) 
Test$diagnosis<- as.numeric(Test$diagnosis)
Test$diagnosis<- as.factor(Test$diagnosis)
str(Test)
Test$diagnosis
str(Train)
train2 <- Train 
test2<- Test
test2$diagnosis <- as.numeric(test2$diagnosis)
train2$diagnosis <- as.numeric(train2$diagnosis)
train2$diagnosis [train2$diagnosis =="2"] <- 0
test2$diagnosis[test2$diagnosis=="2"] <- 0 
#######################################################################################################
#Bartlett test and Kaiser-Meyer-Olkin test for Principal Component Aanalysis (PCA)
#######################################################################################################
#Bartlett test
r <- cor(Train[,-1])
cortest.bartlett(r) 
# if the p value is 1 then the relation between the varibales is strong and Principal Componant Analysis can be processed and if the values is less than 0.50 then PCA is not required 

#KMO
library(psych)
KMO(train2)  
# This test of for sample effectiveness if the value is between 0.80 to 1.0 then most effective variable and if its below 0.60 then less effective varibale 


#######################################################################################################
# Model Building Process and analysis 
#######################################################################################################
#Random Forest#
#library installation 
library(randomForest)
library(caret)

#Model Bulding 
rf <- randomForest(diagnosis~.
                   ,data = Train , type = "classification")
rf

#Prediction of the randomforest
rf1<- predict(rf,Test)

#Variable importance Graph
varImpPlot(rf)

#Confusion Matrix and statistics
confusionMatrix(rf1,Test$diagnosis)

#visualisation of prediction 
plot(rf1,Test$diagnosis,xlab="Actual",ylab="Predicted")

#######################################################################################################
#KNN
#######################################################################################################


#library installation 
library(class)
#Normalize funtion to normalize the data 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
# data frame criation of normalised data 
B_Data_n <- as.data.frame(lapply(breastcancer[2:30], normalize))

B_train_labels <- breastcancer[1:400, 1]
B_test_labels <-breastcancer[401:569, 1]
#Model Building and prediction 
B_test_pred <- knn(train = Train, test = Test, cl = B_train_labels, k = 10) 
table(B_test_labels,B_test_pred)

#chi-square value test 
CrossTable(x = B_test_labels, y =B_test_pred, prop.chisq=FALSE)
#Confusion matrix and statistics 
confusionMatrix(B_test_pred,B_test_labels)
#Visualisation of the prediction outcome 
plot(B_test_pred,B_test_labels,xlab="Actual",ylab="Predicted")

#######################################################################################################
#SVM
#######################################################################################################


#library installation 
library(e1071)
svm_model<- svm(diagnosis~+concave.pointsmean+radiusworst+perimeterworst+areaworst,data = Train, type='C-classification', kernel='radial' )
summary(svm_model)

object <- tune(svm,diagnosis~.,data = breastcancer,ranges = list(gamma =2^(-1:1), cost=2^(2:4), tune.control=tune.control(sampling = "fix")))
summary(object)

svm_model<- svm(diagnosis~+concave.pointsmean+radiusworst+perimeterworst+areaworst,data = Train, 
                       type='C-classification', kernel='radial',gamma = 0.5,Cost = 4 )
summary(svm_model)
#str(Train)
#str(Test)
#summary(Test)
predicted<-predict(svm_model,Test)
#str(predicted)
#predicted<- as.character(predicted)
#predicted<- as.numeric(predicted)
#str(predicted)
#print(predicted)
predicted
CrossTable(predicted,Test$diagnosis)
confusionMatrix(predicted,Test$diagnosis)
plot(predicted,Test$diagnosis, xlab="Actual",ylab="predicted")

#actual<-breastcancer[470:569,]
#actual$diagnosis<- as.character(actual$diagnosis)
#actual$diagnosis<- as.numeric(actual$diagnosis)
#str(actual$diagnosis)
#actual1<- actual$diagnosis


#######################################################################################################
#GBM
#######################################################################################################
library(gbm)
library(caret)
mod_gbm = gbm(diagnosis~+concave.pointsmean+radiusworst+perimeterworst+areaworst,
              data = Train,
              distribution = "multinomial",
              cv.folds = 10,
              shrinkage = .01,
              n.minobsinnode = 15,
              n.trees = 500)
print(mod_gbm)

pred = predict.gbm(object = mod_gbm,
                   newdata = Test,
                   n.trees = 500,
                   type = "response")
print(pred)
labels = colnames(pred)[apply(pred, 1, which.max)]
result = data.frame(Test$diagnosis, labels)
print(result)
plot(result, xlab="Actual",ylab="predicted")
cm = confusionMatrix(Test$diagnosis, as.factor(labels))
print(cm)
tc = trainControl(method = "repeatedcv", number = 10)
model = train(diagnosis ~., data=Train, method="gbm", trControl=tc)
print(model)
pred = predict(model, Test)
result = data.frame(Test$diagnosis, pred)
print(result)

#############################################################################################################
#############################################################################################################



