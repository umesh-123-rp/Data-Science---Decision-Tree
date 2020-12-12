###Problem Statement:
##A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
#Approach - A decision tree can be built with target variable Sale 
#(we will first convert it in categorical variable) & all other variable 
#will be independent in the analysis.  

#About the data: 
#Let's consider a Company dataset with around 10 variables and 400 records. 
#The attributes are as follows: 
#??? Sales -- Unit sales (in thousands) at each location
#??? Competitor Price -- Price charged by competitor at each location
#??? Income -- Community income level (in thousands of dollars)
#??? Advertising -- Local advertising budget for company at each location (in thousands of dollars)
#??? Population -- Population size in region (in thousands)
#??? Price -- Price company charges for car seats at each site
#??? Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
#??? Age -- Average age of the local population
#??? Education -- Education level at each location
#??? Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
#??? US -- A factor with levels No and Yes to indicate whether the store is in the US or not

# Loading the required packages
install.packages("caret")
install.packages("C50")
install.packages("gmodels")
install.packages("psych")
library(caret)
library(C50)
library(gmodels)
library(psych)
# Loading the dataset 
Company_Data<-read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Decision Tree-R\\Company_Data.csv")
View(Company_Data)
str(Company_Data)
# Understanding the distribution of Sales
hist(Company_Data$Sales)
plot(Company_Data$Sales)
summary(Company_Data)
describe(Company_Data)
# The sales distribution seems to be normally distributed
# Defining the categorical data in the given data set
Company_Data$ShelveLoc<-as.factor(Company_Data$ShelveLoc)
Company_Data$Urban<-as.factor(Company_Data$Urban)
Company_Data$US<-as.factor(Company_Data$US)
pairs(Company_Data)
# From the scatter diagram, it indicates that there is a decreasing relationship between price and Sales
# And also there is an increasing relatioship between competitor price and Price
# To check the strength of relation, let's calculate correlation coefficient
cor(Company_Data$Sales,Company_Data$Price)
cor(Company_Data$CompPrice,Company_Data$Price)
# Correlation coefficient is -0.44 for price and sales
# Correlation coefficient is 0.58 between Competitor price and price
# The relataion is not very strong
# We can build a linear regression with the dataset
attach(Company_Data)
model1<-lm(Sales~., data=Company_Data)
summary(model1)
# R^2 value was found to be 0.87
plot(model1)
# It is observed that the sales data follows normal distribution.
# Further analysis was done to optimise the model
install.packages("car")
library(car)
car::vif(model1)
# VIF values are found to be less than 10. There is no Collinearity observed.
library(MASS)
stepAIC(model1)
# AIC value decreases by removing the insignificant variable i.e. Door and US
residualPlots(model1)
avPlots(model1)
qqPlot(model1)
sqrt(sum(model1$residuals^2)/nrow(Company_Data))
# Mean Square is found to be 1.003
model2<-lm(Sales~ CompPrice+Income+Advertising+Price+ShelveLoc+Age, data= Company_Data)
summary(model2)
sqrt(sum(model2$residuals^2)/nrow(Company_Data))
# Both the models give R^2 value is 0.87 and mean square error as 1.003

## Solution through Decision Tree
# Defining Sales in the form of categorical variable
High = ifelse(Company_Data$Sales>10, "Yes", "No")
# Adding High as a column in the company dataset
CD = data.frame(Company_Data, High)
View(CD)
# Remove the column Sales which got replaced by the column "High"
CD1<-CD[,-1]
# Declaring the column "High" as a factor
CD1$High<-as.factor(CD1$High)
View(CD1)
# Splitting data into training and testing
inTraininglocal<-createDataPartition(CD1$High,p=.70,list = F)
training<-CD1[inTraininglocal,]
testing<-CD1[-inTraininglocal,]
#Model Building
model<-C5.0(High~ .,data = training) 
#Generate the model summary
summary(model)
plot(model)
# Training error is found to be 4.3% 
# Price and Shelveloc have 100% usage
#Predict for test data set
pred<-predict.C5.0(model,testing[,-11])
a<-table(testing$High,pred)
sum(diag(a))/sum(a)
plot(model)
# Accuracy of the model is more than 0.82

# To further improve the model, bagging algorithm is applied 
# Bagging Application
library(ipred)
set.seed(300)
bag_model<-bagging(High~.,data=training, nbagg=25)
summary(bag_model)
pred_bagg<-predict(bag_model,testing[,-11])
a<-table(testing$High,pred_bagg)
sum(diag(a))/sum(a)
# Accuracy of the model has improved to 0.86
# To check further improvement in accuracy, boosting algorithm was applied
# Boosting Application
library(xgboost)
set.seed(123)
boost_model<- C5.0(High~.,data=training, trials=10)
summary(boost_model)
plot(boost_model)
pred_boost<-predict.C5.0(boost_model,testing[,-11])
a<-table(testing$High,pred_boost)
sum(diag(a))/sum(a)
plot(boost_model)
# Testing Accuracy improved to 0.87
# To check further improvement, random forest algorithm was applied
# Random Forest
install.packages("randomForest")
library(randomForest)
model_rf<-randomForest(High~., data=training, ntree=1000)
print(model_rf)
# OOB error was found to be 13%
# To understand the importance of variables,reduction in Gini impurity was calculated
print(importance(model_rf))
# Maximum decrease was found in Price and then ShelveLoc
pred_rf<-predict(model_rf,testing[,-11])
a<-table(testing$High,pred_rf)
sum(diag(a))/sum(a)
plot(model_rf)
# Testing accuracy was found to be 0.85.

# To find out further improvement model was prepared with Stacking algorithm
# Stacking
set.seed(123)
install.packages("mboost")
install.packages("caretEnsemble")
library(caretEnsemble)
library(mboost)
# In stacking algorithm, 5 algorithms were used at a time to check the best accuracy
# 5 algorithms are "rpart","glm","rf","treebag","glmbost"
algorithm_to_use <- c("rpart","glm","rf","treebag","glmboost")
model_stack<-caretList(High~.,data=training, methodList=algorithm_to_use, 
    trControl=trainControl("cv",number=10,savePredictions = T,classProbs = T))
summary(model_stack)
stacking_results<-resamples(model_stack)
summary(stacking_results)
# The best mean accuracy was observed in "glm" model.
# Accuracy was obsrrved to 0.92 in "glm"
# Therefore, the final model was chosen as "glm" model
final_stack <- caretStack(model_stack, method="glm",trControl=trainControl("cv",
                          number=10,savePredictions=T, classProbs=T))
pred_stack<- predict(final_stack,testing[,-11])
a<-table(testing$High,pred_stack)
sum(diag(a))/sum(a)
# Testing Accuracy was observed to be 0.94

# CONCLUSION :
# THe Seat Sale data was analysed initially as a Multi Linear regression model taking Sales data as continuous data
# Then further the Sales data was converted into class variable and classification technique was applied
# Decision tree and all linked ensembled techniques like bagging, boosting, randomforest,
# Stacking with glm, rpart, treebag, boosting randomforest etc were applied.
# Finally the glm ( Generalised Linear Model) was found to be with very good accuracy 0.93
# The final model and predictions were prepared with "glm" model
