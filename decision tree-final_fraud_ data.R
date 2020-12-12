###Problem Statement:
# Use Random Forest to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

#Data Description :
  
#Undergrad : person is under graduated or not
#Marital.Status : marital status of a person
#Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
#Work Experience : Work experience of an individual person
#Urban : Whether that person belongs to urban area or not
# Loading the required packages
install.packages("caret")
install.packages("gmodels")
install.packages("psych")
install.packages("randomForest")
library(randomForest)
library(caret)
library(gmodels)
library(psych)
# Loading the dataset 
Fraud_Data<-read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Decision Tree-R\\Fraud_check.csv")
View(Fraud_Data)
str(Fraud_Data)
# Understanding the distribution of taxable income
hist(Fraud_Data$Taxable.Income)
# Histogram with lower intervals for better understanding
hist(Fraud_Data$Taxable.Income, main = "Taxable.Income",xlim = c(0,100000),
     breaks=c(seq(40,60,80)), col = c("blue","red", "green","violet"))
plot(Fraud_Data$Taxable.Income)
summary(Fraud_Data)
describe(Fraud_Data)
# The taxable income is uniform throughout

# Defining the categorical data in the given data set
Fraud_Data$Undergrad<-as.factor(Fraud_Data$Undergrad)
Fraud_Data$Marital.Status<-as.factor(Fraud_Data$Marital.Status)
Fraud_Data$Urban<-as.factor(Fraud_Data$Urban)
pairs(Fraud_Data)
# From the scatter diagram, it indicates that there is no any relationship among the variables 

# Categorise Taxable Income in the form Risky and Good
Risky_Good = ifelse(Fraud_Data$Taxable.Income<=30000, "Risky", "Good")
# Adding Risky_Good as a column in the company dataset
FD = data.frame(Fraud_Data, Risky_Good)
# Declaring the Risky_Good as a categorical variable
FD$Risky_Good<-as.factor(FD$Risky_Good)
View(FD)
# Remove the column Taxable Income and replace it by the column "Risky_Good"
FD1<-FD[,-3]
View(FD1)
str(FD1)
# Splitting data into training and testing
inTraininglocal<-createDataPartition(FD1$Risky_Good,p=.70,list = F)
training<-FD1[inTraininglocal,]
testing<-FD1[-inTraininglocal,]
#Model Building using Random Forest algorithm
attach(FD1)
set.seed(213)
model_rf<-randomForest(Risky_Good~., data=training, ntree=1000)
print(model_rf)
plot(model_rf)
# OOB error was found to be 22%
# To understand the importance of variables,reduction in Gini impurity was calculated
print(importance(model_rf))
# Maximum decrease was found in City Population and Work Experience
pred_rf<-predict(model_rf,testing[-6])
a<-table(testing$Risky_Good,pred_rf)
sum(diag(a))/sum(a)
# Testing accuracy was found to be 0.78

# CONCLUSION :
# THe fraud check data was analysed initially as a Multi Linear regression model taking taxable income data as continuous data
# Then further taxable income data was converted into class variable and classification technique was applied
# Decision tree and all linked ensembled techniques like bagging, boosting, randomforest,
# Stacking with glm, rpart, treebag, boosting randomforest etc were applied.
# Finally 3 models were found to have same accuracy They are glm, rf and boosting
# final model and predictions were prepared with "glm" model as one of the best model
