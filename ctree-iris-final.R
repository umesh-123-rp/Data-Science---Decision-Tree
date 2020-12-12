# Build a decision tree for the 'iris' data with function 'ctree()' in package "party".

# Loading 'iris' dataset
iris<-read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\Decision Tree-R\\iris.csv")
View(iris)
str(iris)
summary(iris)
# Declaring Species as a categorical variable(factor)
iris$Species<-as.factor(iris$Species)
str(iris)
# Loading required packages for EDA
install.packages("psych")
library(psych)
hist(iris$Sepal.Length)
hist(iris$Sepal.Width)
hist(iris$Petal.Length)
hist(iris$Petal.Width)
describe(iris)
boxplot(iris)
# Sepal Length and Sepal width follow normal distribution, Mean and Median are at centres
# Petal length and Petal width do not follow normal distribution; Means do not match with medians
pairs(iris)
# Based on the scatter diagram, relations are observed among sepal length,petal length and peta width
# Checking with Correlation coefficient values
cor(iris$Petal.Length,iris$Petal.Width)
cor(iris$Sepal.Length,iris$Petal.Length)
cor(iris$Sepal.Length,iris$Petal.Width)
cor(iris$Sepal.Width,iris$Petal.Length)
cor(iris$Sepal.Width,iris$Petal.Width)
# There is a strong positive relation among petal length,petal width and sepal length.
# Petal width does not have any strong relationship with any other variables

# Install required package "party"
install.packages("party")
library(party)

# Splitting dataset into training and testing
set.seed(7)
ind <- sample(2,nrow(iris), replace=TRUE, prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]
#Model Building
iris_ctree <- ctree(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,data=trainData)
#Generate the model summary
summary(iris_ctree)
#Predict for test data set
train_predict <- predict(iris_ctree,trainData,type="response")
a<-table(train_predict,trainData$Species)
sum(diag(a))/sum(a)
mean(train_predict != trainData$Species) * 100
# Training accuracy is 0.97
test_predict <- predict(iris_ctree, newdata= testData,type="response")
mean(test_predict != testData$Species) * 100
confusionMatrix(test_predict,testData$Species)
# Testing accuracy is 0.9
print(iris_ctree)
plot(iris_ctree)
plot(iris_ctree, type="simple")

# Application of RandomForest

# Loading the package
library(randomForest)
#Generate Random Forest learning treee
iris_rf <- randomForest(Species~.,data=trainData,ntree=100,proximity=TRUE)
a<-table(predict(iris_rf),trainData$Species)
sum(diag(a)/sum(a))
# Training accuracy is 0.97

# Check the Random Forest model and importance features
print(iris_rf)
importance(iris_rf)
varImpPlot(iris_rf)
# The important features are observed to be Petal length and petal width

# build random forest for testing data
irisPred<-predict(iris_rf,newdata=testData)
table(irisPred, testData$Species)
confusionMatrix(irisPred,testData$Species)
# Accuracy of the model is 0.9
# To check the margin, positive or negative, if positive it means correct classification
plot(margin(iris_rf,testData$Species))
tune.rf <- tuneRF(iris[,-5],iris[,5], stepFactor=0.5)
print(tune.rf)
# With Mtry=1, we get minimum OOB error as 0.04

# Conclusion :
# Tree Diagram indicates that Petal length and petal width are the important variables 
# for distinguishing which type of iris class each flower belongs to. 
# The factors sepal length and sepal width are not necessary to predict which class the flowers belong to.
# Both the algorithms ctree and randomForest give same accuracy i.e. 0.9
# Both the models also indicate that Petal length and petal width are important variables