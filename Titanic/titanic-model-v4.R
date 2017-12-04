# remove all variables
rm(list=ls(all=TRUE))

# load libraries
# install.packages(c("e1071", "caret", "doSNOW", "ipred", "xgboost"))
library(dplyr)
library(forcats)
library(stringr)
library(caret)
library(doSNOW)

#=================================================================
# Load Data
#=================================================================

setwd("/Users/andrewwilson/Documents/Education/Kaggle/Titanic")
train_raw <- read.csv("train.csv", stringsAsFactors = FALSE)
test_raw <- read.csv("test.csv", stringsAsFactors = FALSE)
# View(train)

train_raw$Set <- "train"
test_raw$Set <- "test"
test_raw$Survived <- NA

data <- rbind(train_raw, test_raw)
#str(data)


#=================================================================
# Data Wrangling
#=================================================================

# Replace missing embarked values with mode.
# table(data$Embarked)
data$Embarked[data$Embarked == ""] <- "S"


# Add in median fare if missing
data$Fare[which(is.na(data$Fare))] <- median(data$Fare, na.rm=TRUE)


# Add a feature for tracking missing ages.
summary(data$Age)
data$MissingAge <- ifelse(is.na(data$Age),
                           "Y", "N")


# Add a feature for family size.
data$FamilySize <- 1 + data$SibSp + data$Parch


# Set up factors.
data$Survived <- as.factor(data$Survived)
data$Pclass <- as.factor(data$Pclass)
data$Sex <- as.factor(data$Sex)
data$Embarked <- as.factor(data$Embarked)
data$MissingAge <- as.factor(data$MissingAge)
data$Set <- as.factor(data$Set)



#=================================================================
# Feature engineering
#=================================================================

# create a new factor about whether the passenger came alone
data <- data %>% mutate(IsAlone = factor(ifelse(SibSp == 0, 1, 0)))

# create a factor based on title within name
data$Title <- str_sub(data$Name, str_locate(data$Name, ",")[ , 1] + 2, str_locate(data$Name, "\\.")[ , 1] - 1)
# dataines some passenger titles
male_noble_names <- c("Capt", "Col", "Don", "Dr", "Jonkheer", "Major", "Rev", "Sir")
data$Title[data$Title %in% male_noble_names] <- "male_noble"
female_noble_names <- c("Lady", "Mlle", "Mme", "Ms", "the Countess", "Dona")
data$Title[data$Title %in% female_noble_names] <- "female_noble"
data$Title <- factor(data$Title)
fct_count(data$Title)

# Subset data to features we wish to keep/use.
features <- c("Survived", 
              "Pclass", 
              "Sex", 
              "Age", 
              "Parch", 
              "Embarked", 
              "FamilySize", 
              "IsAlone", 
              "Title", 
              "Set")
data <- data[, features]
#str(train)


#=================================================================
# Impute Missing Ages
#=================================================================

# Leverage bagged decision trees to impute missing values for the Age feature.

# First, transform all feature to dummy variables.
dummy.vars <- dummyVars(~ ., data = select(data, -Survived, -Set))
train.dummy <- predict(dummy.vars, select(data, -Survived, -Set))
#View(train.dummy)

# Now, impute!
pre.process <- preProcess(train.dummy, method = "bagImpute")
imputed.data <- predict(pre.process, train.dummy)
#View(imputed.data)

data$Age <- imputed.data[, 6]
#View(train)

#=================================================================
# Split Data
#=================================================================

# Split out final train and test from original set
train <- filter(data, Set == "train")
test <- filter(data, Set == "test")
train <- select(train, -Set)
test <- select(test, -Set)

# Use caret to create a 70/30% split of the training data,
# keeping the proportions of the Survived class label the
# same across splits.
set.seed(54321)
indexes <- createDataPartition(train$Survived,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
titanic.train <- train[indexes,]
titanic.test <- train[-indexes,]


# Examine the proportions of the Survived class lable across
# the datasets.
prop.table(table(train$Survived))
prop.table(table(titanic.train$Survived))
prop.table(table(titanic.test$Survived))



#=================================================================
# Train Model
#=================================================================

# Set up caret to perform 10-fold cross validation repeated 3 
# times and to use a grid search for optimal model hyperparamter
# values.
train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")


# Leverage a grid search of hyperparameters for xgboost
tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)
# View(tune.grid)


# Use the doSNOW package to enable caret to train in parallel.
# Create a socket cluster using 10 processes. 
cl <- makeCluster(10, type = "SOCK")

# Register cluster so that caret will know to train in parallel.
registerDoSNOW(cl)

# Train the xgboost model using 10-fold CV repeated 3 times 
# and a hyperparameter grid search to train the optimal model.
caret.cv <- train(Survived ~ ., 
                  data = titanic.train,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)
stopCluster(cl)


# Examine caret's processing results
caret.cv


# Make predictions on the test set using a xgboost model 
# trained on all 625 rows of the training set using the 
# found optimal hyperparameter values.
preds <- predict(caret.cv, titanic.test)


# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
confusionMatrix(preds, titanic.test$Survived)



# Create predictions from actual test set
submission_predictions <- predict(caret.cv, test)
submission <- data.frame(PassengerId = test_raw$PassengerId, Survived = submission_predictions)
View(submission)
write.csv(submission, "titanic-submission-v4.csv", row.names=FALSE)