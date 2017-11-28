# remove all variables
rm(list=ls(all=TRUE))

# import libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(Amelia)
library(caTools)
library(forcats)
library(stringr)
library(randomForest)

setwd("/Users/andrewwilson/Documents/Education/Kaggle/Titanic")

# import and combine data sets
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")
comb_raw <- rbind(mutate(train_raw, Set = "train"),
              mutate(test_raw, Set = "test",
                     Survived = 0))


# change features to factors or characters where necessary
comb <- comb_raw %>%
  mutate(Survived = factor(Survived),
         Pclass = factor(Pclass),
         Name = as.character(Name),
         Ticket = as.character(Ticket),
         Cabin = as.character(Cabin),
         Embarked = factor(Embarked))

# remove empty Embarked factor
ggplot(comb, aes(Embarked)) + geom_bar()
comb$Embarked <- fct_collapse(comb$Embarked, S = c("", "S"))

# remove NA fare
comb$Fare[which(is.na(comb$Fare))] <- median(comb$Fare, na.rm=TRUE)

## add passenger title feature
# extracts passenger titles
comb$Title <- str_sub(comb$Name, str_locate(comb$Name, ",")[ , 1] + 2, str_locate(comb$Name, "\\.")[ , 1] - 1)
# combines some passenger titles
male_noble_names <- c("Capt", "Col", "Don", "Dr", "Jonkheer", "Major", "Rev", "Sir")
comb$Title[comb$Title %in% male_noble_names] <- "male_noble"
female_noble_names <- c("Lady", "Mlle", "Mme", "Ms", "the Countess", "Dona")
comb$Title[comb$Title %in% female_noble_names] <- "female_noble"
comb$Title <- factor(comb$Title)
fct_count(comb$Title)

# remove unnecessary features
ggplot(comb, aes(Fare)) + geom_density()
comb_simple <- comb %>%
  select(-Name, -Ticket, -Cabin)

# check which data is missing
missmap(comb_simple)

# impute age with a simple linear model
comb_age <- comb_simple %>% filter(!is.na(Age))
na_age <- comb_simple %>% filter(is.na(Age))
age_lm <- lm(Age ~ . -PassengerId -Set, data=comb_age)
summary(age_lm)
na_age$Age <- round(predict(age_lm, na_age))
comb_final <- rbind(comb_age, na_age)
comb_final <- mutate(comb_final, Embarked = ifelse(Embarked == "", "S", Embarked))
any(is.na(comb_final))

# separate back into training and test sets
train_final <- comb_final %>% filter(Set == "train") %>% select(-Set)
test_final <- comb_final %>% filter(Set == "test") %>% select(-Set)

# train a logistic regression model
# model <- glm(Survived ~ . -PassengerId, family=binomial(link='logit'), data=train_final)
# summary(model)

# train a random forest model
model <- randomForest(Survived ~ . -PassengerId, importance=TRUE, ntree=1000, data=train_final)
model$confusion
model$importance
  
# create submission data frame
submission_predictions <- predict(model, test_final)
submission <- data.frame(PassengerId = test_final$PassengerId, Survived = submission_predictions)
View(submission)
any(is.na(submission))
write.csv(submission, "titanic-submission.csv", row.names=FALSE)



