

      #### Introduction ####
      
      
## Stroke is a neurological disease that occurs when the oxygen supply is diminished to the vulnerable brain tissue. 
## It is often disabilitating because brain tissue barely regenerates and affects 3% of the population over a lifetime.
## Prevention therefor is of paramount importance. However, preventive measures like medication may have side effects.
## The risk of such side effects caused by primary prevention cannot outweigh the benefit. Therefor, risk-adjusted interventions should be applied.
## In this study, a machine learning algorithm to predict stroke in an adult population was trained and validated.
## Predicted class and class probabilities will be compared between models to find the best fit.





      #### Methods ####

      
# Install and load packages
# Rtools REQUIRED! Instructions can be accessed here: https://cran.r-project.org/bin/windows/Rtools/
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(tableone)) install.packages("tablone", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(MLeval)) install.packages("MLeval", repos = "http://cran.us.r-project.org")
if(!require(zoo)) install.packages("zoo", repos = "http://cran.us.r-project.org")
if(!require(xts)) install.packages("xts", repos = "http://cran.us.r-project.org")
if(!require(quantmod)) install.packages("quantmod", repos = "http://cran.us.r-project.org")
if(!require(DMwR)) install.packages("https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

# Import data

data <- read.csv("https://raw.githubusercontent.com/ckuemmerli/cyo_capstone/main/healthcare-dataset-stroke-data.csv")
str(data)
head(data)

   
   # The codebook looks like this:

# 1) id: unique identifier
# 2) gender: "Male", "Female" or "Other"
# 3) age: age of the patient
# 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# 6) ever_married: "No" or "Yes"
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 8) Residence_type: "Rural" or "Urban"
# 9) avg_glucose_level: average glucose level in blood
#10) bmi: body mass index
#11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
#12) stroke: 1 if the patient had a stroke or 0 if not



# Data cleaning


## Exclude non-adults

data <- filter(data, age >= 18)


## adapt class and levels of variables

data$bmi <- as.numeric(data$bmi)

factor_variables <- c("gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status", "stroke")
data[factor_variables] <- lapply(data[factor_variables], factor)

sapply(data[factor_variables], levels)

levels(data$hypertension) <- list("hypertension"= "1", "no_hypertension" = "0")
levels(data$heart_disease) <- list("heart_disease" = "1", "no_heart_disease" = "0")
levels(data$ever_married) <- list("married" = "Yes", "not_married" = "No")
levels(data$stroke) <- list("stroke" = "1", "no_stroke" = "0")


## Assess missing values

missing.values <- data %>% gather(key = "key", value = "val") %>%
  mutate(is.missing = is.na(val)) %>%
  group_by(key, is.missing) %>%
  summarise(num.missing = n()) %>%
  filter(is.missing==T) %>%
  select(-is.missing) %>%
  arrange(desc(num.missing))

mutate(missing.values, percent_missing = num.missing/nrow(data)*100)

### only 201 bmis are missing


### as per WHO definition, there are categories for BMI

data <- data %>% mutate(bmi_class =ifelse(bmi < 25, "normal",
                                          ifelse(bmi < 30, "overweight", "obese")))
data$bmi_class <- as.factor(data$bmi_class)
table(data$stroke, data$bmi_class)
factor_variables <- c(factor_variables, "bmi_class")

## data structure

str(data)


# Split into test and validation set based on the outcome

set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(data$stroke, times = 1, p = 0.2, list = FALSE)
trainset <- data[-test_index,]
testset <- data[test_index,]


# Data exploration

summary(trainset[, -1])

barplots <- function(x){
  ggplot(trainset, aes(x)) +
    geom_bar() +
    xlab("") +
    ylab("Count") +
    theme_minimal()
}

lapply(trainset[, factor_variables], barplots)

## The dependent variable stroke is imbalanced. Apart from the imbalanced dependent variable,
## also heart disease has a low prevalence.
## Prevalence of stroke in the incomplete cases (with missing bmi) is checked.

select(data, bmi, stroke) %>% filter(is.na(bmi) & stroke %in% "stroke") %>% count()


## Presumed confounding by age

trainset %>% group_by(work_type) %>% summarise(mean = mean(age)) # older participants are more often self-employed
trainset %>% group_by(hypertension) %>% summarise(mean = mean(age)) # older participants have more often hypertension
trainset %>% group_by(Residence_type) %>% summarise(mean = mean(age))
trainset %>% group_by(smoking_status) %>% summarise(mean = mean(age)) # older participants more ofte quit smoking


## Discriminatory ability of numeric predictor variable

featurePlot(trainset[, c(3,9,10)], 
            y = trainset$stroke, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")),
            main = "Discriminatory variables")


# group comparison stroke vs no stroke for all features.

CreateTableOne(vars = c(factor_variables[-8],"age","bmi","avg_glucose_level"),
               data = trainset,
               strat = "stroke",
               addOverall = T)


# Correlation among numeric variables (Collienearity)

corr <- select(trainset, age, avg_glucose_level, bmi) %>%
  na.omit() %>%
  cor()
corrplot(corr, method = "number", cl.pos = "n")

## CAVE: prevalence of stroke is low, the distribution of the outcome is unbalanced.


# Preprocessing


## Remove categories with few subjects

trainset <- filter(trainset, gender %in% c('Female', 'Male'))
trainset <- filter(trainset, work_type %in% c('Govt_job', 'Private', 'Self-employed'))


## Encoding manually (dummy and other categorical)

levels(trainset$hypertension) <- list("1"= "hypertension", "0" = "no_hypertension")
levels(trainset$heart_disease) <- list("1" = "heart_disease", "0" = "no_heart_disease")
levels(trainset$gender) <- list("1" = "Female", "0" = "Male")
levels(trainset$ever_married) <- list("1" = "married", "0" = "not_married")
levels(trainset$work_type) <- list("1" = "Govt_job", "1" = "Private", "0" = "Self-employed")
levels(trainset$Residence_type) <- list("1" = "Rural", "0" = "Urban")
levels(trainset$smoking_status) <- list("3" = "formerly smoked", "2" = "never smoked", "1" = "smokes", "0" = "Unknown")
levels(trainset$ever_married) <- list("1" = "married", "0" = "not_married")


## Remove categories with few subjects

testset <- filter(testset, gender %in% c('Female', 'Male'))
testset <- filter(testset, work_type %in% c('Govt_job', 'Private', 'Self-employed'))


## Encoding manually (dummy and other categorical)

levels(testset$hypertension) <- list("1"= "hypertension", "0" = "no_hypertension")
levels(testset$heart_disease) <- list("1" = "heart_disease", "0" = "no_heart_disease")
levels(testset$gender) <- list("1" = "Female", "0" = "Male")
levels(testset$ever_married) <- list("1" = "married", "0" = "not_married")
levels(testset$work_type) <- list("1" = "Govt_job", "1" = "Private", "0" = "Self-employed")
levels(testset$Residence_type) <- list("1" = "Rural", "0" = "Urban")
levels(testset$smoking_status) <- list("3" = "formerly smoked", "2" = "never smoked", "1" = "smokes", "0" = "Unknown")
levels(testset$ever_married) <- list("1" = "married", "0" = "not_married")


## select variables based on exploration results and change class to numeric

trainset <- select(trainset,
                   gender,
                   age,
                   hypertension,
                   work_type,
                   avg_glucose_level,
                   bmi,
                   smoking_status,
                   stroke)

trainset[, 1:7] <- sapply(trainset[, 1:7], as.numeric)

testset <- select(testset,
                  gender,
                  age,
                  hypertension,
                  work_type,
                  avg_glucose_level,
                  bmi,
                  smoking_status,
                  stroke)

testset[, 1:7] <- sapply(testset[, 1:7], as.numeric)


# Imputation

trainset$bmi[is.na(trainset$bmi)] <- mean(trainset$bmi, na.rm = T)
testset$bmi[is.na(testset$bmi)] <- mean(testset$bmi, na.rm = T)


# Remove objects

rm(data, test_index, corr)


      #### Results ####


# Building models

## First, use 10-fold cross validation and add metrics to later extract sensitivity,
## specificity, receiver operating characteristics (ROC) curve area under the curve (AUC) and precision.

ctrl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = T,
  summaryFunction = twoClassSummary)


# Model 1: Generalized linear model, logistic regression

set.seed(123, sample.kind="Rounding")
glmFit <- train(stroke ~ ., data = trainset,
                method = "glm",
                family = "binomial",
                trControl = ctrl,
                preProcess = (c("center", "scale")), # preprocess due to the different range of age, glucose and bmi.
                metric = "ROC") # metric include Sensitivity, Specificity and area under the curve (AUC) of the receiver operating characteristics curve.
evalm(glmFit, positive = "stroke", plots = c("r", "pr"))$pr # plot ROC curve and Precision-Recall curve. Preferably both to take into account the low prevalence, which is only considered in precision.
# The accuracy of the model is high but because the predictive accuracy is solely based
## on the majority class (no stroke) and we miss all or almost all patients with a stroke.


# To address the issue of imbalance with "no stroke" being much more prevalent and "stroke" a rare event, synthetic minority oversampling technique is introduced.
# Other approached would be up or downsampling among others or using Subsampling inside the train function.

set.seed(123, sample.kind="Rounding")
smote_train <- SMOTE(stroke ~ ., data = trainset)                    
table(smote_train$stroke) # now, preavalence of the positive outcome "stroke" is higher.


# Using now logistic regression to train the algorithm on the smote dataset

set.seed(123, sample.kind="Rounding")
glmFit2 <- train(stroke ~ ., data = smote_train, 
                 method = "glm",
                 family = "binomial",
                 trControl = ctrl,
                 preProcess = (c("center", "scale")),
                 metric = "ROC")
evalm(glmFit2, positive = "stroke", plots = c("r", "pr"))$pr

## The model performs much better and detects many patients with a stroke. From a
## health care professionals point of view, we would rather have a sensitive test
## based on the predictors we have.
## First, because all predictors are readily available. Secondly, stroke is a serious
## condition and we would rather increase the sensitivity at the expense of specificity.


# Model 2: Random forest

tnGrid_rf <- expand.grid(mtry = seq(1,20)) # for this model, tuning parameters are available
set.seed(123, sample.kind="Rounding")
rfFit <- train(stroke ~ .,
               data = smote_train,
               method = "rf",
               tuneGrid = tnGrid_rf,
               trControl = ctrl,
               preProcess = (c("center", "scale")),
               metric = "ROC")
rfFit$bestTune


## Model 3: Knn

tnGrid_knn <- data.frame(k = seq(5, 75, 5))
set.seed(123, sample.kind="Rounding")
knnFit <- train(stroke ~ .,
                data = smote_train,
                method = "knn",
                trControl = ctrl,
                tuneGrid = tnGrid_knn,
                metric = "ROC")
knnFit$bestTune
plot(knnFit)


## Model 4: Support vector machine (non-linear) without tuning.

set.seed(123, sample.kind="Rounding")
svmFit <- train(stroke ~.,
                    data = smote_train,
                    method = "svmPoly",
                    trControl = ctrl,
                    preProcess = c("center", "scale"),
                    metric = "ROC")
plot(svmFit)

# Model performance

## Model 1: Logistic regression\

evalm(glmFit, positive = "stroke", plots = c("r", "pr"))$pr

## Model 1 with SMOTE

evalm(glmFit2, positive = "stroke", plots = c("r", "pr"))$pr

## Model 2: Random forest with SMOTE

evalm(rfFit, positive = "stroke", plots = c("r", "pr"))$pr

## Model 3: K-nearest neighbour with SMOTE

evalm(knnFit, positive = "stroke", plots = c("r", "pr"))$pr

## Model 4: Support vector machine with SMOTE

evalm(svmFit, positive = "stroke", plots = c("r", "pr"))$pr


## Summary of training data for educational purposes only

resamps <- resamples(list(GLM = glmFit,
                          GLM_smote = glmFit2,
                          RF = rfFit,
                          knn = knnFit,
                          svm = svmFit))
summary(resamps)


      #### Results ####


# On the test set

## Model 1: Logistic regression

glm_test <- confusionMatrix(predict(glmFit, testset), testset$stroke)
glm_test

## Model 1 with SMOTE

glm2_test <- confusionMatrix(predict(glmFit2, testset), testset$stroke)
glm2_test

## Model 2: Random forest with SMOTE

rf_test <- confusionMatrix(data = predict(rfFit, testset), reference = testset$stroke)
rf_test

## Model 3: K-nearest neighbour with SMOTE

knn_test <- confusionMatrix(data = predict(knnFit, testset), reference = testset$stroke)
knn_test

## Model 4: Support vector machine with SMOTE

svm_test <- confusionMatrix(predict(svmFit, testset), testset$stroke)
svm_test

## and summarised below
tibble(Model = c("Logistic regression", "Logistic regression with SMOTE", "Random forest with SMOTE", "K-nearest neighbour with SMOTE", "Support vector machine with SMOTE"),
  "Accuracy" = c(glm_test$overall[1], glm2_test$overall[1], rf_test$overall[1], knn_test$overall[1], svm_test$overall[1]),
       "Sensitivity" = c(glm_test$byClass[1], glm2_test$byClass[1], rf_test$byClass[1], knn_test$byClass[1], svm_test$byClass[1]),
       "Specificity" = c(glm_test$byClass[2], glm2_test$byClass[2], rf_test$byClass[2], knn_test$byClass[2], svm_test$byClass[2])) %>%
  kable()
      

## Compared to the train dataset, the random forest and k-nearest neighbor show overfitting
## which results in an inferior performance on the test dataset compared to train set.
## The logistic regression model with SMOTE shows the best performance on the test set
## with a sensitivity of 70 % and a specifity of 72.3 %.


      #### Conclusion ####


## The machine learning algorithm to predict stroke reached a sensitivity of 72 % using a
## logistic regression model. The AUC-ROC and AUC PR are 82 % and 71 %, respectively.
## The specificity of 76.2 % indicates that there is a high false positive rate.
## However, considering the often disabling condition after a stroke, we would rather
## treat some individuals (for primary prevention) who never experience a stroke than
## not treat individuals who will develop this severe disease. Especially because the
## side effects of the primary prevention with a drug treatment are minor.
## From a methodological point of view, we have encountered the frequent problem of
## imbalanced outcome. SMOTE was used to subsample the underrepresented minority class
## and the models trained on the SMOTE-data showed a much better performance.
## In addition, some models are overfitted. This problem could have been addressed with Bagging
## decision trees (for random forest, it takes bootstraps from the training data) or
## more sophisticated parameter tuning.\
## Limitations are the imputation methods used on the entire dataset, overtraining
## of the random forest and k-nearest neighbour model and the relatively few positive
## events (= stroke) that required smote that could have been carried out within
## the train function.
## Future research should be carried out with bigger datasets and more positive outcomes
## and should also focus on the increase of sensitivity and specificity of the prediction
## model to correctly identify the people at risk of a stroke.


      #### Acknowledgement ####

## I thank the provider of the stroke prediction dataset, Fedesoriano.

