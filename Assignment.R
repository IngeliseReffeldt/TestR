library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3mbo)
library(mlr3pipelines)
library(paradox)
library(readr)
library(tidyverse)
library(mlr3viz)
library(mlr3torch)
library(torch)
library(future)
library(forcats)
library(mlr3extralearners)
library(OpenML)
library(mgcv)
library(kknn)
library(lightgbm)
library(Metrics)
library(xgboost)
library(iml)
library(DALEX)
library(DALEXtra)
library(fastshap)
library(shapviz)

#### Data processing ####
# Load data
freMPL <- read_csv("freMPL.csv")

# Pre-processing

#We remove ClaimInd from the dataset, since we want to predict a price for a given customer, and we don't know, if the customer will have a claim.
#Remove ...1 since it is just row_numbers.
MPL_data <- subset(freMPL, select = -c(ClaimInd,...1))

#We replace null-values from RecordEnd, so we don't have to deal with null values.
MPL_data$RecordEnd[is.na(MPL_data$RecordEnd)] <- as.Date("2005-01-01")

#We replace all date-values with numeric values, because many models cannot handle date-values.
MPL_data$RecordBeg <- as.numeric(MPL_data$RecordBeg)
MPL_data$RecordEnd <- as.numeric(MPL_data$RecordEnd)

# Ordered factors and factors
# VehAge is an ordered variable
MPL_data$VehAge <- factor(MPL_data$VehAge, levels = c("0", "1", "2", "3", "4", "5", "6-7", "8-9", "10+"))

# VehMaxSpeed is an ordered variable
MPL_data$VehMaxSpeed <- factor(MPL_data$VehMaxSpeed,
                               levels = c("1-130 km/h", "130-140 km/h", "140-150 km/h",
                                          "150-160 km/h", "160-170 km/h", "170-180 km/h", 
                                          "180-190 km/h", "190-200 km/h", "200-220 km/h",
                                          "220+ km/h"))

# HasKmLimit
MPL_data$HasKmLimit <- factor(MPL_data$HasKmLimit)

#Creating a new data set where few observations are lumped together as "other" to avoid problems with test-observations that are not not seen in the training data. 
MPL_data_lump <- MPL_data

factor_cols <- names(MPL_data_lump)[sapply(MPL_data_lump, function(x) is.factor(x) || is.character(x))]

for (col in factor_cols) {
  MPL_data_lump[[col]] <- as.factor(MPL_data_lump[[col]])
  MPL_data_lump[[col]] <- fct_lump_min(MPL_data_lump[[col]], min = 10)
  MPL_data_lump[[col]] <- droplevels(MPL_data_lump[[col]])
}

categorical_vars <- c("VehAge", "Gender", "MariStat", "SocioCateg", "VehUsage", "VehBody", "VehPrice", "VehEngine", "VehEnergy", "VehMaxSpeed", "VehClass", "Garage") # VehMax Speed should perhaps be converted to a numeric variable by taking eg the max number in the observations given interval

MPL_data_lump[categorical_vars] <- lapply(MPL_data_lump[categorical_vars], factor)

#Final data: 
summary(MPL_data_lump)

#### Generic customers ####
##### Generic customer 1: Average customer #####
gns_kunde <- subset(MPL_data_lump[1,], select = -ClaimAmount)

gns_kunde$Exposure <- 1  
gns_kunde$LicAge <- as.numeric(as.integer(mean(MPL_data_lump$LicAge)))
gns_kunde$RecordBeg <- as.numeric(as.Date("2004-01-01"))
gns_kunde$RecordEnd <- as.numeric(as.Date("2005-01-01"))
gns_kunde$VehAge <- factor(names(sort(table(MPL_data_lump$VehAge), decreasing = TRUE))[1],
                           levels = levels(MPL_data_lump$VehAge))
gns_kunde$Gender <- factor(names(sort(table(MPL_data_lump$Gender), decreasing = TRUE))[1],
                           levels = levels(MPL_data_lump$Gender))
gns_kunde$MariStat <- factor(names(sort(table(MPL_data_lump$MariStat), decreasing = TRUE))[1],
                             levels = levels(MPL_data_lump$MariStat))
gns_kunde$SocioCateg <- factor(names(sort(table(MPL_data_lump$SocioCateg), decreasing = TRUE))[1],
                               levels = levels(MPL_data_lump$SocioCateg))
gns_kunde$VehUsage <- factor(names(sort(table(MPL_data_lump$VehUsage), decreasing = TRUE))[1],
                             levels = levels(MPL_data_lump$VehUsage))
gns_kunde$DrivAge <- as.numeric(as.integer(mean(MPL_data_lump$DrivAge)))
gns_kunde$HasKmLimit <- factor(names(sort(table(MPL_data_lump$HasKmLimit), decreasing = TRUE))[1])
gns_kunde$BonusMalus <- as.numeric(as.integer(mean(MPL_data_lump$BonusMalus)))
gns_kunde$VehBody <- factor(names(sort(table(MPL_data_lump$VehBody), decreasing = TRUE))[1],
                            levels = levels(MPL_data_lump$VehBody))
gns_kunde$VehPrice <- factor(names(sort(table(MPL_data_lump$VehPrice), decreasing = TRUE))[1],
                             levels = levels(MPL_data_lump$VehPrice))
gns_kunde$VehEngine <- factor(names(sort(table(MPL_data_lump$VehEngine), decreasing = TRUE))[1],
                              levels = levels(MPL_data_lump$VehEngine))
gns_kunde$VehEnergy <- factor(names(sort(table(MPL_data_lump$VehEnergy), decreasing = TRUE))[1],
                              levels = levels(MPL_data_lump$VehEnergy))
gns_kunde$VehMaxSpeed <- factor(names(sort(table(MPL_data_lump$VehMaxSpeed), decreasing = TRUE))[1],
                                levels = levels(MPL_data_lump$VehMaxSpeed))
gns_kunde$VehClass <- factor(names(sort(table(MPL_data_lump$VehClass), decreasing = TRUE))[1],
                             levels = levels(MPL_data_lump$VehClass))
# gns_kunde$ClaimAmount <- mean(MPL_data_lump$ClaimAmount)
gns_kunde$RiskVar <- as.numeric(as.integer(mean(MPL_data_lump$RiskVar)))
gns_kunde$Garage <- factor(names(sort(table(MPL_data_lump$Garage), decreasing = TRUE))[1],
                           levels = levels(MPL_data_lump$Garage))

##### Generic customer 2: Extreme customer #####
eks_kunde <- subset(MPL_data_lump[1,], select = -ClaimAmount)

eks_kunde$Exposure <- 1  
eks_kunde$LicAge <- as.numeric(60)
eks_kunde$RecordBeg <- as.numeric(as.Date("2004-01-01"))
eks_kunde$RecordEnd <- as.numeric(as.Date("2005-01-01"))
eks_kunde$VehAge <- factor("0")
eks_kunde$Gender <- factor("Male")
eks_kunde$MariStat <- factor("Alone")
eks_kunde$SocioCateg <- factor("CSP50")
eks_kunde$VehUsage <- factor("Private+trip to office")
eks_kunde$DrivAge <- as.numeric(30)
eks_kunde$HasKmLimit <- factor(names(sort(table(MPL_data_lump$HasKmLimit), decreasing = TRUE))[1])
eks_kunde$BonusMalus <- as.numeric(120)
eks_kunde$VehBody <- factor("sport utility vehicle")
eks_kunde$VehPrice <- factor("V")
eks_kunde$VehEngine <- factor("direct injection overpowered")
eks_kunde$VehEnergy <- factor("diesel")
eks_kunde$VehMaxSpeed <- factor("190-200 km/h")
eks_kunde$VehClass <- factor("H")
# eks_kunde$ClaimAmount <- mean(MPL_data_lump$ClaimAmount)
eks_kunde$RiskVar <- as.numeric(20)
eks_kunde$Garage <- factor("Collective garage")


#### TASK and SPLIT ####

# Task
```{r}
task = as_task_regr(MPL_data_lump, target = "ClaimAmount")

#Split
set.seed(2026)
split <- partition(task, ratio = 0.8)

train_set <- split$train
test_set <- split$test


#### BASELINE MODEL FOR COMPARISON ####
baseline <- lrn("regr.featureless")

baseline$train(task, row_ids = split$train)

##### Baseline results #####
pred_baseline <- baseline$predict(task, row_ids = split$test)
pred_baseline$score(msr("regr.rmse"))
pred_baseline$score(msr("regr.mae"))
pred_baseline$score(msr("regr.rsq"))

#### RIDGE and LASSO ####
categorical_selector = selector_type(c("factor", "ordered"))

make_encoded_learner = function(learner) {
  as_learner(
    po("encode", method = "treatment") %>>%
      learner
  )
}

ridgereg = make_encoded_learner(lrn("regr.glmnet", alpha = 0))
lassoreg = make_encoded_learner(lrn("regr.glmnet", alpha = 1))

ridgereg$train(task, row_ids = split$train)
lassoreg$train(task, row_ids = split$train)

##### Ridge and Lasso default results #####
ridgereg$predict(task, row_ids = split$test)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))
lassoreg$predict(task, row_ids = split$test)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

elasticnet_lrn = make_encoded_learner(lrn("regr.glmnet", 
                                          alpha = to_tune(0, 1), 
                                          lambda = to_tune(p_dbl(log(1e-5), log(1e1), trafo = exp))))
instance = tune(
  tuner = tnr("random_search", batch_size=20), 
  task = task$clone()$filter(split$train),
  learner = elasticnet_lrn,
  resampling = rsmp("cv", folds = 5), 
  measures = msrs(c("regr.rmse")),
  terminator = trm("evals", n_evals = 20)
)

#####Tuning results #####
instance$result_learner_param_vals
instance$result_y 

##### result on test set #####
elasticnet_lrn$param_set$values = instance$result_learner_param_vals[[1]]
elasticnet_lrn$train(task$clone()$filter(split$train))
elasticnet_lrn$predict(task$clone()$filter(split$test))$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

#### GAM ####
# Smoothing function on all numeric values
formula_gam = ClaimAmount ~ s(Exposure) + s(RecordBeg) + s(RecordEnd) + 
  VehAge + Gender + MariStat + SocioCateg + VehUsage + s(DrivAge) + 
  HasKmLimit + s(BonusMalus) + VehBody + VehPrice + VehEngine + 
  VehEnergy + VehMaxSpeed + VehClass + s(RiskVar) + Garage

##### Simple GAM model - no tuning #####
lrn_gam = lrn("regr.gam", formula = formula_gam)
lrn_gam$train(task, row_ids = split$train)
lrn_gam$predict(task, row_ids = split$test)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

##### Autotuning #####
lrn_gam_t = lrn("regr.gam",
                formula = formula_gam,
                gamma = to_tune(1,1e3,logscale=TRUE),
                select = to_tune(c(TRUE, FALSE)),
                drop.unused.levels=FALSE) # Default is TRUE, but this may cause errors if splits are unlucky

lrn_gam_tune = auto_tuner(
  tuner = tnr("random_search"),
  learner = lrn_gam_t,
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)

##### Results on test data #####
lrn_gam_tune$train(task, row_ids = train_set)
lrn_gam_tune$tuning_result
lrn_gam_tune$predict(task, row_ids = test_set)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

#### Tree ####

#####Default tree #####
lrn_tree = lrn("regr.rpart")
lrn_tree$train(task, row_ids = split$train)

plot(lrn_tree$model, compress = TRUE, margin = 0.1)
text(lrn_tree$model, use.n = TRUE, cex = 0.7)

lrn_tree$predict(task, row_ids = split$test)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

#####Encoded tree #####
categorical_selector = selector_type(c("factor", "ordered"))

make_encoded_learner = function(learner) {
  as_learner(
    po("encode", method = "treatment") %>>%
      learner
  )
}

lrn_tree_encoded = make_encoded_learner(lrn("regr.rpart"))
lrn_tree_encoded$train(task, row_ids = split$train)

plot(lrn_tree_encoded$model$regr.rpart$model, compress = TRUE, margin = 0.1)
text(lrn_tree_encoded$model$regr.rpart$model, use.n = TRUE, cex = 0.7)

lrn_tree_encoded$predict(task, row_ids = split$test)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

#####Full and pruned tree#####
full_tree = make_encoded_learner(lrn("regr.rpart", cp = 0, xval = 5))
full_tree$train(task, row_ids = split$train)
full_tree$predict(task, row_ids = split$test)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

cp_table = full_tree$model$regr.rpart$model$cptable
min_row = which.min(cp_table[, "xerror"])
one_se_threshold = cp_table[min_row, "xerror"] + cp_table[min_row, "xstd"]

selected_row = which(cp_table[, "xerror"] <= one_se_threshold)[1]
selected_cp = cp_table[selected_row, "CP"]
selected_cp

pruned_tree = make_encoded_learner(lrn("regr.rpart", cp = selected_cp, xval = 5))
pruned_tree$train(task, row_ids = split$train)
pruned_tree$predict(task, row_ids = split$test)$score(msrs(c("regr.rmse", "regr.mae", "regr.rsq")))

#### KNN model ####
##### DEFINING KNN MODEL #####
graph_knn <- 
  po("encode", method = "one-hot") %>>%
  po("scale") %>>%
  lrn("regr.kknn")

learner_knn <- as_learner(graph_knn)

learner_knn$train(task, row_ids = train_set)

prediction_knn <- learner_knn$predict(task, row_ids = test_set)

prediction_knn$score(msr("regr.rmse"))
prediction_knn$score(msr("regr.mae"))
prediction_knn$score(msr("regr.rsq"))

##### KNN TUNING #####
learner_knn <- lrn(
  "regr.kknn",
  k = to_tune(1, 50)
)

graph_knn <- 
  po("encode", method = "one-hot") %>>%
  po("scale") %>>%
  learner_knn

graph_learner_knn <- as_learner(graph_knn)

at_knn <- auto_tuner(
  tuner = tnr("random_search"),
  learner = graph_learner_knn,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 20)
)

at_knn$train(task, row_ids = train_set)

at_knn$tuning_result

pred_knn <- at_knn$predict(task, row_ids = test_set)

pred_knn$score(msr("regr.rmse"))
pred_knn$score(msr("regr.mae"))
pred_knn$score(msr("regr.rsq"))

#### RF Model ####
##### DEFAULT RF MODEL #####
learner_rf <- lrn(
  "regr.ranger",             #ranger implements Random Forest
  importance = "impurity",
  predict_type = "response"
)

learner_rf$train(task, row_ids = train_set)

pred_rf <- learner_rf$predict(task, row_ids = test_set)

pred_rf$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### TARGET ENCODING #####
target_encoder <- po(
  "encodeimpact",
  affect_columns = selector_type(c("factor", "character"))
)

learner_rf <- lrn(
  "regr.ranger",
  importance = "impurity",
  predict_type = "response"
)

graph_rf_te <- target_encoder %>>% learner_rf

graph_learner_rf_te <- as_learner(graph_rf_te)

graph_learner_rf_te$train(
  task,
  row_ids = train_set
)

pred_rf_te <- graph_learner_rf_te$predict(
  task,
  row_ids = test_set
)

pred_rf_te$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### TUNING #####
learner_rf_tuned <- as_learner(
  po("encode", method = "one-hot") %>>%
    lrn("regr.ranger",
        importance = "impurity",
        predict_type = "response",
        num.trees = to_tune(c(500:1500)),
        min.node.size = to_tune(c(1:20)),
        mtry = to_tune(c(2:12)),
        sample.fraction = to_tune(p_dbl(lower = 0.5, upper = 1))
    )
)

at_rf <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_rf_tuned,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 25)
)

at_rf$train(task, row_ids = train_set)

at_rf$tuning_result

pred_tuned <- at_rf$predict(task, row_ids = test_set)

pred_tuned$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### TUNING WITH NEW NODE SIZES #####
learner_rf_tuned_2 <- as_learner(
  po("encode", method = "one-hot") %>>%
    lrn("regr.ranger",
        importance = "impurity",
        predict_type = "response",
        num.trees = to_tune(c(500:1500)),
        min.node.size = to_tune(c(100:2000)),
        mtry = to_tune(c(2:12)),
        sample.fraction = to_tune(p_dbl(lower = 0.5, upper = 1))
    )
)

at_rf_2 <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_rf_tuned_2,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 25)
)

at_rf_2$train(task, row_ids = train_set)

at_rf_2$tuning_result

pred_tuned_2 <- at_rf_2$predict(task, row_ids = test_set)

pred_tuned_2$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### RESULTS RF #####

results_rf <- data.frame(
  Model = c("Baseline", "RF Default", "RF Target Encoded", "RF Tuned", "RF Tuned Node"),
  RMSE = c(pred_baseline$score(msr("regr.rmse")), pred_rf$score(msr("regr.rmse")), pred_rf_te$score(msr("regr.rmse")), pred_tuned$score(msr("regr.rmse")), pred_tuned_2$score(msr("regr.rmse"))),
  MAE = c(pred_baseline$score(msr("regr.mae")), pred_rf$score(msr("regr.mae")), pred_rf_te$score(msr("regr.mae")), pred_tuned$score(msr("regr.mae")), pred_tuned_2$score(msr("regr.mae"))),
  RSQ = c(pred_baseline$score(msr("regr.rsq")), pred_rf$score(msr("regr.rsq")), pred_rf_te$score(msr("regr.rsq")), pred_tuned$score(msr("regr.rsq")), pred_tuned_2$score(msr("regr.rsq")))
)
results_rf

#### XGBoost ####
##### DEFAULT XGBoost MODEL #####
learner_xgb_default <- as_learner(
  po("encode", method = "one-hot") %>>%
    lrn(
      "regr.xgboost",
      objective = "reg:squarederror",
      eval_metric = "rmse"
    )
)

learner_xgb_default$train(
  task,
  row_ids = train_set
)

pred_xgb_default <- learner_xgb_default$predict(
  task,
  row_ids = test_set
)

pred_xgb_default$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### TARGET ENCODING #####

learner_xgb_te <- as_learner(
  po("encodeimpact") %>>% #encodeimpact performs target encoding
    lrn(
      "regr.xgboost",
      objective = "reg:squarederror",
      eval_metric = "rmse"
    )
)

learner_xgb_te$train(
  task,
  row_ids = train_set
)

pred_xgb_te <- learner_xgb_te$predict(
  task,
  row_ids = test_set
)

pred_xgb_te$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))
#The performance of the two models with the different encoding methods was very similar. However, the one-hot encoding lead to a slightly better model performance and therefore, we proceed with tuning that models hyperparameters.

##### TUNING #####

learner_xgb_tuned <- as_learner(
  po("encode", method = "one-hot") %>>%
    lrn(
      "regr.xgboost",
      eta = to_tune(p_dbl(0.01, 0.3)),
      max_depth = to_tune(c(1:10)),
      nrounds = to_tune(c(100:1000))
    )
)

at_xgb <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_xgb_tuned,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 50)
)

at_xgb$train(task, row_ids = train_set)

pred_xgb_tuned <- at_xgb$predict(task, row_ids = test_set)

pred_xgb_tuned$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### RESULTS XGBoost #####

results_xgb <- data.frame(
  Model = c("Baseline", "XGB Default", "XGB Target Encoded", "XGB Tuned"),
  RMSE = c(pred_baseline$score(msr("regr.rmse")), pred_xgb_default$score(msr("regr.rmse")), pred_xgb_te$score(msr("regr.rmse")), pred_xgb_tuned$score(msr("regr.rmse"))),
  MAE = c(pred_baseline$score(msr("regr.mae")), pred_xgb_default$score(msr("regr.mae")), pred_xgb_te$score(msr("regr.mae")), pred_xgb_tuned$score(msr("regr.mae"))),
  RSQ = c(pred_baseline$score(msr("regr.rsq")), pred_xgb_default$score(msr("regr.rsq")), pred_xgb_te$score(msr("regr.rsq")), pred_xgb_tuned$score(msr("regr.rsq")))
)
results_xgb

#### LightGBM ####
##### DEFAULT LIGHTGBM MODEL #####

learner_lgb_default <- as_learner(
  po("encode", method = "one-hot") %>>%
    lrn("regr.lightgbm",
        objective = "regression",
        verbose = -1
    )
)

learner_lgb_default$train(
  task,
  row_ids = train_set
)

pred_lgb_default <- learner_lgb_default$predict(
  task,
  row_ids = test_set
)

pred_lgb_default$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### TARGET ENCODING #####

learner_lgb_te <- as_learner(
  po("encodeimpact") %>>%
    lrn(
      "regr.lightgbm",
      objective = "regression",
      verbose = -1
    )
)

learner_lgb_te$train(
  task,
  row_ids = train_set
)

pred_lgb_te <- learner_lgb_te$predict(
  task,
  row_ids = test_set
)

pred_lgb_te$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

#Once again, the target encoding does not improve performance, so we proceed with tuning the hyperparameters of the default model.

##### TUNING #####

learner_lgb_tuned <- as_learner(
  po("encode", method = "one-hot") %>>%
    lrn("regr.lightgbm",
        learning_rate = to_tune(p_dbl(0.01, 0.1)),
        num_leaves = to_tune(c(20:80)),
        max_depth = to_tune(c(1:10)),
        min_data_in_leaf = to_tune(c(20:80)),
        num_iterations = to_tune(c(100:500))
    )
)

at_lgb <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_lgb_tuned,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 50)
)

at_lgb$train(task, row_ids = train_set)

pred_lgb_tuned <- at_lgb$predict(task, row_ids = test_set)

pred_lgb_tuned$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### TUNING WITH LARGER LEAVES #####

learner_lgb_tuned_2 <- as_learner(
  po("encode", method = "one-hot") %>>%
    lrn("regr.lightgbm",
        learning_rate = to_tune(p_dbl(0.01, 0.1)),
        num_leaves = to_tune(c(20:80)),
        max_depth = to_tune(c(1:10)),
        min_data_in_leaf = to_tune(c(100:2000)),
        num_iterations = to_tune(c(100:500))
    )
)

at_lgb_2 <- auto_tuner(
  tuner = tnr("random_search"),
  learner = learner_lgb_tuned,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.rmse"),
  terminator = trm("evals", n_evals = 50)
)

at_lgb_2$train(task, row_ids = train_set)

pred_lgb_tuned_2 <- at_lgb_2$predict(task, row_ids = test_set)

pred_lgb_tuned_2$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### RESULTS LGBM #####

results_lgb <- data.frame(
  Model = c("Baseline", "LGBM Default", "LGBM Target Encoded", "LGBM Tuned", "LGBM Tuned Leaf size"),
  RMSE = c(pred_baseline$score(msr("regr.rmse")), pred_lgb_default$score(msr("regr.rmse")), pred_lgb_te$score(msr("regr.rmse")), pred_lgb_tuned$score(msr("regr.rmse")), pred_lgb_tuned_2$score(msr("regr.rmse"))),
  MAE = c(pred_baseline$score(msr("regr.mae")), pred_lgb_default$score(msr("regr.mae")), pred_lgb_te$score(msr("regr.mae")), pred_lgb_tuned$score(msr("regr.mae")), pred_lgb_tuned_2$score(msr("regr.mae"))),
  RSQ = c(pred_baseline$score(msr("regr.rsq")), pred_lgb_default$score(msr("regr.rsq")), pred_lgb_te$score(msr("regr.rsq")), pred_lgb_tuned$score(msr("regr.rsq")), pred_lgb_tuned_2$score(msr("regr.rsq")))
)
results_lgb

#### DEFAULT NEURAL NETWORK MODEL ####

# MLP (Multi layer perceptron): Adam optimizer. SGD on mini batches (batch size = 32). See mlr_learners$get("regr.mlp")
graph_nn =
  po("encode", method = "one-hot") %>>%
  po("scale") %>>%
  lrn(
    "regr.mlp",
    epochs = 100,
    batch_size = 32,
    patience = 20, # Stopping process after 10 epochs without any improvement on validation set.
    measures_valid = msr("regr.rmse")
  )

learner_nn = as_learner(graph_nn)

set_validate(learner_nn, validate = 0.2) 
# Splitting the training data into new training and validation sets. This is used for early stopping.

learner_nn$train(
  task,
  row_ids = train_set
)

pred_nn = learner_nn$predict(
  task,
  row_ids = test_set
)

pred_nn$score(list(
  msr("regr.rmse"),
  msr("regr.mae"),
  msr("regr.rsq")
))

##### RESULTS NN #####
results_nn <- data.frame(
  Model = c("Baseline", "NN Default"),
  RMSE = c(pred_baseline$score(msr("regr.rmse")), pred_nn$score(msr("regr.rmse"))),
  MAE = c(pred_baseline$score(msr("regr.mae")), pred_nn$score(msr("regr.mae"))),
  RSQ = c(pred_baseline$score(msr("regr.rsq")), pred_nn$score(msr("regr.rsq")))
)
results_nn

#### Partial Linear Regression #####

#We start by making a matrix for the results we want 
results <- matrix(0, nrow=4, ncol=3)
rownames(results) = c("LicAge", "Exposure","DrivAge","BonusMalus")
colnames(results) = c("regr.rmse", "regr.mae", "regr.rsq")

#Now we make a model for each of our five chosen coefficient, where we focus on one off the coefficients and put all of the other together.

#####LicAge #####
task_amount_on_allother = as_task_regr(MPL_data_lump[split$train,], target="ClaimAmount")$select(setdiff(names(MPL_data_lump), c("LicAge", "ClaimAmount")))
task_LicAge_on_allother = as_task_regr(MPL_data_lump[split$train,], target="LicAge")$select(setdiff(names(MPL_data_lump), c("LicAge", "ClaimAmount")))

lrn_amount_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=ClaimAmount ~ s(Exposure) + s(RecordBeg) + s(RecordEnd) + VehAge + Gender + MariStat + SocioCateg + VehUsage + s(DrivAge) + HasKmLimit + s(BonusMalus) + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)
lrn_LicAge_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=LicAge ~ s(Exposure) + s(RecordBeg) + s(RecordEnd) + VehAge + Gender + MariStat + SocioCateg + VehUsage + s(DrivAge) + HasKmLimit + s(BonusMalus) + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)

cross_fit = function(task, learner, K){
  set.seed(2026)
  outer_resampling = rsmp("cv", folds=K)
  rr = resample(task, learner, outer_resampling)
  pred = rr$prediction() 
  pred$truth - pred$response
}

amount_residuals_smooth = cross_fit(task_amount_on_allother, lrn_amount_on_allother_smooth, K=5)
LicAge_residuals_smooth = cross_fit(task_LicAge_on_allother, lrn_LicAge_on_allother_smooth, K=5)

lm_amount_on_LicAge_smooth = lm(amount_residuals_smooth ~ LicAge_residuals_smooth)

results[1,] <- c(sqrt(mean(lm_amount_on_LicAge_smooth$residuals^2)),
                 mae(MPL_data_lump$ClaimAmount, predict.lm(lm_amount_on_LicAge_smooth,MPL_data_lump[split$train,])),
                 summary(lm_amount_on_LicAge_smooth)$r.squared)
#####Exposure#####
task_amount_on_allother = as_task_regr(MPL_data_lump[split$train,], target="ClaimAmount")$select(setdiff(names(MPL_data_lump), c("Exposure", "ClaimAmount")))
task_Exposure_on_allother = as_task_regr(MPL_data_lump[split$train,], target="Exposure")$select(setdiff(names(MPL_data_lump), c("Exposure", "ClaimAmount")))

lrn_amount_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=ClaimAmount ~ s(LicAge) + s(RecordBeg) + s(RecordEnd) + VehAge + Gender + MariStat + SocioCateg + VehUsage + s(DrivAge) + HasKmLimit + s(BonusMalus) + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)
lrn_Exposure_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=Exposure ~ LicAge + RecordBeg + RecordEnd + VehAge + Gender + MariStat + SocioCateg + VehUsage + s(DrivAge) + HasKmLimit + s(BonusMalus) + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)

amount_residuals_smooth = cross_fit(task_amount_on_allother, lrn_amount_on_allother_smooth, K=5)
Exposure_residuals_smooth = cross_fit(task_Exposure_on_allother, lrn_Exposure_on_allother_smooth, K=5)

lm_amount_on_Exposure_smooth = lm(amount_residuals_smooth ~ Exposure_residuals_smooth)

results[2,] <- c(sqrt(mean(lm_amount_on_Exposure_smooth$residuals^2)),
                 mae(MPL_data_lump$ClaimAmount, predict.lm(lm_amount_on_Exposure_smooth,MPL_data_lump[split$train,])),
                 summary(lm_amount_on_Exposure_smooth)$r.squared)
#####DrivAge#####
task_amount_on_allother = as_task_regr(MPL_data_lump[split$train,], target="ClaimAmount")$select(setdiff(names(MPL_data_lump), c("DrivAge", "ClaimAmount")))
task_DrivAge_on_allother = as_task_regr(MPL_data_lump[split$train,], target="DrivAge")$select(setdiff(names(MPL_data_lump), c("DrivAge", "ClaimAmount")))

lrn_amount_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=ClaimAmount ~ s(LicAge) + s(Exposure) + s(RecordBeg) + s(RecordEnd) + VehAge + Gender + MariStat + SocioCateg + VehUsage + HasKmLimit + s(BonusMalus) + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)
lrn_DrivAge_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=DrivAge ~ s(LicAge) + s(Exposure) + s(RecordBeg) + s(RecordEnd) + VehAge + Gender + MariStat + SocioCateg + VehUsage + HasKmLimit + s(BonusMalus) + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)

amount_residuals_smooth = cross_fit(task_amount_on_allother, lrn_amount_on_allother_smooth, K=5)
DrivAge_residuals_smooth = cross_fit(task_DrivAge_on_allother, lrn_DrivAge_on_allother_smooth, K=5)

lm_amount_on_DrivAge_smooth = lm(amount_residuals_smooth ~ DrivAge_residuals_smooth)

results[3,] <- c(sqrt(mean(lm_amount_on_DrivAge_smooth$residuals^2)),
                 mae(MPL_data_lump$ClaimAmount, predict.lm(lm_amount_on_DrivAge_smooth,MPL_data_lump[split$train,])),
                 summary(lm_amount_on_DrivAge_smooth)$r.squared)
#####BonusMalus #####
task_amount_on_allother = as_task_regr(MPL_data_lump[split$train,], target="ClaimAmount")$select(setdiff(names(MPL_data_lump), c("BonusMalus", "ClaimAmount")))
task_BonusMalus_on_allother = as_task_regr(MPL_data_lump[split$train,], target="BonusMalus")$select(setdiff(names(MPL_data_lump), c("BonusMalus", "ClaimAmount")))

lrn_amount_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=ClaimAmount ~ s(LicAge) + s(Exposure) + s(RecordBeg) + s(RecordEnd) + VehAge + Gender + MariStat + SocioCateg + VehUsage + s(DrivAge) + HasKmLimit + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)
lrn_BonusMalus_on_allother_smooth = auto_tuner(
  tnr("random_search"),
  lrn(
    "regr.gam",
    formula=BonusMalus ~ s(LicAge) + s(Exposure) + s(RecordBeg) + s(RecordEnd) + VehAge + Gender + MariStat + SocioCateg + VehUsage + s(DrivAge) + HasKmLimit + VehBody + VehPrice + VehEngine+VehEnergy+VehMaxSpeed + VehClass + s(RiskVar) + Garage,
    gamma = to_tune(1,1e3,logscale=TRUE),
    select = to_tune(c(TRUE, FALSE)),
    drop.unused.levels=FALSE # Default is TRUE, but this may cause errors if splits are unlucky
  ),
  measure = msr("regr.rmse"),
  resampling = rsmp("cv", folds=5),
  terminator = trm("evals", n_evals=20)
)

amount_residuals_smooth = cross_fit(task_amount_on_allother, lrn_amount_on_allother_smooth, K=5)
BonusMalus_residuals_smooth = cross_fit(task_BonusMalus_on_allother, lrn_BonusMalus_on_allother_smooth, K=5)

lm_amount_on_BonusMalus_smooth = lm(amount_residuals_smooth ~ BonusMalus_residuals_smooth)

results[4,] <- c(sqrt(mean(lm_amount_on_BonusMalus_smooth$residuals^2)),
                 mae(MPL_data_lump$ClaimAmount, predict.lm(lm_amount_on_BonusMalus_smooth,MPL_data_lump[split$test,])),
                 summary(lm_amount_on_BonusMalus_smooth)$r.squared)

#### Predictions for the three chosen models: ####
# 1) Second tuned random forrest ; at_rf_2
at_rf_2$predict_newdata(gns_kunde)
at_rf_2$predict_newdata(eks_kunde)

# 2) Second tuned LightGBM ; at_lgb_2
at_lgb_2$predict_newdata(gns_kunde)
at_lgb_2$predict_newdata(eks_kunde)

# 3) Neural network ; learner_nn 
learner_nn$predict_newdata(gns_kunde)
learner_nn$predict_newdata(eks_kunde)

#### Shapley values ####
predict_fun <- function(model,task){ 
  Predictor$new(
    model = model,
    data = task$data(cols = task$feature_names),
    y = task$data(cols = task$target_names)[[1]]
  )
}

shapley_values <- function(model,task,new_data,n=1000){
  Shapley$new(
    predictor = predict_fun(model,task),
    x.interest = new_data,
    sample.size = n
  )$results$phi
}

#####Shapley values for gns_kunde: Random forrest second model made with tuning#####
df_shapley_values <- 
  data.frame(
    x = task$feature_names,
    y = shapley_values(at_rf_2,task,as.data.frame(gns_kunde[, task$feature_names]),1000)
  )

ggplot(df_shapley_values, aes(x=y,y=x)) +
  geom_col() +
  labs(x="Shapley Value", y = "Feature", title = "Average customer, Model: Random Forrest")
#####Shapley values for eks_kunde: Random forrest second model made with tuning #####
df_shapley_values_eks <- 
  data.frame(
    x = task$feature_names,
    y = shapley_values(at_rf_2,task,as.data.frame(eks_kunde[, task$feature_names]),1000)
  )

ggplot(df_shapley_values_eks, aes(x=y,y=x)) +
  geom_col() +
  labs(x="Shapley Value", y = "Feature", title = "Extreme customer, Model: Random Forrest")

#####Shapley values for gns_kunde: LightGBM model made with tuning#####
df_shapley_values_lgbm <- 
  data.frame(
    x = task$feature_names,
    y = shapley_values(at_lgb_2,task,as.data.frame(gns_kunde[, task$feature_names]),1000)
  )

ggplot(df_shapley_values_lgbm, aes(x=y,y=x)) +
  geom_col() +
  labs(x="Shapley Value", y = "Feature", title = "Average customer, Model: LightGBM")

#####Shapley values for eks_kunde: LightGBM model made with tuning#####
df_shapley_values_lgbm_eks <- 
  data.frame(
    x = task$feature_names,
    y = shapley_values(at_lgb_2,task,as.data.frame(eks_kunde[, task$feature_names]),1000)
  )

ggplot(df_shapley_values_lgbm_eks, aes(x=y,y=x)) +
  geom_col() +
  labs(x="Shapley Value", y = "Feature", title = "Extreme customer, Model: LightGBM")

#####Shapley values for gns_kunde: NN model made with tuning #####
df_shapley_values_NN <- 
  data.frame(
    x = task$feature_names,
    y = shapley_values(learner_nn,task,as.data.frame(gns_kunde[, task$feature_names]),1000)
  )

ggplot(df_shapley_values_NN, aes(x=y,y=x)) +
  geom_col() +
  labs(x="Shapley Value", y = "Feature", title = "Average customer, Model NN")
#####Shapley values for eks_kunde: NN model made with tuning#####
df_shapley_values_NN_eks <- 
  data.frame(
    x = task$feature_names,
    y = shapley_values(learner_nn,task,as.data.frame(eks_kunde[, task$feature_names]),1000)
  )

ggplot(df_shapley_values_NN_eks, aes(x=y,y=x)) +
  geom_col() +
  labs(x="Shapley Value", y = "Feature", title = "Extreme customer, Model NN")
#### PDP and ALE ####
explainer_fun <- function(learner_trained, task){
  explainer = DALEXtra::explain_mlr3(
    learner_trained,
    data = task$data(cols = task$feature_names),
    y = task$data(cols = task$target_names)[[1]],
    colorize = FALSE
  )
}

#####PDP and ALE for random forrest#####
explainer <- explainer_fun(at_rf_2, task)

pdps <- DALEX::model_profile(explainer, type="partial")
ales <- DALEX::model_profile(explainer, type="accumulated")

plot(pdps)
plot(ales)

##### PDP for Categoricals #####
pdp_categ <- model_profile(
  explainer,
  variables = c("VehAge", "Gender", "MariStat", "SocioCateg", "VehUsage", "VehBody", "VehPrice", "VehEngine", "VehEnergy", "VehMaxSpeed", "VehClass"),
  type = "partial"
)

plot(pdp_categ) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      hjust = 1
    )
  )



##### ALE for Categoricals #####
ale_categ <- model_profile(
  explainer,
  variables = c("VehAge", "Gender", "MariStat", "SocioCateg", "VehUsage", "VehBody", "VehPrice", "VehEngine", "VehEnergy", "VehMaxSpeed", "VehClass"),
  type = "accumulated"
)

plot(ale_categ) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      hjust = 1
    )
  )


##### Beeswarm plot #####
#SHAP values

predict_fun <- function(model,task){ 
  Predictor$new(
    model = model,
    data = task$data(cols = task$feature_names),
    y = task$data(cols = task$target_names)[[1]]
  )
}

predictor_rf <- predict_fun(model = at_rf_2, task = task)

shap_values <- fastshap::explain(
  predictor_rf$model,
  X = as.data.frame(task$data(rows = split$test)), 
  pred_wrapper = function(object, newdata) {
    predict(object, newdata = newdata)
  },
  nsim = 10
)

#Shapviz object
sv <- shapviz::shapviz(shap_values, X = as.data.frame(task$data(rows = split$test)), baseline = mean(MPL_data_lump$ClaimAmount))
#Beeswarm plot
shapviz::sv_importance(sv, kind = "beeswarm")



#### Sensitive or protected features - Gender ####
test_data <- MPL_data_lump[test_set, ]

test_male   <- test_data
test_female <- test_data

test_male$Gender   <- "Male"
test_female$Gender <- "Female"


pred_male <- at_rf_2$learner$predict_newdata(test_male)
pred_female <- at_rf_2$learner$predict_newdata(test_female)



pred_fair <- (pred_male$response +
                pred_female$response) / 2


y_true <- test_data$ClaimAmount



rmse_fair <- sqrt(mean((y_true - pred_fair)^2))

mae_fair <- mean(abs(y_true - pred_fair))

rsq_fair <- 1 -
  sum((y_true - pred_fair)^2) /
  sum((y_true - mean(y_true))^2)

results_fair <- data.frame(
  Model = "RF Fair",
  RMSE = rmse_fair,
  MAE  = mae_fair,
  RSQ  = rsq_fair
)

results_fair


####Predictions for sensitive model ####
# Gns_kunde
gns_kunde_sens_m <- gns_kunde
gns_kunde_sens_k <- gns_kunde
gns_kunde_sens_k$Gender <- "Female"

1/2*(at_rf_2$predict_newdata(gns_kunde_sens_m)$response+at_rf_2$predict_newdata(gns_kunde_sens_k)$response)

# Eks_kunde
eks_kunde_sens_m <- eks_kunde
eks_kunde_sens_k <- eks_kunde
eks_kunde_sens_k$Gender <- "Female"

1/2*(at_rf_2$predict_newdata(eks_kunde_sens_m)$response+at_rf_2$predict_newdata(eks_kunde_sens_k)$response)


