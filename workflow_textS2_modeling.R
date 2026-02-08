############################################################
# Workflow-level scripts for descriptor cleaning,
# feature selection (RFE/Boruta), tuning & CV evaluation,
# and final model fitting (Random Forest as example).
#
# Inputs (expected):
#   1) raw_descriptor.xlsx : descriptor matrix (numeric columns),
#      may contain character columns (will be coerced).
#   2) Feature_sel.xlsx    : columns = Compound, (RT or logRF), descriptors...
#
# Outputs (written to working directory):
#   - cleaned_descriptors.csv
#   - cleaned_feature_list.csv
#   - rfe_results.csv
#   - selected_features_rfe_{MODEL}.csv
#   - selected_features_boruta_confirmed.csv
#   - rf_baseline_metrics.csv
#   - rf_cv_metrics.csv
############################################################

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(caret)
  library(tidyr)
  library(ggplot2)
  library(Boruta)
  library(randomForest)
  library(yardstick)
  library(tidymodels)
  library(finetune)
})

# ---------------------------
# User settings
# ---------------------------
set.seed(100)

# Choose target variable in Feature_sel.xlsx
# y_name <- "logRF"
y_name <- "RT"

id_name <- "Compound"

# Descriptor cleaning settings
cor_cutoff <- 0.8

# RFE settings
rfe_folds <- 5
rfe_sizes <- c(2:25, 30, 35, 40, 45, 50)

# tidymodels CV & tuning settings
N_v <- 10       # v-fold CV
N_level <- 5    # grid levels for regular grid
select_metric <- "rsq"  # "rsq" or "mae"

# ---------------------------
# Part A: Descriptor cleaning
# ---------------------------
cat("=== Part A: Descriptor cleaning ===\n")

features <- read_excel("raw_descriptor.xlsx") %>% as.data.frame()

detect_abnormal_values <- function(col) {
  abnormal_pattern <- "[^0-9eE.+-]|inf|Inf|INF|infinity|Infinity|INFINITY"
  grepl(abnormal_pattern, as.character(col))
}

# Coerce characters to numeric and replace abnormal strings with NA
features_fixed <- features %>%
  mutate(across(everything(), ~{
    if (is.character(.x)) {
      abnormal_flags <- detect_abnormal_values(.x)
      if (any(abnormal_flags)) {
        cat("Discover outliers: column", cur_column(), "->",
            paste(unique(.x[abnormal_flags]), collapse = ", "), "\n")
        .x[abnormal_flags] <- NA
      }
      as.numeric(.x)
    } else {
      suppressWarnings(as.numeric(.x))
    }
  }))

features <- features_fixed

# 1) Remove columns with any NA
na_counts <- colSums(is.na(features))
features_clean <- features[, na_counts == 0, drop = FALSE]

# 2) Remove near-zero variance columns
nzv <- nearZeroVar(features_clean, saveMetrics = TRUE)
if (any(nzv$zeroVar)) {
  features_clean <- features_clean[, !nzv$zeroVar, drop = FALSE]
}

# 3) Replace Inf with NA then remove columns with NA
features_clean <- features_clean %>%
  mutate(across(everything(), ~ ifelse(is.infinite(.x), NA, .x)))

na_counts2 <- colSums(is.na(features_clean))
features_clean <- features_clean[, na_counts2 == 0, drop = FALSE]

# 4) Remove highly correlated columns
if (ncol(features_clean) >= 2) {
  cor_matrix <- cor(features_clean, use = "pairwise.complete.obs")
  high_cor_cols <- findCorrelation(cor_matrix, cutoff = cor_cutoff, names = TRUE)
  if (length(high_cor_cols) > 0) {
    features_clean <- features_clean[, !(colnames(features_clean) %in% high_cor_cols), drop = FALSE]
  }
}

cat("Remaining descriptors after cleaning:", ncol(features_clean), "\n")

# Save cleaned descriptors and feature list (useful as "checkpoint" outputs)
write.csv(features_clean, "cleaned_descriptors.csv", row.names = FALSE)
write.csv(data.frame(feature = colnames(features_clean)),
          "cleaned_feature_list.csv", row.names = FALSE)

# ---------------------------
# Part B: Load modeling dataset and define x/y
# ---------------------------
cat("\n=== Part B: Load modeling dataset (Feature_sel.xlsx) ===\n")

Feature_sel <- read_excel("Feature_sel.xlsx") %>% as.data.frame()

stopifnot(id_name %in% names(Feature_sel))
stopifnot(y_name %in% names(Feature_sel))

desc_cols <- setdiff(names(Feature_sel), c(id_name, y_name))
Feature_sel[desc_cols] <- lapply(Feature_sel[desc_cols], function(z) as.numeric(as.character(z)))

y <- Feature_sel[[y_name]]
x <- Feature_sel[, desc_cols, drop = FALSE]
model_data <- Feature_sel[, c(y_name, desc_cols), drop = FALSE]
rownames(model_data) <- Feature_sel[[id_name]]

cat("Target:", y_name, "\n")
cat("n:", nrow(model_data), "p:", ncol(x), "\n")

# Common formula used later
form <- reformulate(colnames(x), response = y_name)

# ---------------------------
# Part C: RFE (LM / RF / SVM)
# ---------------------------
cat("\n=== Part C: RFE feature selection ===\n")

rfeControl_lm <- rfeControl(functions = lmFuncs,
                            method = "cv",
                            number = rfe_folds,
                            saveDetails = TRUE,
                            allowParallel = TRUE)

rfeControl_rf <- rfeControl(functions = rfFuncs,
                            method = "cv",
                            number = rfe_folds,
                            saveDetails = TRUE,
                            allowParallel = TRUE)

rfeControl_svm <- rfeControl(functions = caretFuncs,
                             method = "cv",
                             number = rfe_folds,
                             saveDetails = TRUE,
                             allowParallel = TRUE)

lmProfile <- rfe(x, y, sizes = rfe_sizes,  rfeControl = rfeControl_lm)
rfProfile <- rfe(x, y, sizes = rfe_sizes,  rfeControl = rfeControl_rf)
svmProfile <- rfe(x, y, sizes = rfe_sizes, method = "svmRadial", rfeControl = rfeControl_svm)

# Combine & save RFE results
rfe_res <- bind_rows(
  lmProfile$results %>% mutate(model = "LM"),
  rfProfile$results %>% mutate(model = "RF"),
  svmProfile$results %>% mutate(model = "SVM")
)
write.csv(rfe_res, "rfe_results.csv", row.names = FALSE)

# Save selected feature sets for transparency
write.csv(data.frame(feature = predictors(lmProfile)),  "selected_features_rfe_LM.csv", row.names = FALSE)
write.csv(data.frame(feature = predictors(rfProfile)),  "selected_features_rfe_RF.csv", row.names = FALSE)
write.csv(data.frame(feature = predictors(svmProfile)), "selected_features_rfe_SVM.csv", row.names = FALSE)

# Optional plot (viewer can regenerate)
ggplot(rfe_res, aes(x = Variables, y = Rsquared, color = model)) +
  geom_line() + geom_point() +
  labs(x = "Number of features", y = "R-squared")

# ---------------------------
# Part D: Boruta (confirmed features)
# ---------------------------
cat("\n=== Part D: Boruta feature selection ===\n")

boruta_form <- reformulate(termlabels = colnames(x), response = y_name)
Var.Selec <- Boruta(boruta_form, data = model_data, maxRuns = 1000, doTrace = 1)

# Extract confirmed features
boruta_stats <- attStats(Var.Selec)
boruta_confirmed <- rownames(boruta_stats)[boruta_stats$decision == "Confirmed"]

cat("Boruta confirmed features:", length(boruta_confirmed), "\n")
write.csv(data.frame(feature = boruta_confirmed),
          "selected_features_boruta_confirmed.csv", row.names = FALSE)

# ---------------------------
# Part E: Baseline RF (default params) + metrics
# ---------------------------
cat("\n=== Part E: Baseline randomForest (default params) ===\n")

sel_model <- randomForest(form, data = model_data, importance = TRUE)
pre <- predict(sel_model, newdata = model_data)

rf_base_metrics <- tibble(truth = model_data[[y_name]], estimate = pre) %>%
  summarise(
    rsq = rsq_vec(truth, estimate),
    mae = mae_vec(truth, estimate)
  )

print(rf_base_metrics)
write.csv(rf_base_metrics, "rf_baseline_metrics.csv", row.names = FALSE)

# ---------------------------
# Part F: tidymodels tuning + CV evaluation (RF example)
# ---------------------------
cat("\n=== Part F: tidymodels tuning + CV evaluation (RF example) ===\n")

rf_tune <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

rf_recipe <- recipe(form, data = model_data)

rf_wf <- workflow() %>%
  add_model(rf_tune) %>%
  add_recipe(rf_recipe)

cv_data <- vfold_cv(model_data, v = N_v)

rf_grid <- rf_wf %>%
  extract_parameter_set_dials() %>%
  grid_regular(levels = N_level)

rf_hyper_tune <- tune_race_anova(
  object = rf_wf,
  resamples = cv_data,
  grid = rf_grid,
  metrics = metric_set(rsq, mae),
  control = control_race(verbose_elim = TRUE, verbose = TRUE)
)

# Select best parameters using a clear rule
best_rf <- select_best(rf_hyper_tune, metric = select_metric)

final_rf_wf <- finalize_workflow(rf_wf, best_rf)

rf_model_cv <- fit_resamples(
  final_rf_wf,
  cv_data,
  metrics = metric_set(rsq, mae),
  control = control_resamples(save_pred = TRUE)
)

rf_cv_metrics <- collect_metrics(rf_model_cv)
print(rf_cv_metrics)
write.csv(rf_cv_metrics, "rf_cv_metrics.csv", row.names = FALSE)

# Final model fitted on full dataset
rf_fit_full <- fit(final_rf_wf, data = model_data)

