library(naivebayes)
library(discrim)
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(kknn)
library(kernlab)
install.packages("rsem")
library(rsem)
library(ggplot2)
install.packages("modeltime")
library(modeltime)


Ctrain <- vroom("C:/Users/isaac/Downloads/ghols/traind.csv")
#Itrain <- vroom("C:/Users/isaac/Downloads/trainWithMissingValues.csv")
Ctest <- vroom("C:/Users/isaac/Downloads/ghols/test.csv")

my_recipe <- recipe(type ~., data=Ctrain) %>%
  #step_impute_mean(all_numeric_predictors())
  step_impute_knn(var, impute_with = all_numeric_predictors()
                    imp_vars(all_numeric_predictors()), neighbors= 5)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = Itrain)
baked2 <- bake(prep, new_data = Ctrain)

rmse_vec(Ctrain[is.na(Itrain)], baked[is.na(Itrain)])




my_recipe <- recipe(type ~., data=Ctrain) %>%    # combines categorical values that occur <% into an "other" value
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(color)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = Ctrain)

## knn model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

#TUNING
tuning_grid <- grid_regular(neighbors(),
                            levels = 5)


## Split data for CV
folds <- vfold_cv(Ctrain, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
finalknn_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Ctrain)

## Predict
knn_pred <- finalknn_wf %>%
  predict(new_data = Ctest, type="class")


#FORMAT
knn_pred <- knn_pred %>%
  select(type = .pred_class)
Aknnpred <- data.frame(Id = Ctest$id, knn_pred)

#MAKE FILE
vroom_write(Aknnpred, file="KNNghoul.csv", delim=",")









nn_recipe <- recipe(type ~., data=Ctrain) %>%
  update_role(id, new_role="id") %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

prep <- prep(nn_recipe)
bake(prep(nn_recipe), new_data=Ctrain)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
            set_engine("nnet") %>%
            set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 10)), levels=3)

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model) 

folds <- vfold_cv(Ctrain, v = 5, repeats=3)

tuned_nn <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy))

bestTuned <- tuned_nn %>%
  select_best("accuracy")

finalnn_wf <-
  nn_wf %>%
  finalize_workflow(bestTuned) %>%
  fit(data=Ctrain)


nn_pred2 <- finalnn_wf %>%
  predict(new_data = Ctest, type="prob")

tuned_nn %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()




#-----------------boost--------
install.packages("bonsai")
install.packages("lightgbm")
library(bonsai)
library(lightgbm)


Ctrain <- vroom("C:/Users/isaac/Downloads/ghols/traind.csv")
#Itrain <- vroom("C:/Users/isaac/Downloads/trainWithMissingValues.csv")
Ctest <- vroom("C:/Users/isaac/Downloads/ghols/test.csv")

my_recipe <- recipe(type ~., data=Ctrain) %>%    # combines categorical values that occur <% into an "other" value
  step_normalize(all_numeric_predictors()) %>%
  step_lencode_glm(color, outcome = vars(type))

prep <- prep(my_recipe)
baked <- bake(prep, new_data = Ctrain)


boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

for_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

#TUNING
tuning_grid <- grid_regular(tree_depth(),trees(),learn_rate(),
                            levels = 3)


## Split data for CV
folds <- vfold_cv(Ctrain, v = 3, repeats=2)

## Run the CV
CV_results <- for_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")

## Finalize the Workflow & fit it
finalboost_wf <-
  for_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=Ctrain)

## Predict
boost_pred <- finalboost_wf %>%
  predict(new_data = Ctest, type="prob")


#FORMAT
boost_pred <- boost_pred %>%
  mutate(prediction = case_when(
    .pred_Ghost > .pred_Ghoul & .pred_Ghost > .pred_Goblin ~ "Ghost",
    .pred_Ghoul > .pred_Ghost & .pred_Ghoul > .pred_Goblin ~ "Ghoul",
    .pred_Goblin > .pred_Ghost & .pred_Goblin > .pred_Ghoul ~ "Goblin",
    TRUE ~ "tie"
  ))
boosted_pred <- data.frame(Id = Ctest$id, type = boost_pred$prediction)

#MAKE FILE
vroom_write(boosted_pred, file="BoostGhoul.csv", delim=",")




#---------Naive Bayes
#MyMod
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naivebayes eng


#WORKFLOW
nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

#TUNING
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 3)


## Split data for CV
folds <- vfold_cv(Ctrain, v = 3, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")

## Finalize the Workflow & fit it
finalnb_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Ctrain)

## Predict
nb_pred <- finalnb_wf %>%
  predict(new_data = Ctest, type="prob")


nb_pred <- nb_pred %>%
  mutate(prediction = case_when(
    .pred_Ghost > .pred_Ghoul & .pred_Ghost > .pred_Goblin ~ "Ghost",
    .pred_Ghoul > .pred_Ghost & .pred_Ghoul > .pred_Goblin ~ "Ghoul",
    .pred_Goblin > .pred_Ghost & .pred_Goblin > .pred_Ghoul ~ "Goblin",
    TRUE ~ "tie"
  ))
finnb_pred <- data.frame(Id = Ctest$id, type = nb_pred$prediction)

#MAKE FILE
vroom_write(finnb_pred, file="NBGhoul.csv", delim=",")






