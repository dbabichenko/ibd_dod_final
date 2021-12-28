from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import numpy as np
from os import path
import os
import pandas as pd
import xgboost as xgb

from model_svm import ModelSVM
from model_xgboost_continuous import ModelXGBoostContinuous
from model_xgboost_multiclass import ModelXGBoostMulticlass
from model_svm import ModelSVM

# ==========================================================================
# config/setup
# ==========================================================================
global_random_seed = 42

# ==========================================================================
# load data
# ==========================================================================
data_file_path = path.join(os.getcwd(), "ibddataprocessing2020",
                           "scripts", "model", "assets", "preprocessed_data.csv")
df_data = pd.read_csv(data_file_path)

# drop unnecessary columns
df_data = df_data.drop(columns=[
    'project_patient_id',
    'window_id',
    'window_start',
    'window_end',
    'window_train_breakpoint',
    'grand_max_date',
    'grand_min_date',
    'grand_timespan',
    'ethnic_group',
    'race',
    'is_alive',
])

df_continuous_targets = df_data[[
    column for column in df_data.columns if column.startswith('target')]]
df_data.drop(columns=df_continuous_targets.columns, inplace=True)

# encode multiclass results
df_classification_targets = df_continuous_targets.copy()
for target in df_classification_targets.columns:
    df_classification_targets[target] = df_classification_targets[target].apply(lambda x: 0 if x < 3 else 1)

continuous_results = ModelXGBoostContinuous().train_and_validate(df_data, df_continuous_targets, global_random_seed)
for result_key in continuous_results.keys():
    result = continuous_results[result_key]
    print("-------------------------------------------------------------------")
    print("Target:", result_key)
    print("Folds:", result['folds'])
    print("Average R^2 in cross val", result['avgR2'])
    print("MAE on test:", result['test_mae'])
    print("RMSE on test:", result['testRmse'])
    print("Explained variance on test", result['testEv'])


print('=============XGBOOST================')
multi_results = ModelXGBoostMulticlass().train_and_validate(
    df_data, df_classification_targets, global_random_seed)
for target in df_classification_targets.columns:
    print(f"Target {target}:", multi_results[target])

print('===============SVM==================')
svm_results = ModelSVM().train_and_validate(
    df_data, df_classification_targets, global_random_seed)
for target in df_classification_targets.columns:
    print(f"Target {target}:", svm_results[target])

print('=============CLASSIFICATION PROPORTIONS================')
for target in df_classification_targets.columns:
    print(target, df_classification_targets[target].value_counts())

# recursive feature elimination (RFE)
# graphical bayesian network tool: genie
# Peter & Clark algorithm
