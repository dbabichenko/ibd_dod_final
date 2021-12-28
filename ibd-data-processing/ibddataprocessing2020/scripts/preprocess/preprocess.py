from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import sys

from preprocess_utils import load_file
from PreprocessConfig import PreprocessConfig, ConfigKey
from PreprocessingContext import PreprocessingContext
from PreprocessingWindowMode import PreprocessingWindowMode
from preprocess_encounters import preprocess_encounters
from preprocess_labs import preprocess_labs
from preprocess_meds import preprocess_meds
from preprocess_patient_history_ranges import preprocess_patient_history_ranges
from preprocess_patient_windows import preprocess_patient_windows
from preprocess_patients import preprocess_patients
from preprocess_problems import preprocess_problems
from preprocess_target_vars import preprocess_target_vars
from preprocess_tobacco_use import preprocess_tobacco_use

# =============================================================="
# SCRIPT SETUP
# =============================================================="
zero = timedelta(days=0)
one_year = timedelta(days=365.25)
four_years = one_year * 4
output_path = os.path.join(
    os.getcwd(), "ibddataprocessing2020", "scripts", "preprocess", "output")

# this lets us selectively execute parts of the script for debugging,
# or if we want to train models based on some subset of the available features.
#
# during execution of the script, if config is None, data corresponding to
# ALL string keys (e.g. 'labs', 'meds', etc.) WILL be evaluated and preprocessed.
# if config is defined, ONLY data associated with present keys will be used.
# so if you want to run the entire script, no holds barred, you can just comment out
# config. if you want to run it selectively, make sure it's uncommented and
# contains only keys you're interested in.
preprocessing_context = PreprocessingContext(
    config=PreprocessConfig([
        ConfigKey.ENCOUNTERS,
        ConfigKey.LABS,
        ConfigKey.MEDS,
        ConfigKey.PROBLEMS,
        ConfigKey.PROCEDURES,
        ConfigKey.TARGET,
        ConfigKey.VITALS,
    ]),
    files={
        'encounters': 'encounters_merged.csv',
        'labs': 'labs_merged.csv',
        'meds': 'meds_merged.csv',
        'patients': 'patients_merged.csv',
        'problems': 'problem_list_merged.csv',
        'procedures': 'procedures_merged.csv',
        'targets': 'hospitalizations_and_er_visits_merged.csv'
    },
    window_lengths={
        # how many years to use for prediction (e.g. using 3 years to predict the following year)
        'train': one_year,
        'test': zero,  # how many years to aggregate for period we're predicting (in the above example, just one)
        'total': one_year,  # total length of each patient window (basically train + test)
        'offset': one_year,
    },
    window_mode=PreprocessingWindowMode.ONE_WINDOW_PER_PATIENT_YEAR,
    working_directory=os.getcwd()
)

print()
print("# ==============================================================")
print("# WELCOME")
print("# ==============================================================")
print(f"Loading and preprocessing patient data from root directory '{preprocessing_context.working_directory}'.",)
print()

# this frame will ultimately hold all of our data - one record per patient per year of data.
# all of the scripts called by this one (i.e. preprocess_encounters, etc.) return a frame
# with the same index that gets appended to df_complete. At the end, we dump df_complete
# to a file.
df_complete = pd.DataFrame(columns=["PROJECT_PATIENT_ID"])
df_complete.set_index(["PROJECT_PATIENT_ID"], inplace=True)

print("# ==============================================================")
print("# Loading patients")
print("# ==============================================================")
df_patients = load_file(preprocessing_context.files['patients'])
df_patients.set_index("PROJECT_PATIENT_ID", inplace=True)
print(f"{df_patients.shape[0]} patients found.")
print()

# ==============================================================
# This block determines the total length (in time) of the data
# we have for each patients. Eligible patients have to have at least
# four years between their first appearance in any dataset and their
# last.
#
# Once we've isolated patients who meet that criterion, we then
# record their first appearance date and the date four years after that,
# as we only want data within that window, even if a patient has more.
# ==============================================================
print("# ==============================================================")
print("# Resolving patient history length")
print("# ==============================================================")
df_patients = pd.merge(df_patients, preprocess_patient_history_ranges(
    df_patients.index, preprocessing_context), left_index=True, right_index=True)

# filter out all patients with less than four years of data
print()
print("Filtering patients...")

df_patients['GRAND_TIMESPAN'] = df_patients['GRAND_MAX_DATE'] - df_patients['GRAND_MIN_DATE']
df_patients = df_patients[df_patients['GRAND_TIMESPAN'] > four_years]
print("Patients with sufficient data (four years or more): ", df_patients.shape[0])

# RESOLVE PATIENT WINDOWS
# this gives of a frame with one record per eligible window of treatment
# that is, a patient who was treated from 2012-2020 will have 5 windows if the window size is set to 4 years,
# 2012-2016, 2013-2017, 2014-2018, 2015-2019, and 2016-2020
print()
print("# ==============================================================")
print("# Resolving patient windows")
print("# ==============================================================")
df_patient_windows = preprocess_patient_windows(df_patients, preprocessing_context)

print(f"{len(df_patients)} patients had {len(df_patient_windows)} windows.")

# TODO: maybe just declare df_complete here
df_complete = df_patient_windows.copy()
df_complete.to_csv(f"{output_path}\\complete-patients.csv")

# ==============================================================
# PREPROCESSING: tobacco use
# ==============================================================
if preprocessing_context.config.is_enabled(ConfigKey.TOBACCO):
    print()
    print("# ==============================================================")
    print("# Preprocessing tobacco use")
    print("# ==============================================================")
    print("Working...")
    df_tobacco_use = preprocess_tobacco_use(df_patient_windows, preprocessing_context)
    df_complete = df_complete.merge(df_tobacco_use, how='left', left_index=True, right_index=True)
    df_complete[df_tobacco_use.columns].fillna(0, inplace=True)
    df_complete.to_csv(f"{output_path}\\complete-tobacco.csv")
    print("Done.")

# ==============================================================
# PREPROCESSING: encounters
# ==============================================================
if preprocessing_context.config.is_enabled(ConfigKey.ENCOUNTERS):
    print()
    print("# ==============================================================")
    print("# Preprocessing encounters")
    print("# ==============================================================")
    print("Working...")
    df_encounters = preprocess_encounters(df_patient_windows, preprocessing_context)
    df_complete = df_complete.merge(df_encounters, how='left', left_index=True, right_index=True)
    df_complete[df_encounters.columns].fillna(0, inplace=True)
    df_complete.to_csv(f"{output_path}\\complete-encounters.csv")
    print("Done.")

# ==============================================================
# PREPROCESSING: (problem list) psychiatric comorbidities
# ==============================================================
if preprocessing_context.config.is_enabled(ConfigKey.PROBLEMS):
    print()
    print("# ==============================================================")
    print("# Preprocessing problem list")
    print("# ==============================================================")
    print("Working...")
    df_psychiatric_comorbidities = preprocess_problems(df_patient_windows, preprocessing_context)
    df_complete = df_complete.merge(df_psychiatric_comorbidities, how='left', left_index=True, right_index=True)
    df_complete[df_psychiatric_comorbidities.columns].fillna(0, inplace=True)
    df_complete.to_csv(f"{output_path}\\complete-problems.csv")
    print("Done.")

# ==============================================================
# PREPROCESSING: meds
# ==============================================================
if preprocessing_context.config.is_enabled(ConfigKey.MEDS):
    print()
    print("# ==============================================================")
    print("# Preprocessing meds")
    print("# ==============================================================")
    print("Working...")
    df_meds = preprocess_meds(df_patient_windows, preprocessing_context)
    df_complete = df_complete.merge(df_meds, how='left', left_index=True, right_index=True)
    df_complete[df_meds.columns].fillna(0, inplace=True)
    df_complete.to_csv(f"{output_path}\\complete-meds.csv")
    print("Done.")

# ==============================================================
# PREPROCESSING: labs
# ==============================================================
if preprocessing_context.config.is_enabled(ConfigKey.LABS):
    print()
    print("# ==============================================================")
    print("# Preprocessing labs")
    print("# ==============================================================")
    print("Working...")
    df_labs = preprocess_labs(df_patient_windows, preprocessing_context)
    df_complete = pd.merge(df_complete, df_labs, how='left', left_index=True, right_index=True)
    df_complete.to_csv(f"{output_path}\\complete-labs.csv")
    df_complete[df_labs.columns].fillna(0, inplace=True)
    print("Done.")

# ==============================================================
# PREPROCESSING: target
# ==============================================================
if preprocessing_context.config.is_enabled(ConfigKey.TARGET):
    print()
    print("# ==============================================================")
    print("# Preprocessing target (response) variables")
    print("# ==============================================================")
    print("Working...")
    df_target = preprocess_target_vars(df_patient_windows, preprocessing_context)
    df_complete = pd.merge(df_complete, df_target, how='left', left_index=True, right_index=True)
    df_complete.to_csv(f"{output_path}\\complete-targets.csv")
    print("Done.")

# we do general patient info last because we're going to end up just joining this
# to the giant data frame on PROJECT_PATIENT_ID, basically appending the general info
# for the patient to each row that represents a year of their data
# ==============================================================
# PREPROCESSING: general patient data
# ==============================================================
print()
print("# ==============================================================")
print("# Preprocessing patients")
print("# ==============================================================")
print("Working...")
df_patient_general = preprocess_patients(df_patients)
# reset the index of df_complete before we do this
# because patient_general has only a single index (PROJECT_PATIENT_ID) but complete's is a multiindex and weird stuff happens
df_complete = df_complete.reset_index()
df_complete = df_complete.merge(df_patient_general, how='inner', left_on='PROJECT_PATIENT_ID', right_index=True)
print("Done.")

print()
print("# ==============================================================")
print("####### FINAL CLEANUP ########")
print("# ==============================================================")
print("Filling null values...")
columns_with_nulls = df_complete.columns[df_complete.isna().any()].tolist()
df_complete[columns_with_nulls] = df_complete[columns_with_nulls].fillna(0)

print("Standardizing column names...")
column_names = df_complete.columns
column_names = [column.replace(' ', '_').lower() for column in column_names]
df_complete.columns = column_names

print("Resetting indices for file dump...")
df_complete.reset_index(inplace=True)
df_complete = df_complete.drop(columns='index')

print("Writing to file...")
final_file_path = f"{output_path}/preprocessed_data.csv"
df_complete.to_csv(final_file_path)
print(f"Preprocessing done! File saved to {final_file_path}.")
