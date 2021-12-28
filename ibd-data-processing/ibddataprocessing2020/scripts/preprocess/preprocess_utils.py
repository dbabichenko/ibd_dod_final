import os
import numpy as np
import pandas as pd

root_dir = os.path.join(os.getcwd(), 'ibddataprocessing2020\\scripts\\preprocess\\data')


def load_file(file_name):
    # todo: sniff for excel?
    df_new = pd.read_csv(os.path.join(
        root_dir, file_name))

    if "Unnamed: 0" in df_new.columns:
        df_new = df_new.drop(columns="Unnamed: 0")

    print(f"Loaded {df_new.shape[0]} records from {file_name}.")

    return df_new


def get_patient_treatment_year(row, field_name):
    delta = row[field_name] - row["PATIENT_WINDOW_START"]
    return delta.days // 365.2425
