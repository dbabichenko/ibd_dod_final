import math
import numpy as np
import pandas as pd
from PreprocessingWindowMode import PreprocessingWindowMode


def preprocess_patient_windows(df_patients, preprocessing_context):
    df_windows = pd.DataFrame(columns=[
        'PROJECT_PATIENT_ID',
        'window_id',
        'window_start',
        'window_train_breakpoint',
        'window_end'])

    if preprocessing_context.window_mode in [PreprocessingWindowMode.FIXED_LENGTH, PreprocessingWindowMode.ONE_WINDOW_PER_PATIENT_YEAR]:
        df_windows = resolve_patient_fixed_windows(df_patients, df_windows, preprocessing_context)
    elif preprocessing_context.window_mode == PreprocessingWindowMode.ONE_WINDOW_PER_PATIENT:
        df_windows = resolve_one_window_per_patient(df_patients, df_windows, preprocessing_context)

    df_windows.set_index(['PROJECT_PATIENT_ID', 'window_id'], inplace=True)
    return df_windows


def resolve_patient_fixed_windows(df_patients, df_windows, preprocessing_context):
    for index, patient in df_patients.iterrows():
        thing = patient['GRAND_TIMESPAN'] - preprocessing_context.window_lengths['total']
        number_of_windows = math.floor(thing.days / preprocessing_context.window_lengths['offset'].days)

        for window_number in range(number_of_windows):
            window_end = patient['GRAND_MAX_DATE'] - (window_number * preprocessing_context.window_lengths['offset'])
            window_start = window_end - preprocessing_context.window_lengths['total']

            df_windows = df_windows.append({
                'PROJECT_PATIENT_ID': index,
                'window_id': window_number,
                'window_start': window_start,
                'window_train_breakpoint': window_start + preprocessing_context.window_lengths['train'],
                'window_end': window_end
            }, ignore_index=True)

    return df_windows


def resolve_one_window_per_patient(df_patients, df_windows, preprocessing_context):
    for index, patient in df_patients.iterrows():
        one_year = pd.Timedelta(365.25, 'D')
        num_years = math.floor(patient['GRAND_TIMESPAN'].days / one_year.days)
        patient_start_date = patient['GRAND_MAX_DATE'] - (one_year * num_years)
        patient_train_end_date = patient['GRAND_MAX_DATE'] - one_year
        patient_end_date = patient['GRAND_MAX_DATE']

        df_windows = df_windows.append({
            'PROJECT_PATIENT_ID': index,
            'window_id': 0,
            'window_start': patient_start_date,
            'window_train_breakpoint': patient_train_end_date,
            'window_end': patient_end_date
        }, ignore_index=True)

    return df_windows


# def resolve_one_window_per_patient_year(df_patients, df_windows, preprocessing_context):
#     return df_windows
