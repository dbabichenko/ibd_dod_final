from datetime import timedelta
import numpy as np
import pandas as pd
from preprocess_utils import load_file


# currently encoding any non-blank as abnormal (possibly values are HIGH, LOW, ABNORMAL)
def __encode_result(x):
    if x == '':
        return 1
    
    return 0


def __encode_frequency(x, lab_groups, years):
    for group in lab_groups:
        total = 0

        for year in years:
            total += x[f'{year}_{group}']

        # 0 years is never (0), 1-2 years is transient (1), 3 years is persistent (2)
        final_value = 0

        if total == 0:
            final_value = 0
        elif total < 3:
            final_value = 1
        else:
            final_value = 2

        x[f'LABS_{group.upper()}'] = final_value

    return x


def __compose_results_per_year(x):
    if x["RESULT_FLAG"] == 1:
        column_name = f'RESULT_YEAR_{int(x["RESULT_YEAR"])}_ABNORMAL_{x["GROUP"]}'
        x[column_name] = 1

    return x


def __get_patient_year(x):
    delta = x["RESULT_DATE"] - x["window_start"]
    return int(delta.days // 365.2425)


def preprocess_labs(df_patient_windows, context):
    df_labs = load_file(context.files['labs'])
    df_labs.set_index('PROJECT_PATIENT_ID', inplace=True)

    # coerce data types
    df_labs["RESULT_DATE"] = pd.to_datetime(df_labs["RESULT_DATE"])

    # merge patient data
    df_labs = df_labs.merge(df_patient_windows.reset_index(), left_index=True, right_on='PROJECT_PATIENT_ID')

    # filter results that aren't in the window
    df_labs = df_labs.query("RESULT_DATE >= window_start & RESULT_DATE < window_train_breakpoint")
    df_labs = df_labs[df_labs["GROUP"].notna()]

    # figure out which year a result is in for each patient
    df_labs['RESULT_YEAR'] = df_labs.apply(__get_patient_year, axis='columns')
    df_labs.drop(columns=['RESULT_DATE'], inplace=True)

    # encode result
    df_labs['RESULT_FLAG'] = df_labs['RESULT_FLAG'].apply(__encode_result)
    df_labs['RESULT_FLAG'].fillna(0, inplace=True)

    # drop unnecessary columns
    df_labs = df_labs.drop(columns=[
        'ORDER_PROC_ID',
        'COMPONENT_NAME',
        'ORD_VALUE',
        'ORD_NUM_VALUE',
        'REFERENCE_UNIT',
        'REFERENCE_LOW',
        'REFERENCE_HIGH',
        *df_patient_windows.columns
    ])

    # we have to compose a column for each combination of test type and year (e.g. a patient's max albumin result for year 2)
    lab_groups = pd.get_dummies(df_labs["GROUP"], prefix='ABNORMAL').columns
    years = pd.get_dummies(df_labs["RESULT_YEAR"], prefix='RESULT_YEAR').columns
    year_group_columns = []

    for group in lab_groups:
        for year in years:
            year_group_columns.append(f'{year}_{group}')

    df_new_columns = pd.DataFrame(columns=year_group_columns)
    df_labs = df_labs.append(df_new_columns)
    df_labs.fillna(0, inplace=True)

    # this does the magic
    df_labs = df_labs.apply(__compose_results_per_year, axis='columns')

    # we can drop the original columns now
    df_labs = df_labs.drop(columns=["GROUP", "RESULT_YEAR", "RESULT_FLAG"])

    # append pivoted group data
    df_labs.reset_index(inplace=True)
    df_labs.drop(columns=["index"], inplace=True)
    df_labs_grouped = df_labs.groupby(['PROJECT_PATIENT_ID', 'window_id']).agg('max')

    # we have to create our final columns in advance or else pandas throws up
    for group in lab_groups:
        df_labs_grouped[f'LABS_{group.upper()}'] = np.nan

    # for each type of test, we want to know if it was never positive, if it was positive in one or two years
    # or if it was positive in all three
    df_labs_grouped = df_labs_grouped.apply(lambda x:
                                            __encode_frequency(x, lab_groups, years), axis="columns")
    df_labs_grouped.drop(columns=year_group_columns, inplace=True)

    return df_labs_grouped
