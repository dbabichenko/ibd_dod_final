from datetime import datetime, timedelta
import pandas as pd
from preprocess_utils import load_file


def preprocess_problems(df_patient_windows, context):
    # load and filter
    df_problems = load_file(context.files['problems'])
    df_problems['DX_DATE'] = pd.to_datetime(df_problems['DX_DATE'])
    df_problems = df_problems.query((
        'DX_CODE_TYPE == "ICD10"'
        ' & DX_CODE.str.len() > 0'
        ' & DX_CODE.str.startswith("F")'
        ' & DX_CODE != "F17"'
    ))

    # drop extra cols
    df_problems = df_problems.drop(columns=['DX_CODE_TYPE', 'DX_CODE', 'DX_NAME'])

    # merge patients - check index
    df_problems = df_problems.merge(df_patient_windows.reset_index(), how='inner',
                                    left_on='PROJECT_PATIENT_ID', right_on='PROJECT_PATIENT_ID')

    # filter data that isn't in the window and aggregate
    df_problems = df_problems.query("DX_DATE >= window_start & DX_DATE < window_end")
    df_problems = df_problems.groupby('PROJECT_PATIENT_ID').agg('sum')
    df_problems['psych_comorbidity'] = 1

    df_problems.reset_index(inplace=True)
    df_problems.set_index(["PROJECT_PATIENT_ID", "window_id"], inplace=True)

    return df_problems[['psych_comorbidity']]
