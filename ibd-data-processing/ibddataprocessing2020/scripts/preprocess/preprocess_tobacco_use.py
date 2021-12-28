from datetime import datetime, timedelta
import pandas as pd
from preprocess_utils import load_file


def __aggregate_tobacco_user(series):
    if "Yes" in series:
        return "Yes"
    if "Quit" in series:
        return "Quit"
    return "No"


def preprocess_tobacco_use(df_patient_windows, context):
    df_social_hx = load_file(context.files['tobacco'])
    df_social_hx["TOBACCO_USER"] = df_social_hx["TOBACCO_USER"].fillna("")
    df_social_hx["CONTACT_DATE"] = pd.to_datetime(df_social_hx["CONTACT_DATE"])
    df_social_hx["SMOKING_QUIT_DATE"] = pd.to_datetime(df_social_hx["SMOKING_QUIT_DATE"])
    df_return = None

    # join patients
    df_social_hx = df_social_hx.merge(df_patient_windows.reset_index(), how='inner',
                                      left_on='PROJECT_PATIENT_ID', right_on='PROJECT_PATIENT_ID')

    # drop data for each patient that does not fall in the window
    df_social_hx = df_social_hx.query('CONTACT_DATE >= window_start & CONTACT_DATE < window_end')

    # aggregate
    # do we care about matching the contact date to the quit date?
    # https://github.com/pandas-dev/pandas/issues/32156
    df_return = df_social_hx.groupby(["PROJECT_PATIENT_ID", 'window_id']).agg(
        {
            'CONTACT_DATE': 'max',
            'SMOKING_QUIT_DATE': 'max',
            'TOBACCO_USER': __aggregate_tobacco_user
        }, as_index=False
    )

    # true if they say yes or if their quit date is in the window
    df_return["TOBACCO_USE"] = df_return.apply(
        lambda x: 1 if x["TOBACCO_USER"] == "Yes" or x["TOBACCO_USER"] == "QUIT" else 0, axis=1)

    return df_return[["TOBACCO_USE"]]
