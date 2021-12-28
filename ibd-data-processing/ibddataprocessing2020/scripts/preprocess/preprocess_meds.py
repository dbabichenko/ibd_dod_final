from datetime import timedelta
from preprocess_utils import load_file, get_patient_treatment_year
import pandas as pd


def __encode_frequency(x):
    if x == 0:
        return 0
    if x < 3:
        return 1
    return 2


def __calculate_duration(x):
    if x["END_DATE"] is None:
        return x["window_train_breakpoint"] - x["ORDERING_DATE"]
    else:
        return x["END_DATE"] - x["ORDERING_DATE"]


def preprocess_meds(df_patient_window_dates, context):
    df_meds = load_file(context.files['meds'])
    df_meds.set_index('PROJECT_PATIENT_ID', inplace=True)

    # coerce data types
    df_meds["START_DATE"] = pd.to_datetime(df_meds["START_DATE"])
    df_meds["END_DATE"] = pd.to_datetime(df_meds["END_DATE"])
    df_meds["ORDERING_DATE"] = pd.to_datetime(df_meds["ORDERING_DATE"])

    # merge in patient dates
    df_meds = df_meds.merge(df_patient_window_dates.reset_index(), left_index=True, right_on='PROJECT_PATIENT_ID')

    # filter meds that weren't ordered in the window
    df_meds = df_meds.query("ORDERING_DATE >= window_start & ORDERING_DATE < window_train_breakpoint")

    # determine the duration of
    df_meds["PRESCRIPTION_DURATION"] = df_meds.apply(__calculate_duration, axis="columns")
    # drop extra columns
    df_meds = df_meds.drop(columns=df_patient_window_dates.columns)

    # create columns to flag
    df_meds["GROUP"] = df_meds["GROUP"].str.replace(" ", "_")
    df_meds = df_meds.append(pd.get_dummies(df_meds["GROUP"], prefix="MED"))
    df_meds['MED_5ASA'] = df_meds.apply(
        lambda x: 1 if x["GROUP"] == "5 ASA" else 0, axis='columns')
    df_meds['MED_AI'] = df_meds.apply(
        lambda x: 1 if x["GROUP"] == "ANTI INTEGRIN" else 0, axis='columns')
    df_meds['MED_AIL12'] = df_meds.apply(
        lambda x: 1 if x["GROUP"] == "ANTI IL12" else 0, axis='columns')
    df_meds['MED_ANTIBIOTICS'] = df_meds.apply(
        lambda x: 1 if x["GROUP"] == "Immunomodulators" else 0, axis='columns')
    df_meds['MED_ATNF'] = df_meds.apply(
        lambda x: 1 if x["GROUP"] == "ANTI TNF" else 0, axis='columns')
    df_meds['MED_ST'] = df_meds.apply(
        lambda x: 1 if x["GROUP"] == "Systemic steroids" else 0, axis='columns')
    df_meds['MED_VITD'] = df_meds.apply(
        lambda x: 1 if x["GROUP"] == "Vitamin D" else 0, axis='columns')

    # ditch group, we don't need it anymore
    df_meds = df_meds.drop(columns=['GROUP'])

    # aggregate
    df_return = df_meds.groupby(["PROJECT_PATIENT_ID", "window_id"]).agg(
        {
            'MED_5ASA': 'sum',
            'MED_AI': 'sum',
            'MED_AIL12': 'sum',
            'MED_ANTIBIOTICS': 'sum',
            'MED_ATNF': 'sum',
            'MED_ST': 'sum',
            'MED_VITD': 'sum',
        }, as_index=False
    ).reset_index()

    df_return.set_index(["PROJECT_PATIENT_ID", "window_id"], inplace=True)

    # encode
    # for training, we're categorizing these as 0 (never), 1 (transient), and 2 (persistent)
    df_return['MED_5ASA'] = df_return['MED_5ASA'].apply(__encode_frequency)
    df_return['MED_AI'] = df_return['MED_AI'].apply(__encode_frequency)
    df_return['MED_AIL12'] = df_return['MED_AIL12'].apply(__encode_frequency)
    df_return['MED_ANTIBIOTICS'] = df_return['MED_ANTIBIOTICS'].apply(
        __encode_frequency)
    df_return['MED_ATNF'] = df_return['MED_ATNF'].apply(__encode_frequency)
    df_return['MED_ST'] = df_return['MED_ST'].apply(__encode_frequency)
    df_return['MED_VITD'] = df_return['MED_VITD'].apply(__encode_frequency)

    return df_return
