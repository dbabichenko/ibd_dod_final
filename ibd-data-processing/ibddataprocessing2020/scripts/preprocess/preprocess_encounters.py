from datetime import datetime, timedelta
import pandas as pd
from preprocess_utils import load_file


def __encode_encounter_counts(value):
    if value == 0:
        return 0
    if value < 3:
        return 1
    return 2


def preprocess_encounters(df_patient_windows, context):
    df_enc = load_file(context.files['encounters'])
    # only using a few cols right now
    df_enc = df_enc[["PROJECT_PATIENT_ID", "ENCOUNTER_TYPE", "ENCOUNTER_DATE"]].copy()
    df_enc['ENCOUNTER_DATE'] = pd.to_datetime(df_enc['ENCOUNTER_DATE'])
    # return value
    df_return = None

    # join patients
    df_enc = df_enc.merge(df_patient_windows.reset_index(), how='inner',
                          left_on='PROJECT_PATIENT_ID', right_on='PROJECT_PATIENT_ID')

    # categorize encounter types
    df_enc["is_contact"] = df_enc.apply(lambda x: x['ENCOUNTER_TYPE'] in [
        'LETTER',
        'LETTER (OUT)',
        'OP REPORT',
        'PATIENT EMAIL',
        'Patient Message',
        'SCAN',
        'TELEPHONE',
        'Telephone'
    ], axis=1)

    df_enc["is_office_visit"] = df_enc.apply(lambda x: x['ENCOUNTER_TYPE'] in [
        'APPOINTMENT',
        'Consult',
        'HISTORY',
        'IP CONSULT'
        'Office Visit',
        'OFFICE VISIT',
        'New Patient Visit',
        'NEW PATIENT VISIT',
        'Procedure Visit'
    ], axis=1)

    df_enc["is_outpatient_procedure"] = df_enc.apply(lambda x: x['ENCOUNTER_TYPE'] in [
        'BPA',
        'EKG',
        'GI',
    ], axis=1)

    df_enc = df_enc.query('ENCOUNTER_DATE >= window_start & ENCOUNTER_DATE < window_end')

    # group by patient and type
    df_return = df_enc.groupby(["PROJECT_PATIENT_ID", "window_id"]).agg('sum')

    # for the purposes of training, patients get either 0 ("never"), 1 ("normal"),
    # or 2 ("high") depending on the number of encounters of each type they had
    df_return['contact_encounter_frequency'] = df_return['is_contact'].apply(__encode_encounter_counts)
    df_return['office_encounter_frequency'] = df_return['is_office_visit'].apply(
        __encode_encounter_counts)
    df_return['outpatient_procedure_encounter_frequency'] = df_return['is_outpatient_procedure'].apply(
        __encode_encounter_counts)

    df_return.fillna(0, inplace=True)
    return df_return[['contact_encounter_frequency', 'office_encounter_frequency', 'outpatient_procedure_encounter_frequency']]
