import pandas as pd


def import_meds(file_name, patient_ids=None):
    df_meds = pd.read_excel(file_name)

    if patient_ids is not None and len(patient_ids) > 0:
        df_meds = df_meds.query('AUTO_ID == @patient_ids')

    df_meds = df_meds[[
        'AUTO_ID',
        'MED_NAME',
        'SIMPLE_GENERIC_NAME',
        'ORDERING_DATE',
        'PHARM_CLASS',
        'START_DATE',
        'END_DATE'
    ]]

    df_meds['comments'] = df_meds.apply(
        lambda x: f"Prescribed a {x['SIMPLE_GENERIC_NAME']} in class {x['PHARM_CLASS']}", axis=1)
    df_meds = df_meds.drop(columns=['SIMPLE_GENERIC_NAME', 'PHARM_CLASS'])
    df_meds = df_meds.rename(columns={
        'ORDERING_DATE': 'date',
        'MED_NAME': 'title',
        'AUTO_ID': 'pid',
        'START_DATE': 'begdate',
        'END_DATE': 'enddate',
    })
    df_meds['activity'] = 1
    df_meds['type'] = 'medication'

    return df_meds
