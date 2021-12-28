import os
import pandas as pd


def import_encounters(file_name, patient_ids=None):
    df_encounters = pd.read_excel(file_name)
    df_encounters = df_encounters[['ENCOUTNER_DATE', 'DEPT_NAME',
                                   'ENCOUNTER_TYPE', 'AUTO_ID']]

    if patient_ids is not None:
        df_encounters = df_encounters.query('AUTO_ID == @patient_ids')

    df_encounters = df_encounters.rename(columns={'ENCOUTNER_DATE': 'date', 'AUTO_ID': 'pid'})
    df_encounters['reason'] = df_encounters.apply(lambda x: f"{x['ENCOUNTER_TYPE']} by way of {x['DEPT_NAME']}", axis=1)
    df_encounters['facility'] = 'UPMC Oakland'
    df_encounters['billing_facility'] = 3
    df_encounters['facility_id'] = 3
    df_encounters['sensitivity'] = 'normal'
    df_encounters['pc_catid'] = 13
    df_encounters['provider_id'] = 1
    df_encounters = df_encounters.drop(columns=['DEPT_NAME', 'ENCOUNTER_TYPE'])

    return df_encounters
