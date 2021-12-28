import pandas as pd
import datetime


def import_allergies(file_name, patient_ids=None):
    df_allergies = pd.read_excel(file_name)

    if patient_ids is not None and len(patient_ids) > 0:
        df_allergies = df_allergies.query('AUTO_ID == @patient_ids')

    df_allergies = df_allergies.fillna('unassigned')
    df_allergies['SEVERITY'] = df_allergies['SEVERITY'].str.lower()
    df_allergies['REACTION'] = df_allergies['REACTION'].str.lower()

    df_allergies = df_allergies.rename(columns={
        'AUTO_ID': 'pid',
        'DESCRIPTION': 'title',
        'SEVERITY': 'severity_al',
        'REACTION': 'reaction'
    })

    df_allergies['activity'] = 1
    df_allergies['begdate'] = datetime.date.today()
    df_allergies['type'] = 'allergy'

    return df_allergies
