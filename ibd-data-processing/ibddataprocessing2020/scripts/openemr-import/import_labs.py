import pandas as pd


def import_labs(file_name, patient_ids=None):
    df_labs = pd.read_excel(file_name)

    if patient_ids is not None and len(patient_ids) > 0:
        df_labs = df_labs.query('AUTO_ID == @patient_ids')

    # todo: everything
    return df_labs
