import pandas as pd


def import_problems(file_name, patient_ids=None):
    df_probs = pd.read_excel(file_name)

    if patient_ids is not None and len(patient_ids) > 0:
        df_probs = df_probs.query('AUTO_ID == @patient_ids').copy()

    df_probs['comments'] = df_probs.apply(lambda x: f'Patient presents with {x["DX_NAME"]} today.', axis=1)
    df_probs['diagnosis'] = df_probs.apply(lambda x: f"{x['DX_CODE_TYPE']}:{x['DX_CODE']}", axis=1)
    df_probs['type'] = 'medical_problem'
    df_probs['activity'] = 1
    df_probs = df_probs.drop(columns=['DX_CODE_TYPE', 'DX_CODE'])

    df_probs = df_probs.rename(columns={
        'AUTO_ID': 'pid',
        'DX_NAME': 'title',
        'DX_DATE': 'date'
    })

    return df_probs
