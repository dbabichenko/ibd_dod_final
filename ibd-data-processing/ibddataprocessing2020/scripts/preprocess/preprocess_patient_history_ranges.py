import pandas as pd
from PreprocessConfig import PreprocessConfig, ConfigKey
from preprocess_utils import load_file


def preprocess_patient_history_ranges(patients_index, context):
    config = context.config
    assert isinstance(config, PreprocessConfig)

    print("Processing patient date data...")
    df_patient_history = pd.DataFrame(index=patients_index)
    df_patient_history.index.rename("PROJECT_PATIENT_ID", inplace=True)

    # NOTE: originally, i had this very complicated logic to determine each patient's start/end of treatment. that's what's commented out below.
    # in response to a request from claudia and dmitriy, i simplified this to be based on decision rules on encounters and contacts, which is
    # encapsulated in the method `get_date_ranges_from_discharges_and_encounters`.
    df_patient_history = get_date_ranges_from_discharges_and_encounters(df_patient_history, context)

    # however we got the values for df_patient_history (via original logic or the below method, return here)
    return df_patient_history


def get_date_ranges_from_complete_history(df_patient_history, context):
    config = context.config
    df_patient_history = df_patient_history if not config.is_enabled(ConfigKey.ENCOUNTERS) else merge_patient_df(
        df_patient_history, get_patient_min_max_dates("encounters_merged.xlsx", 'ENCOUNTER_DATE', date_column_prefix='ENC_'))
    df_patient_history = df_patient_history if not config.is_enabled(ConfigKey.LABS) else merge_patient_df(df_patient_history, get_patient_min_max_dates(
        context.files['labs'], 'RESULT_DATE', date_column_prefix='LAB_'))
    df_patient_history = df_patient_history if not config.is_enabled(ConfigKey.MEDS) else merge_patient_df(df_patient_history, get_patient_min_max_dates(
        "meds_merged.xlsx", 'ORDERING_DATE', date_column_prefix='MED_'))
    df_patient_history = df_patient_history if not config.is_enabled(ConfigKey.PROBLEMS) else merge_patient_df(df_patient_history, get_patient_min_max_dates(
        "problem_list_merged.xlsx", 'DX_DATE', date_column_prefix='PROB_'))
    df_patient_history = df_patient_history if not config.is_enabled(ConfigKey.PROCEDURES) else merge_patient_df(df_patient_history, get_patient_min_max_dates(
        "proc_merged.xlsx", 'ORDER_DATE', date_column_prefix='PROC_'))
    df_patient_history = df_patient_history if not config.is_enabled(ConfigKey.VITALS) else merge_patient_df(df_patient_history, get_patient_min_max_dates(
        "deid_IBD_Registry_BA1951_Vitals_BP_Ht_Wt.xlsx", 'CONTACT_DATE', date_column_prefix='VITAL_'))

    print("Calculating min and max dates for each patient...")

    # include date fields for all enabled components
    date_prefixes = []

    if config.is_enabled(ConfigKey.ENCOUNTERS):
        date_prefixes.append('ENC')
    if config.is_enabled(ConfigKey.LABS):
        date_prefixes.append('LAB')
    if config.is_enabled(ConfigKey.MEDS):
        date_prefixes.append('MED')
    if config.is_enabled(ConfigKey.PROBLEMS):
        date_prefixes.append('PROB')
    if config.is_enabled(ConfigKey.PROCEDURES):
        date_prefixes.append('PROC')
    if config.is_enabled(ConfigKey.VITALS):
        date_prefixes.append('VITAL')

    df_patient_history['GRAND_MIN_DATE'] = df_patient_history[[
        f'{prefix}_MIN_DATE' for prefix in date_prefixes]].min(axis=1)
    df_patient_history['GRAND_MAX_DATE'] = df_patient_history[[
        f'{prefix}_MAX_DATE' for prefix in date_prefixes]].min(axis=1)

    df_patient_history.index.rename("PROJECT_PATIENT_ID", inplace=True)
    df_patient_history = df_patient_history[["GRAND_MIN_DATE", "GRAND_MAX_DATE"]].reset_index()
    df_patient_history.set_index("PROJECT_PATIENT_ID", inplace=True)


def get_date_ranges_from_discharges_and_encounters(df_patients, context):
    df_targets = load_file(context.files['targets'])
    df_enc = load_file(context.files['encounters'])

    df_targets['CONTACT_DATE'] = pd.to_datetime(df_targets['CONTACT_DATE'])
    df_enc['ENCOUNTER_DATE'] = pd.to_datetime(df_enc['ENCOUNTER_DATE'])

    # first we need a min date for each patient - this strategy uses encounters only
    df_patients['GRAND_MIN_DATE'] = df_enc.groupby('PROJECT_PATIENT_ID')['ENCOUNTER_DATE'].min()

    # for each patient, we're now trying either their last ER/hospitalization date or
    # their last occurrence of a specific type of encounter as their max date

    # get the last ER/HOSP date for each patient
    df_patients['MAX_CONTACT_DATE'] = df_targets.groupby('PROJECT_PATIENT_ID')['CONTACT_DATE'].max()

    # for patients that have no such event, we have to use their last procedure of a certain type OR
    # their last phone, visit, or email from given departments
    depts = [
        'GAS HBC OAKLAND DDC PH',
        'GASTRO CLD HBC',
        'GASTRO GI LAB PUH',
        'GASTRO IBD MED HOME',
        'DW GASTRO SHDYSD OFC',
        'GASTRO MAGEE HBC',
        'GASTRO ASSOC MNRVL',
        'GASTRO ASSOC PH 110',
        'GASTRO ASSOC SHADY',
        'XGASTRO MAGEE HBC',
        'GASTRO HORIZON OFC',
        'GASTRO MONROE OFFICE',
        'GASTRO DDC',
        'UPP GASTRO ST MARGARET',
        'XGASTRO BUTLER HEP',
        'HBC GASTRO PSVNT MAIN',
        'UPP ASSOC IN GASTRO',
        'UPA GASTRO CHP INPT',
        'GASTRO MAGEE GI LAB',
        'GASTRO PUH',
        'GASTRO ASSOC BETHEL',
        'GASTRO ASSOC PH 300',
        'HORIZON GASTRO',
        'GASTRO ASSOC DHEC',
        'GAS CTR LIVER DIS HBC',
        'GASTRO ASSOC EAST HOSP',
        'GASTRO HORIZON GI/HEP',
        'GASTRO MCKEESPORT GEN',
        'MHC GASTRO',
        'XGASTRO ASSOC DGHRTY D',
        'GASTRO MONROE PROCED',
        'DW GAS SHADY SURGCTR',
        'GASTRO MCKEESPORT HBC',
        'GASTRO MWH GI LAB',
    ]

    df_enc = df_enc.query(
        "DEPT_NAME in @depts and (ENCOUNTER_TYPE in ['TELEPHONE', 'PATIENT EMAIL', 'Patient Message'] or ENCOUNTER_TYPE.str.contains('VISIT'))", engine='python')
    df_enc = df_enc.drop(columns=['ENCOUNTER_TYPE', 'DEPT_NAME', 'ICD9_CODE', 'ICD10_CODE', 'PRIMARY_DX'])
    df_patients['MAX_ENCOUNTER_DATE'] = df_enc.groupby('PROJECT_PATIENT_ID')['ENCOUNTER_DATE'].max()

    # if the patient has an ER/Hospitalization contact event, that's the end of their period
    # if they don't, their last encounter with a given department is the end of their period
    df_patients['GRAND_MAX_DATE'] = df_patients.apply(
        lambda x: x['MAX_CONTACT_DATE'] if x['MAX_CONTACT_DATE'] != pd.NaT else x['MAX_ENCOUNTER_DATE'], axis='columns')

    return df_patients[['GRAND_MAX_DATE', 'GRAND_MIN_DATE']]


def get_patient_min_max_dates(file_name, date_column, patient_id_column='PROJECT_PATIENT_ID', date_column_prefix=''):
    df = load_file(file_name)
    final_date_column = date_column

    # if the date column wasn't interpreted as a datetime, parse it using pandas
    if df[date_column].dtype == "object":
        final_date_column = f'${date_column}_to_date'
        df[final_date_column] = pd.to_datetime(df[date_column])

    df_min = df.groupby(patient_id_column)[final_date_column].min().reset_index()
    df_max = df.groupby(patient_id_column)[final_date_column].max().reset_index()
    df_min_max = pd.merge(df_min, df_max, how='inner', on=patient_id_column)
    df_min_max.columns = ['PROJECT_PATIENT_ID', f'{date_column_prefix}MIN_DATE', f'{date_column_prefix}MAX_DATE']
    df_min_max.set_index('PROJECT_PATIENT_ID', inplace=True)

    return df_min_max[[f'{date_column_prefix}MIN_DATE', f'{date_column_prefix}MAX_DATE']]


def merge_patient_df(df_patient_history, df_data):
    return pd.merge(df_patient_history, df_data, how='left', left_index=True, right_index=True)
