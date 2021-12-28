import numpy as np
import pandas as pd
from preprocess_utils import load_file
from PreprocessingWindowMode import PreprocessingWindowMode


def preprocess_target_vars(df_patient_windows, context):
    df_target = load_file(context.files['targets'])
    df_target.reset_index(inplace=True)

    # # using the same rule as registry that a patient can only have one hospitalization per day
    # df_enc_r3_hosp = df_enc_r3[df_enc_r3['ENC_TYPE'] == 'DISCHARGE SUMMARY'].groupby('STUDY_ID').agg({
    #     'START_DATE': 'count'})
    # # but any number of ER visits?
    # df_enc_r3_er = df_enc_r3[df_enc_r3['ENC_TYPE'] == 'ER REPORT'].groupby('STUDY_ID').agg({'VISIT_ID': 'count'})

    # print('patients with hospitalizations', len(df_enc_r3_hosp))
    # print('patients with ER visits', len(df_enc_r3_er))

    # merge patient window data
    df_target = df_target.merge(df_patient_windows.reset_index(), how='inner',
                                left_on="PROJECT_PATIENT_ID", right_on="PROJECT_PATIENT_ID")
    df_target.index.name = None

    # coerce data types
    df_target["CONTACT_DATE"] = pd.to_datetime(df_target["CONTACT_DATE"])

    date_query = None
    # ok, here's the deal:
    #
    # in one case, we're creating these windows with x years of training and y years of prediction (like training on 3 years and predicting target vars the 4th)
    if context.window_mode != PreprocessingWindowMode.ONE_WINDOW_PER_PATIENT_YEAR:
        # so that's this case right here - ditch data not in prediction part of the window
        date_query = "CONTACT_DATE >= window_train_breakpoint & CONTACT_DATE < window_end"
    else:
        # the other case creates one-year windows for every year of patient care and tries to predict the target
        # vars in the same year (a more classical ML formulation)
        date_query = "CONTACT_DATE >= window_start & CONTACT_DATE < window_end"
    df_target = df_target.query(date_query)

    # drop unnecessary columns and rename
    df_target = pd.DataFrame(df_target[["PROJECT_PATIENT_ID", "window_id", "IS_HOSPITALIZATION", "IS_ER_VISIT"]].copy())
    df_target.rename(columns={'IS_HOSPITALIZATION': 'TARGET_HOSPITALIZATIONS',
                              'IS_ER_VISIT': 'TARGET_ER_VISITS'}, inplace=True)

    # append pivoted group data
    df_target = df_target.groupby(["PROJECT_PATIENT_ID", "window_id"]).agg('sum')

    return df_target[['TARGET_HOSPITALIZATIONS', 'TARGET_ER_VISITS']]
