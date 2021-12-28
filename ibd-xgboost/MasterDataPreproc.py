# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:54:52 2019

@author: Suraj
"""
import os
import pandas as pd
import numpy as np
import pgeocode
import datetime

os.chdir('C:\\Users\\Suraj\\Documents\\PITT_MSIS\\IBD_Research')

NOW = datetime.datetime(2018,7,1)
NEVER = datetime.datetime(2099,12,31)
MASTER_PAT = pd.read_csv('rawdata\\deid_patient_master_2018.csv').drop(['Unnamed: 0', 'FYI_FLAG_NAME', 'RACE','ETHNIC_GROUP', 'PATIENT_STATUS',\
                        'DATE_FLAG_CREATED'], axis=1)



# =============================================================================
# PSYCH MEDICATION DATA
# =============================================================================
psych_med_prescrip = pd.read_csv('rawdata\\deid_IBD_Registry_BA1951_Medications_2018-07-05-09-47-15.csv', parse_dates=['ORDERING_DATE']).drop(\
                             ['Unnamed: 0', 'SIMPLE_GENERIC_NAME', 'START_DATE', 'END_DATE', \
                              'PHARM_CLASS_C', 'PHARM_CLASS', 'PHARM_SUBCLASS_C', 'PHARM_SUB_CLASS'], axis=1)
psych_med_prescrip = psych_med_prescrip[psych_med_prescrip.THERA_CLASS_C==80]
psych_med_prescrip.AUTO_ID.nunique() #1538
nb_psych_drugs = psych_med_prescrip.groupby('AUTO_ID').agg({'ORDER_ID':'count', 'ORDERING_DATE':['min','max']})
nb_psych_drugs.columns = ['counts', 'min_dt', 'max_dt']
nb_psych_drugs['PSYCH_DURATION_YEARS'] = nb_psych_drugs.apply(lambda row: max((row.max_dt-row.min_dt).days/365, 1), axis=1)
nb_psych_drugs['PSYCH_ANNUAL_AVG_DOSAGE'] = nb_psych_drugs.apply(lambda row: row.counts/row.PSYCH_DURATION_YEARS, axis=1)
nb_psych_drugs.drop(['counts', 'min_dt', 'max_dt'], axis=1, inplace=True)
nb_psych_drugs = nb_psych_drugs.fillna(0)
# =============================================================================
# END PSYCH MEDICATION DATA
# =============================================================================


# =============================================================================
# OTHER MEDICATION DATA
# =============================================================================
med_df = pd.read_csv('rawdata\\filtered_meds2019.csv', parse_dates=['ORDERING_DATE'])
#group by med group and patient and count distinct orders
patient_drugclass_grp = med_df.groupby(by=['AUTO_ID','GROUP'])
meds_agg = patient_drugclass_grp.agg({'ORDER_ID': 'count', 'ORDERING_DATE':['min','max']})
meds_agg.columns = ['counts', 'min_dt', 'max_dt']
meds_agg['DURATION_YEARS'] = meds_agg.apply(lambda row: max((row.max_dt-row.min_dt).days/365, 1), axis=1)
meds_agg['ANNUAL_AVG_DOSAGE'] = meds_agg.apply(lambda row: row.counts/row.DURATION_YEARS, axis=1)
meds_agg.drop(['counts', 'min_dt', 'max_dt'], axis=1, inplace=True)
meds_agg = meds_agg.unstack(fill_value=0)
meds_agg.columns = [f'{drug_class}_{attribute}' for (attribute, drug_class) in meds_agg.columns.values]
PATIENT_MEDICATION = pd.concat([nb_psych_drugs, meds_agg], axis=1).fillna(0)
# =============================================================================
# END OTHER MEDICATION DATA
# =============================================================================


# =============================================================================
# LABS - TBD identify number of std.dev away from normalcy. Requires standardization of data. Could possibly identify std.dev for each class of ranges but it's a slow and long ordeal.

# labs = pd.read_csv('filtered_labs.csv', parse_dates=['ORDER_DATE','RESULT_DATE']).drop(['Unnamed: 0', 'Unnamed: 0.1', \
#                   'PROC_CODE', 'PROC_NAME', 'ORDER_STATUS', 'CPT_CODE', 'LAB_COMP_ID'], axis=1)
# labs = labs.drop(labs.index[np.logical_or(labs.ORD_NUM_VALUE.isnull(), labs.ORD_NUM_VALUE>=9999999)])
# labs = labs.drop(labs.index[labs.REF_LOW.str.isalpha()==True])
# labs.REF_LOW = np.where(labs.REF_NORMAL_VALS.notnull(), labs.REF_NORMAL_VALS, labs.REF_LOW)
# 
# labs[['NEW_LOW','NEW_HIGH']] = labs[labs.REF_LOW.str.contains('-')==True]['REF_LOW'].str.split('-').apply(pd.Series, 1)
# labs['NEW_LOW'] = np.where(labs.REF_LOW.str.contains(r'[<=]'), labs.REF_LOW.str.replace(r'[<=]',''), labs.NEW_LOW)
# labs['NEW_HIGH'] = np.where(labs.REF_LOW.str.contains(r'[<=]'), labs.REF_LOW.str.replace(r'[<=]',''), labs.NEW_HIGH)
# labs['NEW_LOW'] = np.where(labs.NEW_LOW.isnull(), labs.REF_LOW, labs.NEW_LOW)
# labs.drop(labs[labs.REF_HIGH.str.contains('<')==True].index, inplace=True) 
# labs['NEW_HIGH'] = np.where(labs.NEW_HIGH.isnull(), labs.REF_HIGH, labs.NEW_HIGH)
# labs['NEW_HIGH'] = labs['NEW_HIGH'].astype(float)
# labs['NEW_LOW'] = labs['NEW_LOW'].astype(float)
# labs['REF_UNIT'] = labs['REF_UNIT'].str.lower()
# 
# 
# # INTERPOLATE MISSING RANGES FROM KNOWN DATA --- START
# #about 32k entries don't have a clinical range defined
# unknown_labs = labs[labs.REF_LOW.isnull()].drop(['NEW_HIGH','NEW_LOW'], axis=1)
# unknown_labs.groupby(['GROUP','REF_UNIT']).count() # purely diagnostic. get a sense of what we're dealing with
# 
# #find most popular ranges for given group and ref_unit
# grp_units_range_counts = labs.groupby(['REF_UNIT','GROUP', 'NEW_LOW', 'NEW_HIGH']).size().reset_index()
# idx = grp_units_range_counts.groupby(['REF_UNIT','GROUP'])[0].transform(max) == grp_units_range_counts[0]
# popular_units_ranges = grp_units_range_counts[idx]
# newly_known_labs1 = unknown_labs.merge(popular_units_ranges, how='left', on=['REF_UNIT', 'GROUP']).drop_duplicates().drop(0, axis=1)
# 
# # about 10k entries don't have ref_unit
# still_unknown = newly_known_labs1[newly_known_labs1.NEW_HIGH.isnull()].drop(['NEW_HIGH','NEW_LOW'], axis=1)
# grp_range_counts = labs.groupby(['GROUP', 'NEW_LOW', 'NEW_HIGH']).size().reset_index()
# idx = grp_range_counts.groupby(['GROUP'])[0].transform(max) == grp_range_counts[0]
# popular_ranges = grp_range_counts[idx]
# newly_known_labs2 = still_unknown.merge(popular_ranges, how='left', on='GROUP').drop_duplicates().drop(0, axis=1)
# 
# #merge both newly_known labs
# newly_known = pd.concat([newly_known_labs1[newly_known_labs1.NEW_HIGH.notnull()], newly_known_labs2])
# # INTERPOLATE MISSING RANGES FROM KNOWN DATA --- END
# 
# ALL_LABS = pd.concat([labs[labs.REF_LOW.notnull()],  newly_known])
# ALL_LABS['NEW_FLAG'] = np.nan
# ALL_LABS['NEW_FLAG'] = np.where(ALL_LABS.ORD_NUM_VALUE < ALL_LABS.NEW_LOW, -1, ALL_LABS.NEW_FLAG)
# ALL_LABS['NEW_FLAG'] = np.where(ALL_LABS.ORD_NUM_VALUE > ALL_LABS.NEW_HIGH, 1, ALL_LABS.NEW_FLAG) # -- About 15k more entries flagged
# ALL_LABS.to_csv('Indiv_Labs_Data.csv')
# 
# =============================================================================



# =============================================================================
# LAB DATA
# =============================================================================
#Treating each lab group according to meeting notes
labs_raw = pd.read_csv('rawdata\\Indiv_Labs_Data.csv', parse_dates=['ORDER_DATE'])[['AUTO_ID', 'ORDER_DATE', 'GROUP', 'NEW_FLAG']].drop_duplicates()
labs_raw['year'] = labs_raw.ORDER_DATE.dt.year
LABS = pd.DataFrame(index=labs_raw.AUTO_ID.unique().astype(int))
# SCORES range from[-1,1] and are the proportions of highs and lows on the total number of lab tests done
LABS['EOS_HIGH_SCORE'] = labs_raw[labs_raw.GROUP=='eos'].groupby('AUTO_ID').NEW_FLAG.sum()/labs_raw[labs_raw.GROUP=='eos'].groupby('AUTO_ID').ORDER_DATE.nunique()
LABS['MONO_HIGH_SCORE'] = labs_raw[labs_raw.GROUP=='monocytes'].groupby('AUTO_ID').NEW_FLAG.sum()/labs_raw[labs_raw.GROUP=='monocytes'].groupby('AUTO_ID').ORDER_DATE.nunique()
LABS['ALBUMIN_HIGH_SCORE'] = labs_raw[labs_raw.GROUP=='albumin'].groupby('AUTO_ID').NEW_FLAG.sum()/labs_raw[labs_raw.GROUP=='albumin'].groupby('AUTO_ID').ORDER_DATE.nunique()
LABS['HEMO_HIGH_SCORE'] = labs_raw[labs_raw.GROUP=='hemoglobin'].groupby('AUTO_ID').NEW_FLAG.sum()/labs_raw[labs_raw.GROUP=='hemoglobin'].groupby('AUTO_ID').ORDER_DATE.nunique()
LABS['ESR_HIGH_SCORE'] = labs_raw[labs_raw.GROUP=='esr'].groupby('AUTO_ID').NEW_FLAG.sum()/labs_raw[labs_raw.GROUP=='esr'].groupby('AUTO_ID').ORDER_DATE.nunique()
LABS['CRP_HIGH_SCORE'] = labs_raw[labs_raw.GROUP=='crp'].groupby('AUTO_ID').NEW_FLAG.sum()/labs_raw[labs_raw.GROUP=='crp'].groupby('AUTO_ID').ORDER_DATE.nunique()
LABS['VITD_HIGH_SCORE'] = labs_raw[labs_raw.GROUP=='vitamin_d'].groupby('AUTO_ID').NEW_FLAG.sum()/labs_raw[labs_raw.GROUP=='vitamin_d'].groupby('AUTO_ID').ORDER_DATE.nunique()


def get_calendar_transience(df, FLAG=1):
    if FLAG==1:
        #  1 if any high flags within the same calendar year, else 0
        df2=df.groupby(['AUTO_ID', 'year']).max().replace({-1:0}).fillna(0).reset_index(0)
    else:
        df2=df.groupby(['AUTO_ID', 'year']).min().replace({1:0}).fillna(0).reset_index(0)
    # identify flag change across years for a patient
    df2['cumsum'] = (df2.NEW_FLAG.diff(1)!=0).cumsum()
    df2_grp = df2.groupby(['AUTO_ID','cumsum'])
    # get calendar durations of 1 and 0 flags for each patient
    df3 = pd.DataFrame({'BeginDate' : df2_grp.ORDER_DATE.first(), \
                  'EndDate' : df2_grp.ORDER_DATE.last(),\
                  'NbTests' : df2_grp.size(), \
                  'HLFlag' : df2_grp.NEW_FLAG.first()})
    # We only need to see FLAG (high for albumin, esr. low for hemo)
    df4 = df3[df3.HLFlag==FLAG]
    df4['FlagDuration'] = (df4.EndDate-df4.BeginDate).dt.days/365
    # Boundaries based on https://docs.google.com/document/d/1xRDOteIWgyBaJTDR9oUXfqAgGJLnG3Ol8Hfbrs-9Ml4
    df4['ClinicalFlag'] = pd.cut(df4['FlagDuration'], bins=[-1, 0, 3, 100]).apply(lambda x:x.right).astype(float)
    
    df5 = df4.groupby('AUTO_ID')['ClinicalFlag'].max().replace({0:'Never', 3:'Transient', 100:'Persistent'})
    df5.index = df5.index.astype(int)
    return df5
# =============================================================================
# END LAB DATA
# =============================================================================



# =============================================================================
# DEMOGRAPHIC DATA
# =============================================================================
# club discrete values together
map_maritalstatuses={'Married':'Married', 
                     'Single':'Single', 
                     'Divorced':'Single', 
                     'Widowed':'Single', 
                     'Unknown':'Unknown', 
                     'Legally Separated':'Single', 
                     'Significant other':'Married'
                     }
MASTER_PAT.rename(columns={'MARITAL STATUS':'MARITAL_STATUS'}, inplace=True)
MASTER_PAT['MARITAL_STATUS'].replace(map_maritalstatuses,inplace=True)

# same with employment status
map_employment={'Full Time'             :'Employed', 
                'Not Employed'          :'NotEmployed', 
                'Student - Full Time'   :'Student', 
                'Retired'               :'NotEmployed',
                'Self Employed'         :'Employed', 
                'Unknown'               :'Unknown', 
                'Part Time'             :'Employed', 
                'Student - Part Time'   :'Student'
                }
MASTER_PAT['EMPLOYMENT_STATUS'].replace(map_employment,inplace=True)
MASTER_PAT = MASTER_PAT.join(pd.get_dummies(MASTER_PAT.GENDER.replace({'M':'Male','F':'Female'}))).drop('GENDER', axis=1)
MASTER_PAT['AGE'] = 2018 - MASTER_PAT.BIRTH_YEAR
del MASTER_PAT['BIRTH_YEAR']
# calc distance from upmc
dist = pgeocode.GeoDistance('us')
MASTER_PAT['DISTANCE_KM'] = dist.query_postal_code(list(MASTER_PAT.ZIP.str.slice(0,5)), ['15217' for x in range(len(MASTER_PAT.ZIP))])
del MASTER_PAT['ZIP']
# =============================================================================
# END DEMOGRAPHIC DATA
# =============================================================================




# =============================================================================
# SIBDQ SCORES
# =============================================================================
# WIP: MORE CLEAR REPRESENTATION OF SIBDQ SCORES
sibdq = pd.read_csv('rawdata\\deid_IBD_Registry_BA1951_SIBDQ_Pain_Questionnaires_2018-07-05-10-02-02.csv').drop(['Unnamed: 0', 'PAIN_TODAY', 'ENC_TYPE'], axis=1).dropna(how='any')
sibdq.AUTO_ID = pd.to_numeric(sibdq.AUTO_ID).astype(int)
#sibdq_ewma_scores = sibdq.groupby('AUTO_ID').apply(lambda x:x['SIBDQ_TOTAL_SCORE'].ewm(alpha=0.2).mean())
avg_sibdq_scores = sibdq.groupby('AUTO_ID')['SIBDQ_TOTAL_SCORE'].mean()
avg_sibdq_scores['AUTO_ID'] = avg_sibdq_scores.index
MASTER_PAT['AVG_SIBDQ_SCORE'] = avg_sibdq_scores
# =============================================================================
# END SIBDQ SCORES
# =============================================================================





# =============================================================================
# CHARGES DATA
# =============================================================================
#http://www.usinflationcalculator.com/inflation/current-inflation-rates/
#I took it from the December column as it is explained on the website (last 12 months inflation)
#year,inflation
inflationrates={2005:3.4,
                2006:2.5,
                2007:4.1,
                2008:0.1,
                2009:2.7,
                2010:1.5,
                2011:3.0,
                2012:1.7,
                2013:1.5,
                2014:0.8,
                2015:0.7,
                2016:2.1,
                2017:2.1 } #last update: last charges made in 2018, so need to take into account only till 2017
for (year,rate) in inflationrates.items():
    inflationrates[year]=rate*0.01+1

#calculate coefficients    
lastyear=np.max(list(inflationrates.keys()))
inflation_coeff={lastyear+1:1.0}
for year in range(lastyear, 2005, -1):
    inflation_coeff[year]=inflation_coeff[year+1]*inflationrates[year]

#charges
inpatient_charges = pd.read_csv("rawdata\\deid_IP_Charges_Aug_2018_12_6_18.csv",thousands=r',', parse_dates=['ADMISSION DATE', 'DISCHARGE DATE']).drop(\
                               'Unnamed: 0', axis=1)
inpatient_charges.drop(inpatient_charges.index[inpatient_charges.AUTO_ID=='NO_MATCH'], axis=0, inplace=True)
inpatient_charges.index=inpatient_charges.index.astype(int)

#adjust by inflation
inpatient_charges['INPATIENT_CHARGES']=inpatient_charges.apply(lambda row: row['TOTAL CHARGES']*inflation_coeff[ row['DISCHARGE DATE'].year ], axis=1)
inpatient_charges.AUTO_ID = inpatient_charges.AUTO_ID.astype(int)
ip_charges = inpatient_charges.groupby('AUTO_ID').agg('sum').filter(items=['INPATIENT_CHARGES'])
ip_charges['IP_FIRST_DT'] = inpatient_charges.groupby('AUTO_ID').agg('min').filter(items=['ADMISSION DATE'])


outpatient_charges_transactions = pd.read_csv("rawdata\\deid_OP_charges_2018.csv", parse_dates=['ORIG_SERVICE_DATE'])
outpatient_charges_transactions['AMOUNT']=outpatient_charges_transactions.apply(lambda row: row['AMOUNT']*inflation_coeff[ row['ORIG_SERVICE_DATE'].year ], axis=1) #adjust by inflation
outpatient_charges=outpatient_charges_transactions[outpatient_charges_transactions['AMOUNT']>0]
outpatient_charges.drop(outpatient_charges[outpatient_charges.AUTO_ID=='NO_MATCH'].index, inplace=True)
outpatient_charges.AUTO_ID=outpatient_charges.AUTO_ID.astype(int)
op_charges = outpatient_charges.groupby('AUTO_ID').agg({'AMOUNT':'sum', 'ORIG_SERVICE_DATE':'min'})
op_charges.columns = ['OUTPATIENT_CHARGES', 'OP_FIRST_DT']

CHARGES = pd.merge(ip_charges, op_charges, how='outer', on='AUTO_ID') #None, left_on=ip_charges.index, right_on=op_charges.index)
CHARGES['TOTAL'] = CHARGES.INPATIENT_CHARGES.replace(np.nan, 0) + CHARGES.OUTPATIENT_CHARGES.replace(np.nan, 0)
CHARGES['FIRST_CHARGES_DT'] =  CHARGES[['IP_FIRST_DT','OP_FIRST_DT']].T.min()
CHARGES['CHARGE_DURATION_YRS'] = (pd.Series([NOW]*CHARGES.shape[0], index=CHARGES.index) - CHARGES['FIRST_CHARGES_DT']).dt.days/365
CHARGES['ANNUAL_AVG_CHARGE'] = CHARGES.TOTAL/CHARGES.CHARGE_DURATION_YRS
# =============================================================================
# END CHARGES DATA
# =============================================================================


#NOT SURE ABOUT USING THIS RN
# =============================================================================
# surgeries_df = outpatient_charges_transactions[outpatient_charges_transactions["TYPE_OF_SERVICE"]=='20-SURGERY MA'] 
# surgeries_df.drop_duplicates(subset=["AUTO_ID","ORIG_SERVICE_DATE"],inplace=True)
# surgeries_df.drop(columns=['Unnamed: 0', 'AMOUNT', 'DX','PROC_CODE', 'PROC_NAME', 'PROC_GROUP_NAME'],inplace=True).drop(surgeries_df[surgeries_df["AUTO_ID"]=="NO_MATCH"].index,inplace=True)
# surgeries_df.AUTO_ID=surgeries_df.AUTO_ID.astype(int)
# surgeries_df_agg = surgeries_df.groupby('AUTO_ID').agg({'AUTO_ID':'count', 'ORIG_SERVICE_DATE':'min'})
# surgeries_df_agg['timedelta_yrs'] = surgeries_df_agg.apply(lambda row: max((NOW-row.ORIG_SERVICE_DATE).days/365, 1), axis=1)
# surgeries_df_agg['SURGERIES_PER_YEAR']=surgeries_df_agg.AUTO_ID/surgeries_df_agg.timedelta_yrs
# surgeries_df_agg.drop(columns=['timedelta_yrs'],inplace=True)
# surgeries_df_agg.columns= ['SURGERY_COUNT', 'FIRST_SURGERY_DT', 'SURGERIES_PER_YEAR']
# =============================================================================


# =============================================================================
# ENCOUNTERS DATA
# =============================================================================
enc_df = pd.read_csv("rawdata\\deid_IBD_Registry_BA1951_Office_Phone_Email_Encs_2018-07-05-09-43-22.csv", parse_dates=['CONTACT_DATE']).drop('Unnamed: 0', axis=1)
enc_df.ENC_TYPE_NAME = enc_df.ENC_TYPE_NAME.replace({'Telephone':'TEL', 'Office Visit':'OFF', 'Patient Email':'TEL', 'Procedure Visit':'PROC', \
                                                     'New Patient Visit':'OFF', 'Consult':'CONSULT'})
ENC = enc_df.groupby(['AUTO_ID', 'ENC_TYPE_NAME']).agg({'CONTACT_DATE':['count','min', 'max']})
ENC.columns = ['nb_enc', 'min_dt', 'max_dt']
ENC['AVG_ANNUAL_ENC'] = ENC.apply(lambda row: row.nb_enc/max((row.max_dt-row.min_dt).days/365, 1), axis=1)
ENC.drop(['nb_enc', 'min_dt', 'max_dt'], axis=1, inplace=True)
ENC = ENC.unstack().replace(np.nan, 0) 
ENC.columns = ['_'.join(t) for t in ENC.columns.values]
# =============================================================================
# END ENCOUNTERS DATA
# =============================================================================



# =============================================================================
# MERGE ALL CONTINOUS DATA
# =============================================================================
MASTER_PAT = MASTER_PAT.merge(ENC, 'outer', 'AUTO_ID')
MASTER_PAT = MASTER_PAT.merge(LABS, 'outer', on=None, left_on='AUTO_ID', right_on=LABS.index)
MASTER_PAT = MASTER_PAT.merge(PATIENT_MEDICATION, 'outer', 'AUTO_ID')


MASTER_PAT = MASTER_PAT.merge(CHARGES[['CHARGE_DURATION_YRS', 'ANNUAL_AVG_CHARGE']], 'outer', 'AUTO_ID')
MASTER_PAT['CHARGE_FLAG_85pctile'] = np.where(MASTER_PAT['ANNUAL_AVG_CHARGE']<MASTER_PAT['ANNUAL_AVG_CHARGE'].quantile(0.85), 0, 1)
MASTER_PAT['CHARGE_FLAG_85pctile'] = np.where(MASTER_PAT['ANNUAL_AVG_CHARGE'].isnull(), np.nan, MASTER_PAT['CHARGE_FLAG_85pctile'])

MASTER_PAT.MARITAL_STATUS = MASTER_PAT.MARITAL_STATUS.replace({'Married':1, 'Single':0, 'Unknown':2})
MASTER_PAT.EMPLOYMENT_STATUS = MASTER_PAT.EMPLOYMENT_STATUS.replace({'Employed':1, 'NotEmployed':0, 'Student':2, 'Unknown':3, np.nan:3})


# =============================================================================
# start NEW CLINICAL SCHEME DISCRETIZED DATA
# =============================================================================
clin = MASTER_PAT[['AUTO_ID','PAT_ID', 'PAT_MRN_ID', 'MARITAL_STATUS', 'EMPLOYMENT_STATUS', 'CHARGE_FLAG_85pctile']]

clin.MARITAL_STATUS = clin.MARITAL_STATUS.replace({1:'Married', 0:'Single', 2:'Unknown'})
clin.EMPLOYMENT_STATUS = clin.EMPLOYMENT_STATUS.replace({1:'Employed', 0:'NotEmployed', 2:'Student', 3:'Unknown'})

clin['GENDER'] = np.where(MASTER_PAT.Female==1, 'Female', 'Male')

clin['AGE'] = pd.cut(MASTER_PAT['AGE'], bins=[18, 34, 50, 69, 100], labels=['Age1', 'Age2', 'Age3', 'Age4'])
clin['DISTANCE_KM'] = pd.cut(MASTER_PAT['DISTANCE_KM'], bins=[-1, 20, 80, 400, 7500], labels=['Dist1', 'Dist2', 'Dist3', 'Dist4'])
clin['SIBDQ'] = np.where(MASTER_PAT.AVG_SIBDQ_SCORE<50, 'Bad', 'Good')
clin['SIBDQ'] = np.where(MASTER_PAT.AVG_SIBDQ_SCORE.isnull(), np.nan, clin['SIBDQ'])

clin['ENC_CONSULT'] = pd.cut(MASTER_PAT['AVG_ANNUAL_ENC_CONSULT'], bins=[-1, 0, 2, 100], labels=['Never', 'Normal', 'High'])
clin['ENC_OFF'] = pd.cut(MASTER_PAT['AVG_ANNUAL_ENC_OFF'], bins=[-1, 0, 2, 100], labels=['Never', 'Normal', 'High'])
clin['ENC_PROC'] = pd.cut(MASTER_PAT['AVG_ANNUAL_ENC_PROC'], bins=[-1, 0, 1, 100], labels=['Never', 'Normal', 'High'])
clin['ENC_TEL'] = pd.cut(MASTER_PAT['AVG_ANNUAL_ENC_TEL'], bins=[-1, 0, 5, 100], labels=['Never', 'Normal', 'High'])

#MASTER_PAT['EOS_EVER_HIGH'] = labs_raw[labs_raw.GROUP=='eos'].groupby('AUTO_ID').NEW_FLAG.max().replace({-1:0}).fillna(0).replace({1:'Y',0:'N'})
clin['EOS_EVER_HIGH'] = labs_raw[labs_raw.GROUP=='eos'].fillna(0).groupby('AUTO_ID').NEW_FLAG.max().replace({-1:'N', 1:'Y',0:'N'})
#MASTER_PAT['MONO_EVER_HIGH'] = labs_raw[labs_raw.GROUP=='monocytes'].groupby('AUTO_ID').NEW_FLAG.max().replace({-1:0}).fillna(0).replace({1:'Y',0:'N'})
clin['MONO_EVER_HIGH'] = labs_raw[labs_raw.GROUP=='monocytes'].fillna(0).groupby('AUTO_ID').NEW_FLAG.max().replace({-1:'N', 1:'Y',0:'N'})

clin['ALBUMIN'] = get_calendar_transience(labs_raw[labs_raw.GROUP=='albumin']) # Not sure if this is to be -1?
clin['ALBUMIN'] = np.where(np.logical_and(clin.ALBUMIN.isnull(), clin.AUTO_ID.isin(labs_raw['AUTO_ID'].drop_duplicates())), 'Never', clin.ALBUMIN)
#MASTER_PAT['ALBUMIN'].fillna('Never', inplace=True)
clin['HEMO'] = get_calendar_transience(labs_raw[labs_raw.GROUP=='hemoglobin'], -1)
clin['HEMO'] = np.where(np.logical_and(clin.HEMO.isnull(), clin.AUTO_ID.isin(labs_raw['AUTO_ID'].drop_duplicates())), 'Never', clin.HEMO)
#MASTER_PAT['HEMO'].fillna('Never', inplace=True)
clin['ESR'] = get_calendar_transience(labs_raw[labs_raw.GROUP=='esr'])
clin['ESR'] = np.where(np.logical_and(clin.ESR.isnull(), clin.AUTO_ID.isin(labs_raw['AUTO_ID'].drop_duplicates())), 'Never', clin.ESR)
#MASTER_PAT['ESR'].fillna('Never', inplace=True)
clin['CRP'] = get_calendar_transience(labs_raw[labs_raw.GROUP=='crp'])
clin['CRP'] = np.where(np.logical_and(clin.CRP.isnull(), clin.AUTO_ID.isin(labs_raw['AUTO_ID'].drop_duplicates())), 'Never', clin.CRP)
#MASTER_PAT['CRP'].fillna('Never', inplace=True)
clin['VITD_LAB'] = np.where(MASTER_PAT.VITD_HIGH_SCORE<0, 'Low', 'High')
clin['VITD_LAB'] = np.where(MASTER_PAT.VITD_HIGH_SCORE.isnull(), np.nan, clin['VITD_LAB'])

medication_cols = ['PSYCH_ANNUAL_AVG_DOSAGE', '5_ASA_ANNUAL_AVG_DOSAGE', 'ANTIBIOTICS_ANNUAL_AVG_DOSAGE', 'ANTI_IL12_ANNUAL_AVG_DOSAGE', 'ANTI_INTEGRIN_ANNUAL_AVG_DOSAGE', 'ANTI_TNF_ANNUAL_AVG_DOSAGE', 'Immunomodulators_ANNUAL_AVG_DOSAGE', 'Systemic_steroids_ANNUAL_AVG_DOSAGE', 'Vitamin_D_ANNUAL_AVG_DOSAGE']
for c in medication_cols:
    clin[c.replace('_ANNUAL_AVG_DOSAGE', '')] = pd.cut(MASTER_PAT[c], bins=[-1, 0, 1, 50], labels=['Never', 'Transient', 'Persistent'])


# =============================================================================
# END NEW CLINICAL SCHEME DISCRETIZED DATA
# =============================================================================



# =============================================================================
# START PSYCH DIAGNOSIS DATA
# =============================================================================
psych_dx = pd.read_csv('rawdata\\deid_09-07-2018 resections-psych.csv').drop(['Unnamed: 0', \
                      'DATE_OF_ENTRY', 'NOTED_DATE', 'RESOLVED_DATE', 'UPDATE_DATE', 'PROBLEM_DESCRIPTION', 'ICD9_CODE', \
                      'ICD10_CODE', 'DIAGNOSIS_NAME', 'CLASS_OF_PROBLEM', 'PROBLEM_TYPE'], axis=1)
psych_dx = psych_dx[psych_dx.PROBLEM_STATUS=='ACTIVE']
del psych_dx['PROBLEM_STATUS']
psych_dx['PRIORITY'] = psych_dx['PRIORITY'].fillna('LOW')
psych_dx['PRIORITY'] = psych_dx['PRIORITY'].replace({'LOW':1, 'MEDIUM':2, 'HIGH':3})
psych_dx = psych_dx.drop_duplicates()

# =============================================================================
# END PSYCH DIAGNOSIS DATA
# =============================================================================



# =============================================================================
# START TOBACCO HABIT
# =============================================================================
tobacco = pd.read_csv('rawdata\\deid_IBD_Registry_BA1951_Social_Hx_2018-07-05-11-09-38.csv')
tobacco = tobacco[['AUTO_ID', 'TOBACCO_USER']].drop_duplicates().dropna()
tobacco.drop(tobacco.index[tobacco.AUTO_ID=='NO_MATCH'], axis=0, inplace=True)
tobacco['TOBACCO_USER'] = tobacco['TOBACCO_USER'].replace({'Never': 0, 'Quit': 1, 'Yes': 2, 'Passive': 1, 'Not Asked':0})
tobacco.AUTO_ID= tobacco.AUTO_ID.astype(float).astype(int)
tobacco_use = tobacco.groupby('AUTO_ID').TOBACCO_USER.max()
# =============================================================================
# END TOBACCO HABIT
# =============================================================================

clin['PSYCH_DX'] = psych_dx.groupby('AUTO_ID').PRIORITY.max().rename('PSYCH_DISEASES').replace({1:'LOW', 2:'MEDIUM', 3:'HIGH'}) #single 'HIGH' flag results gives the patient HIAGH (3)
MASTER_PAT['PSYCH_DX'] = psych_dx.groupby('AUTO_ID').PRIORITY.max().rename('PSYCH_DISEASES') 
MASTER_PAT['TOBACCO_USE'] = tobacco_use
clin['TOBACCO_USE'] = tobacco_use.replace({0:'Never', 1:'Quit', 2:'Yes'})

MASTER_PAT[['AUTO_ID', 'PAT_ID', 'PAT_MRN_ID', 'MARITAL_STATUS',
       'EMPLOYMENT_STATUS', 'Female', 'Male', 'AGE', 'DISTANCE_KM',
       'AVG_SIBDQ_SCORE', 'AVG_ANNUAL_ENC_CONSULT', 'AVG_ANNUAL_ENC_OFF',
       'AVG_ANNUAL_ENC_PROC', 'AVG_ANNUAL_ENC_TEL', 'EOS_HIGH_SCORE',
       'MONO_HIGH_SCORE', 'ALBUMIN_HIGH_SCORE', 'HEMO_HIGH_SCORE',
       'ESR_HIGH_SCORE', 'CRP_HIGH_SCORE', 'VITD_HIGH_SCORE',
       'PSYCH_DURATION_YEARS', 'PSYCH_ANNUAL_AVG_DOSAGE',
       '5_ASA_DURATION_YEARS', 'ANTIBIOTICS_DURATION_YEARS',
       'ANTI_IL12_DURATION_YEARS', 'ANTI_INTEGRIN_DURATION_YEARS',
       'ANTI_TNF_DURATION_YEARS', 'Immunomodulators_DURATION_YEARS',
       'Systemic_steroids_DURATION_YEARS', 'Vitamin_D_DURATION_YEARS',
       '5_ASA_ANNUAL_AVG_DOSAGE', 'ANTIBIOTICS_ANNUAL_AVG_DOSAGE',
       'ANTI_IL12_ANNUAL_AVG_DOSAGE', 'ANTI_INTEGRIN_ANNUAL_AVG_DOSAGE',
       'ANTI_TNF_ANNUAL_AVG_DOSAGE', 'Immunomodulators_ANNUAL_AVG_DOSAGE',
       'Systemic_steroids_ANNUAL_AVG_DOSAGE', 'Vitamin_D_ANNUAL_AVG_DOSAGE', 'PSYCH_DX', 'TOBACCO_USE',
       'CHARGE_DURATION_YRS', 'ANNUAL_AVG_CHARGE', 'CHARGE_FLAG_85pctile']].to_csv('ContinuousTrainData.csv', index=False)

clin[['AUTO_ID', 'PAT_ID', 'PAT_MRN_ID', 'MARITAL_STATUS',
       'EMPLOYMENT_STATUS', 'GENDER', 'AGE', 'DISTANCE_KM', 'SIBDQ',
       'ENC_CONSULT', 'ENC_OFF', 'ENC_PROC', 'ENC_TEL', 'EOS_EVER_HIGH',
       'MONO_EVER_HIGH', 'ALBUMIN', 'HEMO', 'ESR', 'CRP', 'VITD_LAB', 'PSYCH',
       '5_ASA', 'ANTI_IL12', 'ANTI_INTEGRIN', 'ANTI_TNF', 'Immunomodulators',
       'Systemic_steroids', 'Vitamin_D', 'ANTIBIOTICS', 'PSYCH_DX', 'TOBACCO_USE', 'CHARGE_FLAG_85pctile']].to_csv('DiscreteTrainData.csv', index=False)