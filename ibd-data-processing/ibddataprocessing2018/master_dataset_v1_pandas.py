import csv, os
import ast
import pandas as pd
import math
import datetime as dt
import locale
import sys
import datetime
import numpy as np
os.system('python zipcode/zipcode_distance.py')

#root_folder = "/Users/dmitriyb/Box Sync/Research/IBD Research/Inflammatory Bowel Disease DOD Grant/data/De-identified with linkers 11-24-2017/"
#root_folder = "C:/Users/Marcin/Documents/studies/IBD/data/De-identified with linkers 11-24-2017/"
root_folder ="/home/marcin/work/IBD/InflammatoryBowelDiseaseDODGrant/data/Deidentifiedwithlinkers11242017/"

output_folder =""
#output_folder ="/home/marcin/work/IBD/"
#output_folder ="C:/Users/Marcin/Documents/studies/IBD/data/output/"

# some of the statistics are calculteed til "now" -- it is assumed that it maybe 
# the day we create datafile. Although, we can set it up for the date of the last 
# data retrieval from the system database.
#
# At the moment it does not filter the data in terms of last possible date, which 
# can be implemented later
#now = datetime.datetime.now()
now = datetime.date(2017,9,1) #Sept 2017
blankdate=now-dt.timedelta(1)
blankdate=dt.date(blankdate.year, blankdate.month, blankdate.day)


def calcAnnualAvg(df, col_result, col_val, col_ft ) :
    df[col_result]=df[col_val]/((now-df[col_ft])/dt.timedelta(365))



print("--== Constructing dataset ==--")

#_____________________________________________________________________________________________
# Demographic info
print("Loading demographic information about patients...")

demographic_info_file = "deid_IBD_Registry_Participant_List_2017-01-19.csv"

# Columns:
demographic_columns = {'AUTO_ID' : 0,
                       'GENDER' : 1,
                       'BIRTH_YEAR' : 2,
                       'MARITAL STATUS' : 3,
                       'PATIENT_STATUS' : 4,
                       'RACE' : 5,
                       'ETHNIC_GROUP' : 6,
                       'EMPLOYMENT_STATUS' : 7,
                       'FYI_FLAG_NAME' : 8,
                       'ZIP' : 9,
                       'DATE_FLAG_CREATED' : 10
                      }

output_columns = ['AUTO_ID','GENDER','BIRTH_YEAR','MARITAL_STATUS','PATIENT_STATUS','RACE',
                  'ETHNIC_GROUP','EMPLOYMENT_STATUS','FYI_FLAG_NAME','ZIP','DATE_FLAG_CREATED']


patients_df = pd.read_csv(root_folder + demographic_info_file, parse_dates=['DATE_FLAG_CREATED'])
patients_df.set_index('AUTO_ID')

print("\t- Discretizing (AGE)...\n"),
#age is based on the current year and the year of birth
year=now.year
patients_df['AGE']=patients_df['BIRTH_YEAR'].map(lambda x: year-x)
#discretize
#there are no patients below 18 -- added just in case
patients_df['AGE']=pd.cut(patients_df['AGE'], bins=[-0.01,17.9,34.9,50.9,69.9,200], labels=['age18less','age18_34','age35_50','age51_69','age70above'])

print("\t- Merging states (RACE,MARITAL_STATUS,EMPLOYMENT_STATUS)...\n"),
map_races={'White':'White', 
           'Not Specified':'Other', 
           'Black':'Black', 
           'Vietnamese':'Other', 
           'Other Asian':'Other',
           'Indian (Asian)':'Other', 
           'Declined':'Other', 
           'Alaska Native':'Other', 
           'American Indian':'Other',
           'Japanese':'Other', 
           'Other Pacific Islander':'Other', 
           'Chinese':'Other'
           }
patients_df['RACE'].replace(map_races, inplace=True)
patients_df['RACE'].fillna('Other', inplace=True)

map_maritalstatuses={'Married':'Married', 
                     'Single':'Single', 
                     'Divorced':'Single', 
                     'Widowed':'Single', 
                     'Unknown':'Unknown', 
                     'Legally Separated':'Single', 
                     'Significant other':'Married'
                     }
patients_df.rename(columns={'MARITAL STATUS':'MARITAL_STATUS'}, inplace=True)
patients_df['MARITAL_STATUS'].replace(map_maritalstatuses,inplace=True)
patients_df['MARITAL_STATUS'].fillna('Unknown', inplace=True)

map_employment={'Full Time'             :'Employed', 
                'Not Employed'          :'NotEmployed', 
                'Student - Full Time'   :'Student', 
                'Retired'               :'NotEmployed',
                'Self Employed'         :'Employed', 
                'Unknown'               :'Unknown', 
                'Part Time'             :'Employed', 
                'Student - Part Time'   :'Student'
                }
patients_df['EMPLOYMENT_STATUS'].replace(map_employment,inplace=True)
patients_df['EMPLOYMENT_STATUS'].fillna('Unknown', inplace=True)


print("\t- Mapping ZIP codes...\n"),
#patients_df['DISTANCE']=patients_df['ZIP']
#for i in range(0,2916):
#    if len(patients_df['ZIP'][i])<5:
#        if select_zipcode('0'+patients_df['ZIP'][i][:4])==False:
#            patients_df['DISTANCE'][i]=99999
#        else:
#            patients_df['DISTANCE'][i]=distance('0'+patients_df['ZIP'][i][:4],'15213')
#    else:
#        if select_zipcode('0'+patients_df['ZIP'][i][:5]):
#            patients_df['DISTANCE'][i]=99999
#        else:
#            patients_df['DISTANCE'][i]=distance(patients_df['ZIP'][i][:5],'15213')

median_zip_file = "MedianZIP.csv"
median_zip_df= pd.read_csv(root_folder + median_zip_file)
median_zip_df.set_index('Zip',inplace=True)

def calcDistFromOakland(row):
    if len(row['ZIP'])<5:
        if select_zipcode('0'+row['ZIP'][:4])==False:
            return 9999
        else:
            return distance('0'+row['ZIP'][:4],'15213')
    else:
        if select_zipcode(row['ZIP'][:5])==False:
            return 9999
        else:
            return distance(row['ZIP'][:5],'15213')

def getMedian(row):
    if row['ZIP'][0]=='H':
        return 25000
    if len(row['ZIP'])<5:
        if int(row['ZIP'][:4]) not in median_zip_df.index:
            return None
        else:
            return median_zip_df.at[int(row['ZIP'][:4]),'Median']
    else:
        if int(row['ZIP'][:5]) not in median_zip_df.index:
            return None
        else:
            return median_zip_df.at[int(row['ZIP'][:5]),'Median']

patients_df['DISTANCE_VAL']= patients_df.apply(lambda row: calcDistFromOakland(row), axis=1)
patients_df['DISTANCE']=pd.cut(patients_df['DISTANCE_VAL'], bins=[-0.01,12.5,49.9,1000000], labels=['short','intermediate','long'])
patients_df['MEDIAN_INCOME_VAL']= patients_df.apply(lambda row: getMedian(row), axis=1)
patients_df['MEDIAN_INCOME']=pd.cut(patients_df['MEDIAN_INCOME_VAL'], bins=[-0.01,24000,36000,100000000], labels=['low','medium','high'])
#there is a problem for these zip codes
#54:8618
#274:7076
#323:8648
#428:6473
#621:7095
#1335:8096
#1566:H4R 1P7
#2097:8226
#2451:1730
#2636:8012
#if adding 0 at the beginning does not help -> just put 99999



print("\t\t -- DONE!\nLoading information about psychiatric comordibities...")


#_____________________________________________________________________________________________
# Psychiatric Comorbidities
dx_info_file = "deid_IBD_Reg_Pts_w_Problem_List_2017-01-19.csv"


def lookup_dx(dx_list, icd_code):
    for dx in dx_list:
        if dx in icd_code:
            return True
    return False


output_columns.append("PSYCH")

# ICD 9 Codes:
# Schizophrenia: 295.9
# Schizoaffective: 295.70
# Bipolar: 296.*,
# Depression: 296.3
# Anxiety: 300.02
# Substance abuse: 304.*
# Alcoholism: 303.*
# Opioid abuse
# OCD: 300.3
# PTSD: 309.81
# Tobacco: 305.1
psych_icd9 = ['295.', '296.', '300.02', '304.', '303.', '300.3', '309.81', '305.1'] # -- incomplete list
#or all F[0-9][0-9] excluding F17 ICD10 codes
psych_icd10_regexp='F(?!17)[0-9][0-9]\.'


# Columns
dx_columns = {'AUTO_ID': 0,
              'DATE_OF_ENTRY': 1,
              'NOTED_DATE': 2,
              'RESOLVED_DATE': 3,
              'ICD9_CODE': 4,
              'ICD10_CODE': 5,
              'DIAGNOSIS_NAME': 6}

    
psych_df= pd.read_csv(root_folder + dx_info_file, parse_dates=['DATE_OF_ENTRY','NOTED_DATE','RESOLVED_DATE'])
#drop rows with ICD9_CODE NaN and NA 
#psych_df=psych_df.dropna(subset=['ICD9_CODE'])
psych_df=psych_df.dropna(subset=['ICD10_CODE'])
#peek rows with specific disorder code
#it uses regular expression -- need to change dots into '\.'
#psych_df=psych_df[ psych_df['ICD9_CODE'].str.contains('|'.join(psych_icd9).replace('.','\.'))]
psych_df=psych_df[ psych_df['ICD10_CODE'].str.contains(psych_icd10_regexp)] 
#it may happen that there are multiple records that match different codes
psych_comobidity_patients=psych_df['AUTO_ID'].unique()


patients_df['PSYCH']=patients_df['AUTO_ID'].apply(lambda x: (x in psych_comobidity_patients))

print("\t\t -- DONE!\nLoading information about medications...")

#_____________________________________________________________________________________________
# Medications


med_info_file = "filtered_meds.csv"

# Columns:
med_columns = {'AUTO_ID': 0,
               'ORDER_ID': 1,
               'MED_ID': 2,
               'MED_NAME': 3,
               'SIMPLE_GENERIC_C': 4,
               'SIMPLE_GENERIC_NAME': 5,
               'ORDERING_DATE': 6,
               'THERA_CLASS_C': 7,
               'THERAPEUTIC_CLASS': 8,
               'PHARM_CLASS_C': 9,
               'PHARM_CLASS': 10,
               'PHARM_SUBCLASS_C': 11,
               'PHARM_SUB_CLASS': 12,
               'DESCRIPTION': 13,
               'SIG': 14,
               'GROUP': 15}

cnt = 0
med_dict = {}
med_cols_list = ["5_ASA", 
                 "Immunomodulators", 
                 "Systemic_steroids", 
                 "Vitamin_D", 
                 "ANTI_INTEGRIN", 
                 "ANTI_IL12",
                 "ANTI_TNF"]

sorted(med_cols_list)

output_columns = output_columns + med_cols_list

med_df= pd.read_csv(root_folder + med_info_file, parse_dates=['ORDERING_DATE'])
#group by med group and patient and count distinct orders
med_df['count']=med_df.groupby(by=['AUTO_ID','GROUP'])['ORDER_ID'].transform('count')
med_df['first']=med_df.groupby(by=['AUTO_ID','GROUP'])['ORDERING_DATE'].transform('min')
med_df['last']=med_df.groupby(by=['AUTO_ID','GROUP'])['ORDERING_DATE'].transform('max')


# probably could be done nicer, but as I added a column with counts we can drop duplicates
med_df_columns=med_df[['AUTO_ID','GROUP','count']].drop_duplicates().pivot(index="AUTO_ID",columns='GROUP',values='count')
#replace NaN values with zeros and drop NO_MATCH ID
med_df_columns.fillna(0, inplace=True)
#drop row with 'NO_MATCH'
med_df_columns=med_df_columns.drop("NO_MATCH")
#make a dataframe from pivot table
med_count_df=pd.DataFrame( med_df_columns.to_records())

#make the same pivot for dates and turn it into dataframe
med_df_first_pres=pd.DataFrame(med_df[['AUTO_ID','GROUP','first']].drop_duplicates().pivot(index="AUTO_ID",columns='GROUP',values='first').to_records())
#drop row with 'NO_MATCH'
med_df_first_pres=med_df_first_pres[med_df_first_pres['AUTO_ID']!='NO_MATCH']
med_cols_list_first=["first_"+colname for colname in med_cols_list]
med_df_first_pres.rename(columns=dict(zip(med_cols_list, med_cols_list_first)), inplace=True)
# THE SAME FOR LAST
#make the same pivot for dates and turn it into dataframe
med_df_last_pres=pd.DataFrame(med_df[['AUTO_ID','GROUP','last']].drop_duplicates().pivot(index="AUTO_ID",columns='GROUP',values='last').to_records())
#drop row with 'NO_MATCH'
med_df_last_pres=med_df_last_pres[med_df_last_pres['AUTO_ID']!='NO_MATCH']
med_cols_list_last=["last_"+colname for colname in med_cols_list]
med_df_last_pres.rename(columns=dict(zip(med_cols_list, med_cols_list_last)), inplace=True)


# as one of the AUTO_ID was of value 'NO_MATCH' we need to cast the whole 
# column into integers
med_count_df['AUTO_ID']=med_count_df['AUTO_ID'].astype('int')
med_df_first_pres['AUTO_ID']=med_df_first_pres['AUTO_ID'].astype('int')
med_df_last_pres['AUTO_ID']=med_df_last_pres['AUTO_ID'].astype('int')
med_count_df=med_count_df.set_index('AUTO_ID').join(med_df_first_pres.set_index('AUTO_ID'))
med_count_df=med_count_df.join(med_df_last_pres.set_index('AUTO_ID'))

for colname in med_cols_list:
    #calcAnnualAvg(med_count_df, 'avg_'+colname, colname, "first_"+colname)
    med_count_df['diff_'+colname]=(med_count_df['last_'+colname]-med_count_df['first_'+colname])/dt.timedelta(365)
    med_count_df['med_'+colname]=pd.cut(med_count_df['diff_'+colname], bins=[-0.1,0.01,1.01,9999], labels=['Never','Transient','Persistent'])
    med_count_df['med_'+colname]=np.where(med_count_df[colname]==0.0, 'Never',med_count_df['med_'+colname])

med_count_df.drop(med_cols_list_first+med_cols_list_last+med_cols_list+ ["diff_"+colname for colname in med_cols_list], axis=1, inplace=True)

#change columns names with prefix med_
med_new_col_names={}
for colname in med_cols_list:
    med_new_col_names['med_'+colname]=('med_'+colname).upper().replace(' ','_')
med_count_df.rename(columns=med_new_col_names, inplace=True)
#join dataframes
patients_df=patients_df.set_index('AUTO_ID').join(med_count_df)


print("\t\t -- DONE!\nLoading information about encounters...")


#_____________________________________________________________________________________________

# Encounters
encounters_info_file = "deid_IBD_Reg_Pts_w_Office_Phone_Email_2017-01-19.csv"

# Columns:
encounters_columns = {'AUTO_ID': 0,
                      'ENC_TYPE_C': 1,
                      'ENC_TYPE_NAME': 2,
                      'CONTACT_DATE': 3,
                      'DEPT_ID': 4,
                      'DEPT_NAME': 5,
                      'ICD9_CODE': 6,
                      'ICD10_CODE': 7,
                      'PRIMARY_DX': 8
                      }

enc_cols_list = ['Office Visit', 'New Patient Visit', 'Telephone', 'Consult', 'Patient Email', 'Procedure Visit']
enc_cols_list_merged = ['Office', 'Email or phone', 'Procedure Visit']

sorted(enc_cols_list)

output_columns = output_columns + enc_cols_list

enc_df= pd.read_csv(root_folder + encounters_info_file, parse_dates=['CONTACT_DATE'])
#group by encounter type and patient and count distinct dates
enc_df['count']=enc_df.groupby(by=['AUTO_ID','ENC_TYPE_NAME'])['CONTACT_DATE'].transform('count')
enc_df['firsttime']=enc_df.groupby(by=['AUTO_ID','ENC_TYPE_NAME'])['CONTACT_DATE'].transform('min')

# probably could be done nicer, but as I added a column with counts we can drop duplicates
enc_df_columns=enc_df[['AUTO_ID','ENC_TYPE_NAME','count']].drop_duplicates().pivot(index="AUTO_ID",columns='ENC_TYPE_NAME',values='count')
enc_df_columns_ft=enc_df[['AUTO_ID','ENC_TYPE_NAME','firsttime']].drop_duplicates().pivot(index="AUTO_ID",columns='ENC_TYPE_NAME',values='firsttime')
#replace NaN values with zeros
enc_df_columns.fillna(0, inplace=True)
enc_df_columns_ft.fillna(blankdate, inplace=True)
#make a dataframe from pivot table
enc_count_df=pd.DataFrame( enc_df_columns.to_records())
enc_firsttime_df=pd.DataFrame( enc_df_columns_ft.to_records())
enc_count_df.set_index('AUTO_ID', inplace=True)
enc_firsttime_df.set_index('AUTO_ID', inplace=True)
enc_new_col_names={}
for colname in enc_cols_list:
    enc_new_col_names[colname]=('ft_'+colname)
enc_firsttime_df.rename(columns=enc_new_col_names, inplace=True)

enc_count_df=enc_count_df.join(enc_firsttime_df)

#merge and calculate avgs
enc_count_df['Office']=enc_count_df['Office Visit']+enc_count_df['New Patient Visit']+enc_count_df['Consult']
enc_count_df['ft_Office']= enc_count_df.loc[:, ['ft_Office Visit', 'ft_New Patient Visit', 'ft_Consult']].min(axis=1)
enc_count_df['Email or phone']=enc_count_df['Patient Email']+enc_count_df['Telephone']
enc_count_df['ft_Email or phone']= enc_count_df.loc[:, ['ft_Patient Email', 'ft_Telephone']].min(axis=1)


for colname in enc_cols_list_merged:
    calcAnnualAvg(enc_count_df, 'enc_'+ colname, colname, 'ft_'+colname)
    
#drop unnecessary columns
enc_cols_list=enc_cols_list+['Email or phone','Office']
drop_col_list=enc_cols_list+['ft_'+col for col in enc_cols_list]
enc_count_df.drop(drop_col_list,axis=1,inplace=True)

#change columns names with prefix enc_
enc_new_col_names={}
for colname in enc_cols_list_merged:
    enc_new_col_names['enc_'+colname]=('enc_'+colname).upper().replace(' ','_')
enc_count_df.rename(columns=enc_new_col_names, inplace=True)

#discretize
enc_count_df['ENC_OFFICE']=pd.cut(enc_count_df['ENC_OFFICE'], bins=[-0.1,0.0000001,2.9999,9999], labels=['Never','Normal','High'])
enc_count_df['ENC_EMAIL_OR_PHONE']=pd.cut(enc_count_df['ENC_EMAIL_OR_PHONE'], bins=[-0.1,0.0000001,5.0000001,9999], labels=['Never','Normal','High'])
enc_count_df['ENC_PROCEDURE_VISIT']=pd.cut(enc_count_df['ENC_PROCEDURE_VISIT'], bins=[-0.1,0.0000001,1.9999,9999], labels=['Never','Normal','High'])





#join dataframes
patients_df=patients_df.join(enc_count_df)


print("\t\t -- DONE!\nLoading information about labs...\n"),



#_____________________________________________________________________________________________
# Labs
lab_info_file = "filtered_labs.csv"

# Columns:
lab_columns = {'AUTO_ID': 0,
                'ORDER_DATE': 1,
                'PROC_CODE': 2,
                'PROC_NAME': 3,
                'ORDER_STATUS': 4,
                'CPT_CODE': 5,
                'LAB_COMP_ID': 6,
                'LAB_COMP_NAME': 7,
                'RESULT_DATE': 8,
                'ORD_VALUE': 9,
                'ORD_NUM_VALUE': 10,
                'REF_LOW': 11,
                'REF_HIGH': 12,
                'REF_NORMAL_VALS': 13,
                'REF_UNIT': 14,
                'RESULT_FLAG': 15,
                'GROUP': 16
                }

cnt = 0


lab_col_list = ['monocytes', 'albumin', 'crp', 'eos', 'hemoglobin', 'esr', 'vitamin_d']
lab_dict = {}
labFlag_col_list = ['High', 'Low'] #others are Low Panic, Abnormal, (NONE)
labFlag_dict = {}

lab_df = pd.read_csv(root_folder + lab_info_file,parse_dates=['ORDER_DATE','RESULT_DATE'])
print("\t- Counting groups...\n"),

#group by lab GROUP and patient and count distinct orders
lab_df['count']=lab_df.groupby(by=['AUTO_ID','GROUP'])['ORDER_DATE'].transform('count')
# probably could be done nicer, but as I added a column with counts we can drop duplicates
lab_df_columns=lab_df[['AUTO_ID','GROUP','count']].drop_duplicates().pivot(index="AUTO_ID",columns='GROUP',values='count')
#replace NaN values with zeros
lab_df_columns.fillna(0, inplace=True)
#make a dataframe from pivot table
lab_count_df=pd.DataFrame( lab_df_columns.to_records() )
#change columns names with prefix lab_
lab_new_col_names={}
for colname in lab_col_list:
    lab_new_col_names[colname]=('lab_'+colname).upper().replace(' ','_')
lab_count_df.rename(columns=lab_new_col_names, inplace=True)
#change type of AUTO_ID to int to match patient main dataframe
lab_count_df['AUTO_ID']=lab_count_df['AUTO_ID'].astype('int')
#join dataframes
#patients_df=patients_df.join(lab_count_df.set_index('AUTO_ID'))

#################I can throw something out above.. need to clean

print("\t- Replacing High and Low result flags for Vitamin D...\n"),
#use vitamin_d <30 for low and >30 high instead of official flag
lab_df.loc[ lab_df.query('GROUP==\'vitamin_d\' & (ORD_NUM_VALUE<=30)').index, 
            'RESULT_FLAG']='Low' # 'High'
lab_df.loc[ ((lab_df['GROUP']=='vitamin_d') & (lab_df['ORD_NUM_VALUE']>30) & (lab_df['ORD_NUM_VALUE']<999999)), 'RESULT_FLAG']='High'
lab_df.loc[ np.isnan(lab_df['ORD_NUM_VALUE']), 'RESULT_FLAG']=np.nan


print("\t- Counting High and Low result flags...\n"),

#group by RESULT_FLAG and patient and count distinct orders
lab_df['countRF']=lab_df.groupby(by=['AUTO_ID','RESULT_FLAG'])['ORDER_DATE'].transform('count')
#select High and Low
lab_df_RFHL=lab_df[ lab_df['RESULT_FLAG']=='High' ].append(lab_df[ lab_df['RESULT_FLAG']=='Low' ],ignore_index=True)
# probably could be done nicer, but as I added a column with counts we can drop duplicates
lab_df_RFHL=lab_df_RFHL[['AUTO_ID','RESULT_FLAG','countRF']].drop_duplicates().pivot(index="AUTO_ID",columns='RESULT_FLAG',values='countRF')

#replace NaN values with zeros
lab_df_RFHL.fillna(0, inplace=True)
#make a dataframe from pivot table
lab_count_df=pd.DataFrame( lab_df_RFHL.to_records() )
#change columns names with prefix lab_result_
labFlag_new_col_names={}
for colname in labFlag_col_list:
    labFlag_new_col_names[colname]=('lab_result_'+colname).upper()
lab_count_df.rename(columns=labFlag_new_col_names, inplace=True)


lab_df['firstHigh']=lab_df[ lab_df['RESULT_FLAG']=='High' ].groupby(by=['AUTO_ID','GROUP'])['ORDER_DATE'].transform('min')
lab_df['lastHigh']=lab_df[ lab_df['RESULT_FLAG']=='High' ].groupby(by=['AUTO_ID','GROUP'])['ORDER_DATE'].transform('max')
lab_df['firstLow']=lab_df[ lab_df['RESULT_FLAG']=='Low' ].groupby(by=['AUTO_ID','GROUP'])['ORDER_DATE'].transform('min')
lab_df['lastLow']=lab_df[ lab_df['RESULT_FLAG']=='Low' ].groupby(by=['AUTO_ID','GROUP'])['ORDER_DATE'].transform('max')

#get highs and lows
lab_df['ishigh']=np.where(lab_df['RESULT_FLAG']=='High',1,0)
lab_df['islow']=np.where(lab_df['RESULT_FLAG']=='Low',1,0)


lab_df_group_high_flag= pd.DataFrame(
                         pd.pivot_table(lab_df[['AUTO_ID', 'GROUP', 'ishigh']], 
                         values='ishigh',
                         index=['AUTO_ID'],
                         columns=['GROUP'],
                         aggfunc=np.sum).to_records() 
                       )
lab_df_group_low_flag= pd.DataFrame(
                         pd.pivot_table(lab_df[['AUTO_ID', 'GROUP', 'islow']], 
                         values='islow',
                         index=['AUTO_ID'],
                         columns=['GROUP'],
                         aggfunc=np.sum).to_records() 
                       )
lab_df_high=lab_df.loc[lab_df['ishigh']==1,['AUTO_ID', 'GROUP', 'firstHigh']].drop_duplicates()
lab_df_group_first_high= pd.DataFrame(lab_df_high.pivot(index="AUTO_ID",columns='GROUP',values='firstHigh').to_records())
lab_df_group_first_high.set_index('AUTO_ID',inplace=True)
lab_df_high=lab_df.loc[lab_df['ishigh']==1,['AUTO_ID', 'GROUP', 'lastHigh']].drop_duplicates()
lab_df_group_last_high= pd.DataFrame(lab_df_high.pivot(index="AUTO_ID",columns='GROUP',values='lastHigh').to_records())
lab_df_group_last_high.set_index('AUTO_ID',inplace=True)
lab_df_group_high_flag.set_index('AUTO_ID',inplace=True)

lab_df_low=lab_df.loc[lab_df['islow']==1,['AUTO_ID', 'GROUP', 'firstLow']].drop_duplicates()
lab_df_group_first_low= pd.DataFrame(lab_df_low.pivot(index="AUTO_ID",columns='GROUP',values='firstLow').to_records())
lab_df_group_first_low.set_index('AUTO_ID',inplace=True)
lab_df_low=lab_df.loc[lab_df['islow']==1,['AUTO_ID', 'GROUP', 'lastLow']].drop_duplicates()
lab_df_group_last_low= pd.DataFrame(lab_df_low.pivot(index="AUTO_ID",columns='GROUP',values='lastLow').to_records())
lab_df_group_last_low.set_index('AUTO_ID',inplace=True)
lab_df_group_low_flag.set_index('AUTO_ID',inplace=True)

labs_to_process=['albumin', 'crp', 'hemoglobin', 'esr', 'vitamin_d']
labs_to_process_high=['crp', 'esr','monocytes', 'eos']
labs_to_process_low=['albumin', 'hemoglobin', 'vitamin_d']

labs_high_cols_to_remove=lab_df_high['GROUP'].unique()
labs_low_cols_to_remove=lab_df_low['GROUP'].unique()

#new_col_names={}
#for colname in labs_to_process:
#    new_col_names[colname]=('fth_'+colname)
#lab_df_group_first_high.rename(columns=new_col_names, inplace=True)
#
#new_col_names={}
#for colname in labs_to_process:
#    new_col_names[colname]=('lth_'+colname)
#lab_df_group_last_high.rename(columns=new_col_names, inplace=True)
#
#new_col_names={}
#for colname in labs_to_process:
#    new_col_names[colname]=('valh_'+colname)
#lab_df_group_high_flag.rename(columns=new_col_names, inplace=True)
#
#new_col_names={}
#for colname in labs_to_process:
#    new_col_names[colname]=('ftl_'+colname)
#lab_df_group_first_low.rename(columns=new_col_names, inplace=True)
#
#new_col_names={}
#for colname in labs_to_process:
#    new_col_names[colname]=('ltl_'+colname)
#lab_df_group_last_low.rename(columns=new_col_names, inplace=True)
#
#new_col_names={}
#for colname in labs_to_process:
#    new_col_names[colname]=('vall_'+colname)
#lab_df_group_low_flag.rename(columns=new_col_names, inplace=True)

# labs discretization
lab_count_df['LAB_EOS_EVER_HIGH']=np.where(lab_df_group_high_flag.eos[ lab_count_df['AUTO_ID'] ]>0,'YES','NO')
lab_count_df['LAB_MONOCYTES_EVER_HIGH']=np.where(lab_df_group_high_flag.monocytes[ lab_count_df['AUTO_ID'] ]>0,'YES','NO')

#lab_df_group_first_high.drop(['eos','monocytes'], axis=1,inplace=True)
#lab_df_group_last_high.drop(['eos','monocytes'], axis=1,inplace=True)
#lab_df_group_high_flag.drop(['eos','monocytes'], axis=1,inplace=True)
#lab_df_group_first_low.drop(['eos','monocytes'], axis=1,inplace=True)
#lab_df_group_last_low.drop(['eos','monocytes'], axis=1,inplace=True)
#lab_df_group_low_flag.drop(['eos','monocytes'], axis=1,inplace=True)

lab_count_df['AUTO_ID']=lab_count_df['AUTO_ID'].astype('int')
lab_count_df.set_index('AUTO_ID',inplace=True)
#DO THIS BETTER!
lab_count_df=lab_count_df.join(lab_df_group_first_high,rsuffix='_fth') #just to have doubles out of laziness and time spent on this part
##############
lab_count_df=lab_count_df.join(lab_df_group_first_high,rsuffix='_fth').join(lab_df_group_last_high,rsuffix='_lth').join(lab_df_group_high_flag,rsuffix='_valh')
lab_count_df=lab_count_df.join(lab_df_group_first_low,rsuffix='_ftl').join(lab_df_group_last_low,rsuffix='_ltl').join(lab_df_group_low_flag,rsuffix='_vall')

lab_count_df.drop(labs_high_cols_to_remove ,axis=1,inplace=True)#just to have doubles out of laziness and time spent on this part


labs_to_discretize=[]
for col in labs_to_process_high:
     lab_count_df['LAB_'+col.upper()]= (lab_count_df[col+'_lth']-lab_count_df[col+'_fth'])/dt.timedelta(365)
     labs_to_discretize+=['LAB_'+col.upper()]
     
for col in labs_to_process_low:
     lab_count_df['LAB_'+col.upper()]= (lab_count_df[col+'_ltl']-lab_count_df[col+'_ftl'])/dt.timedelta(365)
     labs_to_discretize+=['LAB_'+col.upper()]

lab_count_df.drop([col+'_valh' for col in labs_high_cols_to_remove]+[col+'_fth' for col in labs_high_cols_to_remove]+[col+'_lth' for col in labs_high_cols_to_remove]+
                  [col+'_vall' for col in labs_low_cols_to_remove]+[col+'_ftl' for col in labs_low_cols_to_remove]+[col+'_ltl' for col in labs_low_cols_to_remove]+
                  ['esr_vall'],#HOT FIX!
                    axis=1,inplace=True)
                    
           
#def calcAnnualAvg(df, col_result, col_val, col_ft ) 
for col in labs_to_discretize:
    lab_count_df[col]=pd.cut(lab_count_df[col], bins=[-0.1,0.0000001,2.9999,9999], labels=['Never','Transient','Persistent'])



#change type of AUTO_ID to int to match patient main dataframe
#join dataframes
patients_df=patients_df.join(lab_count_df)




print("\t\t -- DONE!\nLoading SIBDQ surveys...\n"),

#_____________________________________________________________________________________________
#SIBDQ

#      [u'AUTO_ID', 
#       u'CONTACT_DATE', 
#       u'VISIT_DEPT_NAME', 
#       u'COMPLETED_ON',
#       u'Q01_FEELING_FATIGUE', 
#       u'Q02_SOCIAL_ENGAGEMENT',
#       u'Q03_DIFFICULTY_ACTIVITIES',
#       u'Q04_TROUBLED_PAIN',
#       u'Q05_FELT_DEPRESSED', 
#       u'Q06_PROBLEM_PASSING',
#       u'Q07_MAINTAINING_WEIGHT', 
#       u'Q08_FELT_RELAXED', 
#       u'Q09_TROUBLED_EMPTY',
#       u'Q10_FELT_ANGRY', 
#       u'SIBDQ_TOTAL_SCORE', 
#       u'PAIN_TODAY',
#       u'HAS_MISSING_VALUES']



sibdq_info_file = "cleaned_deid_SIBDQ.csv"
print("\t- Calculating total average\n"),
sibdq = pd.read_csv(root_folder + sibdq_info_file, parse_dates=['CONTACT_DATE'])
#calculate total average for each patient
sibdq_avg=sibdq.groupby(['AUTO_ID']).agg('mean').filter(items=['SIBDQ_TOTAL_SCORE'])
#discretize
sibdq_avg['SIBDQ_TOTAL_SCORE']=pd.cut(sibdq_avg['SIBDQ_TOTAL_SCORE'], bins=[-0.01,49.999999999,70.1], labels=['bad','good'])
#join dataframes and name the column
patients_df=patients_df.join(sibdq_avg.rename(columns={'SIBDQ_TOTAL_SCORE':'SIBDQ_TOTAL_AVG'}))
patients_df['SIBDQ_TOTAL_AVG'].fillna('Missing',inplace=True)


#==============================================================================
# 
# #_____________________________________________________________________________________________
# 
# print("\t- Retrieving information about pain during last visit\n"),
# #pain at last visit
# 
# #get indices of last 'CONTACT_DATE''s
# sibdq_grouped_by_patient=sibdq.groupby(['AUTO_ID'], sort=False)
# sibdq_idx=sibdq_grouped_by_patient['CONTACT_DATE'].transform(max)==sibdq['CONTACT_DATE']
# #pick these rows
# sibdq_last_visits=sibdq[sibdq_idx].drop_duplicates(subset=['AUTO_ID'], keep='first')
# sibdq_last_visits=sibdq_last_visits[['AUTO_ID','PAIN_TODAY']].set_index('AUTO_ID')
# patients_df=patients_df.join(sibdq_last_visits)
# 
#==============================================================================

#_____________________________________________________________________________________________


Ques=[#'no',
      'Q01_FEELING_FATIGUE',    ##################### 0
 'Q02_SOCIAL_ENGAGEMENT',
 'Q03_DIFFICULTY_ACTIVITIES', 
 'Q04_TROUBLED_PAIN',           ##################### 3
 'Q05_FELT_DEPRESSED',          ##################### 4
 'Q06_PROBLEM_PASSING',
 'Q07_MAINTAINING_WEIGHT', 
 'Q08_FELT_RELAXED',            ##################### 7
 'Q09_TROUBLED_EMPTY', 
 'Q10_FELT_ANGRY', 
 'SIBDQ_TOTAL_SCORE',
 'HAS_MISSING_VALUES']
 
#ques_chosen=[0,3,4,7]
sibdq_list_of_ques=['Q01_FEELING_FATIGUE', 'Q04_TROUBLED_PAIN', 'Q05_FELT_DEPRESSED', 'Q08_FELT_RELAXED']
print("\t- Calculating mean over a year before the last visit of particular questions "+str(sibdq_list_of_ques) + "\n"),

sibdq['lastVisit']= sibdq.groupby('AUTO_ID')['CONTACT_DATE'].transform(lambda x: x.max())
sibdq_last_year=sibdq[ sibdq['lastVisit'] - sibdq['CONTACT_DATE']< dt.timedelta(365)]
sibdq_ly_means=sibdq_last_year.groupby('AUTO_ID').aggregate('mean')
sibdq_ly_means=sibdq_ly_means[sibdq_list_of_ques]

for colname in sibdq_list_of_ques:
    sibdq_ly_means[colname]=np.where(sibdq_ly_means[colname]<5, 'Bad', 'Good')

sibdq_new_col_names={}
for colname in sibdq_list_of_ques:
    sibdq_new_col_names[colname]=('SIBDQ_AVG_LY_'+colname)
sibdq_ly_means.rename(columns=sibdq_new_col_names, inplace=True)
patients_df=patients_df.join(sibdq_ly_means)

print("\t\t -- DONE!\nLoading inpatient charges...\n"),

#http://www.usinflationcalculator.com/inflation/current-inflation-rates/
#I took it from the December column as it is explained on the website (12 months inflation)
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
                2017:2.1 }
for (year,rate) in inflationrates.items():
    inflationrates[year]=rate*0.01+1

lastyear=np.max(inflationrates.keys())

#calculate coefficients    
inflation_coeff={lastyear:1.0}

for year in range(2016, 2005, -1):
    inflation_coeff[year]=inflation_coeff[year+1]*inflationrates[year]

    
#_____________________________________________________________________________________________
# inpatient charges
inpatient_info_file = "deid_06-19-2016 IP cost.csv"
inpatient_charges = pd.read_csv(root_folder + inpatient_info_file, parse_dates=['ADMISSION DATE', 'DISCHARGE DATE'])

#charges are in US number format 0,000,000.00
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
inpatient_charges['INPATIENT_CHARGES']= inpatient_charges['TOTAL CHARGES'].apply(locale.atof)
#adjust by inflation
inpatient_charges['INPATIENT_CHARGES']=inpatient_charges.apply(lambda row: row['INPATIENT_CHARGES']*inflation_coeff[ row['year'] ], axis=1)

inpatient_charges_df=inpatient_charges.groupby('AUTO_ID').agg('sum').filter(items=['INPATIENT_CHARGES'])
inpatient_charges_df['infirstdate']=inpatient_charges.groupby('AUTO_ID').agg('min').filter(items=['ADMISSION DATE'])
patients_df=patients_df.join(inpatient_charges_df)

print("\t\t -- DONE!\nLoading outpatient charges...\n"),

#_____________________________________________________________________________________________
# outpatient charges
outpatient_info_file = "deid_06-19-2017 OP Only 2017.csv"
outpatient_charges_transactions = pd.read_csv(root_folder + outpatient_info_file, parse_dates=['ORIG_SERVICE_DATE'])
#adjust by inflation
outpatient_charges_transactions['AMOUNT']=outpatient_charges_transactions.apply(lambda row: row['AMOUNT']*inflation_coeff[ row['Year'] ], axis=1)

#remove refunds <0 (locale already set!)
outpatient_charges=outpatient_charges_transactions[ outpatient_charges_transactions['AMOUNT']>0 ]
outpatient_charges_df=outpatient_charges.groupby('AUTO_ID').agg('sum').filter(items=['AMOUNT'])
outpatient_charges_df['outfirstdate']=outpatient_charges.groupby('AUTO_ID').agg('min').filter(items=['ORIG_SERVICE_DATE'])
outpatient_charges_df.rename(columns={'AMOUNT':'OUTPATIENT_CHARGES'}, inplace=True)
patients_df=patients_df.join(outpatient_charges_df)


print("\t\t -- DONE!\nCalculating avg. charges per year since first charge...\n"),
#combine them and calculate average
blank=now-dt.timedelta(1)
blank=dt.date(blank.year, blank.month, blank.day)
# Add a blank date to make it possible to compare columns
patients_df['outfirstdate'].fillna(blank, inplace=True)
patients_df['infirstdate'].fillna(blank, inplace=True)
# put zeros to avoid missing values where there is just one type missing
patients_df['OUTPATIENT_CHARGES'].fillna(0.0, inplace=True)
patients_df['INPATIENT_CHARGES'].fillna(0.0, inplace=True)
# take minimum of two columns
patients_df['firstchargedate']= patients_df.loc[:, ['outfirstdate', 'infirstdate']].min(axis=1)
patients_df['CHARGE_AVG_PER_YEAR']=(patients_df['OUTPATIENT_CHARGES']+patients_df['INPATIENT_CHARGES'])/((now-patients_df['firstchargedate'])/dt.timedelta(365))
# restore missing values, when there are no charges (can be removed later)
patients_df['CHARGE_AVG_PER_YEAR'].replace([0.0],[None],inplace=True)

#discretize
patients_df['CHARGE_AVG_PER_YEAR']=pd.qcut(patients_df['CHARGE_AVG_PER_YEAR'], q=[0.0,0.5,0.9,1.0], labels=['Low','Mid','High'])
#drop unnecessary columns
patients_df.drop(['firstchargedate','outfirstdate','infirstdate','OUTPATIENT_CHARGES','INPATIENT_CHARGES'],axis=1,inplace=True)

#_____________________________________________________________________________________________
# ZIP codes
zipcodes_stats_file = "MedianZIP.csv"
zipcodes_stats = pd.read_csv(root_folder + zipcodes_stats_file)



print("\t\t -- DONE!\n --------------------- \nSaving to CSV file...\n"),
patients_df.to_csv(output_folder + 'master_Dataset_with_pandas.csv')
print("\n\n\t\t -- ALL DONE!\n\n")




