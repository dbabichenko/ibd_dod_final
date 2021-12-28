import csv, os
import ast
import pandas as pd
import math
import datetime as dt
import locale

#root_folder = "/Users/dmitriyb/Box Sync/Research/IBD Research/Inflammatory Bowel Disease DOD Grant/data/De-identified with linkers 11-24-2017/"
#root_folder = "C:/Users/Marcin/Documents/studies/IBD/data/De-identified with linkers 11-24-2017/"
root_folder ="/home/marcin/work/IBD/InflammatoryBowelDiseaseDODGrant/data/Deidentifiedwithlinkers11242017/"

output_folder =""
#output_folder ="/home/marcin/work/IBD/"
#output_folder ="C:/Users/Marcin/Documents/studies/IBD/data/output/"

# Function definitions

# def group_by_column_with_count(data_dict, data_row, key_col_id, col_name, col_list):
#    if key_col_id in data_dict.keys():
#        if row[col_name] in data_dict[key_col_id].keys():
#            data_dict[key_col_id][row[col_name]] += 1
#        else:
#            data_dict[key_col_id][row[col_name]] = 1
#    else:
#        data_dict[key_col_id] = {}
#        data_dict[key_col_id][row[col_name]] = 1





# Demographic info
demographic_info_file = "deid_IBD_Registry_Participant_List_2017-01-19.csv"

# Columns:
demographic_columns = {'AUTO_ID' : 0,
                       'GENDER' : 1,
                       'BIRTH_YEAR' : 2,
                       'MARITAL_STATUS' : 3,
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


cnt = 0
patient_dict = {}
file = open(root_folder + demographic_info_file)
reader = csv.reader(file)
for row in reader:
    if cnt > 0:
        patient_dict[str(row[0])] = row
    cnt = cnt + 1

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
psych_icd9 = ['295.', '296.', '300.02', '304.', '303.', '300.3', '309.81', '305.1']

# Columns
dx_columns = {'AUTO_ID': 0,
              'DATE_OF_ENTRY': 1,
              'NOTED_DATE': 2,
              'RESOLVED_DATE': 3,
              'ICD9_CODE': 4,
              'ICD10_CODE': 5,
              'DIAGNOSIS_NAME': 6}
cnt = 0
dx_dict = {}
file = open(root_folder + dx_info_file)
reader = csv.reader(file)
for row in reader:
    if cnt > 0:
        dx_dict[str(row[0])] = lookup_dx(psych_icd9, row[dx_columns['ICD9_CODE']])
    cnt = cnt + 1

for key, val in patient_dict.items():
    if key in dx_dict.keys() and patient_dict[key] != None:
        # patient_dict[key] = patient_dict[key].append(dx_dict[key])
        patient_dict[key].append(dx_dict[key])
    else:
        patient_dict[key].append("False")
        #print(patient_dict[key])
# print(len(dx_dict))


#_____________________________________________________________________________________________
# Medications

# add this row to count each medication for each patient
# 
# key_col_id - patient id
# 
def group_by_column_with_count(data_dict, data_row, key_col_id, col_name, col_list):
    if key_col_id != "NO_MATCH":
        if key_col_id not in data_dict.keys():
            data_dict[key_col_id] = {}
            sorted(col_list)
            for col in col_list:
                data_dict[key_col_id][col] = 0
        #if col_list[0]=='High':
        #print(key_col_id + " : " + str(data_dict[key_col_id]))
        #print(data_row[col_name])
            
        data_dict[key_col_id][data_row[col_name]] += 1





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
med_cols_list = ["5_ASA", "Immunomodulators", "Systemic_steroids", "Vitamin_D", "ANTI_INTEGRIN", "ANTI_IL12",
                 "ANTI_TNF"]

sorted(med_cols_list)

output_columns = output_columns + med_cols_list

# print(med_cols_dict)

# count medications in each group for each patient in med datafile that are in med_cols_list
file = open(root_folder + med_info_file)
reader = csv.reader(file)
for row in reader:
    if cnt > 0:
        # med_dict[str(row[0])] = row
        group_by_column_with_count(med_dict, row, str(row[0]), med_columns['GROUP'], med_cols_list)
    cnt = cnt + 1

# print(med_dict)

# for key in med_dict.keys():
#    print(key + ": " + str(med_dict[key]))

# if patient is in among our patients add counts to its row
for key, val in patient_dict.items():
    if key in med_dict.keys() and patient_dict[key] != None:
        # patient_dict[key] = patient_dict[key].append(dx_dict[key])
        # patient_dict[key] = patient_dict[key] + med_dict[key]
        # print(patient_dict[key])
        for med, med_count in med_dict[key].items():
            patient_dict[key].append(med_count)
    else:
        for med, med_count in med_dict['1'].items(): # No.1 has meds
            patient_dict[key].append(None)
    #else:
        #for med_col_name in med_cols_list:
        #    patient_dict[key].append(0)

#for key, val in patient_dict.items():
#    print(val)


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

sorted(enc_cols_list)

output_columns = output_columns + enc_cols_list

# count each encounter type for each patient
cnt = 0
encounters_dict = {}
file = open(root_folder + encounters_info_file)
reader = csv.reader(file)
for row in reader:
    if cnt > 0:
        group_by_column_with_count(encounters_dict, row, str(row[0]), encounters_columns['ENC_TYPE_NAME'],
                                   enc_cols_list)
    cnt = cnt + 1

for key, val in patient_dict.items():
    if key in encounters_dict.keys() and patient_dict[key] != None:
        for enc, enc_count in encounters_dict[key].items():
            patient_dict[key].append(enc_count)
    else:
        for enc, enc_count in encounters_dict['1'].items(): # No.1 has encounters
            patient_dict[key].append(None)
#for key, val in patient_dict.items():
#    print(val)




#_____________________________________________________________________________________________


# Labs
labs_info_file = "filtered_labs.csv"

# Columns:
labs_columns = {'AUTO_ID': 0,
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
labs_dict = {}
labFlag_col_list = ['High', 'Low'] #others are Low Panic, Abnormal, (NONE)
labFlag_dict = {}
file = open(root_folder + labs_info_file)
reader = csv.reader(file)
for row in reader:
    if cnt > 0:
        group_by_column_with_count(labs_dict, row, str(row[0]), labs_columns['GROUP'], lab_col_list)
        if row[labs_columns['RESULT_FLAG']] in labFlag_col_list:
            group_by_column_with_count(labFlag_dict, row, str(row[0]), labs_columns['RESULT_FLAG'], labFlag_col_list)
    cnt = cnt + 1

#for key, val in labs_dict.items():
#    print(key)
#    print(val)

#do we need this?
for key, val in patient_dict.items():
    if key in labs_dict.keys() and patient_dict[key] != None:
        for lab, lab_count in labs_dict[key].items():
            patient_dict[key].append(lab_count)
    else:
        for lab, lab_count in labs_dict['1'].items():
            patient_dict[key].append(None)
    if key in labFlag_dict.keys() and patient_dict[key] != None:
        for labFlag, labFlag_count in labFlag_dict[key].items():
            patient_dict[key].append(labFlag_count)
    else:
        for labFlag, labFlag_count in labFlag_dict['1'].items():
            patient_dict[key].append(None)
output_columns = output_columns + lab_col_list + labFlag_col_list
#_____________________________________________________________________________________________


sibdq_info_file = "cleaned_deid_SIBDQ.csv"
sibdq = pd.read_csv(root_folder + sibdq_info_file, parse_dates=['CONTACT_DATE'])
sibdq_avg=sibdq.groupby(['AUTO_ID']).agg('mean').filter(items=['SIBDQ_TOTAL_SCORE'])
for key,val in sibdq_avg.iterrows():
    #print(val[1])
    if patient_dict[str(key)] != None:
        patient_dict[str(key)].append(val[0])
for key,val in patient_dict.items():
    # index -1 for last element -- this solution is not elegant
    if patient_dict[key][-1]==None or not isinstance(patient_dict[key][-1], float): 
        patient_dict[key].append(None)

output_columns = output_columns + ['SIBDQ_avg']

#_____________________________________________________________________________________________

#pain at last visit

#get indices of last 'CONTACT_DATE''s
sibdq_grouped_by_patient=sibdq.groupby(['AUTO_ID'], sort=False)
sibdq_idx=sibdq_grouped_by_patient['CONTACT_DATE'].transform(max)==sibdq['CONTACT_DATE']
#peek these rows
sibdq_last_visits=sibdq[sibdq_idx]
sibdq_visits_present_ids=[]
#add them to data dictionary
for key,val in sibdq_last_visits.iterrows():
    #print(val[0])
    #print(val[-1])
    if patient_dict[str(val[0])] != None:
        if str(val[-2])=='nan': #in dataframe 'nan' appears for missing value
            patient_dict[str(val[0])].append(None)
        else:
            patient_dict[str(val[0])].append(val[-2]) #
        sibdq_visits_present_ids.append(str(val[0]))

#add missing values for the rest
sibdq_visits_present_ids.sort()
for key,val in patient_dict.items():
    if key not in sibdq_visits_present_ids:
        patient_dict[key].append(None)

output_columns = output_columns + ['pain_last_visit']
#_____________________________________________________________________________________________

#last_year=lambda d: d[ d.groupby('AUTO_ID')['CONTACT_DATE'].max() - d['CONTACT_DATE'] < dt.timedelta(365)]

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

sibdq['lastVisit']= sibdq.groupby('AUTO_ID')['CONTACT_DATE'].transform(lambda x: x.max())
sibdq_last_year=sibdq[ sibdq['lastVisit'] - sibdq['CONTACT_DATE']< dt.timedelta(365)]
sibdq_ly_means=sibdq_last_year.groupby('AUTO_ID').aggregate('mean')

ques_chosen=[0,3,4,7]

for i in ques_chosen:
    sibdq_present_ids=[]
    #add them to data dictionary
    for key,val in sibdq_ly_means.iterrows():
        #print(val[0])
        #print(val[-1])
        #print(key)
        if patient_dict[str(key)] != None:
            if str(sibdq_ly_means[ Ques[i] ][key])=='nan': #in dataframe 'nan' appears for missing value
                patient_dict[str(key)].append(None)
            else:
                patient_dict[str(key)].append( str(sibdq_ly_means[ Ques[i] ][key]) ) #
            sibdq_visits_present_ids.append(str(key))

    #add missing values for the rest
    sibdq_visits_present_ids.sort()
    for key,val in patient_dict.items():
        if key not in sibdq_visits_present_ids:
            patient_dict[key].append(None)
    
    output_columns = output_columns + [ Ques[i]+'_ANNUAL_MEAN' ]
    
#_____________________________________________________________________________________________
# inpatient charges
inpatient_info_file = "deid_06-19-2016 IP cost.csv"
inpatient_charges = pd.read_csv(root_folder + inpatient_info_file, parse_dates=['ADMISSION DATE', 'DISCHARGE DATE'])

#charges are in US number format 0,000,000.00
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')
inpatient_charges['floatCHARGES']= inpatient_charges['TOTAL CHARGES'].apply(locale.atof)

inpatient_charges_dict=inpatient_charges.groupby('AUTO_ID').agg('sum').filter(items=['floatCHARGES']).to_dict()['floatCHARGES']


inpatient_charges_present_ids=[]
#add them to data dictionary
for key,val in inpatient_charges_dict.items():
    #print(val[0])
    #print(val[-1])
    if patient_dict[str(key)] != None:
        patient_dict[str(key)].append(val) #
        inpatient_charges_present_ids.append(str(key))

#add missing values for the rest
inpatient_charges_present_ids.sort()
for key,val in patient_dict.items():
    if key not in inpatient_charges_present_ids:
        patient_dict[key].append(None)


output_columns = output_columns + ['inpatient_charges']
#_____________________________________________________________________________________________
# inpatient charges
outpatient_info_file = "deid_06-19-2017 OP Only 2017.csv"
outpatient_charges_transactions = pd.read_csv(root_folder + outpatient_info_file, parse_dates=['ORIG_SERVICE_DATE'])

#charges are in US number format 0,000,000.00
#outpatient_charges['floatCHARGES']= outpatient_charges['TOTAL CHARGES'].apply(locale.atof)

outpatient_charges=outpatient_charges_transactions[ outpatient_charges_transactions['AMOUNT']>0 ]
outpatient_charges_dict=outpatient_charges.groupby('AUTO_ID').agg('sum').filter(items=['AMOUNT']).to_dict()['AMOUNT']


outpatient_charges_present_ids=[]
#add them to data dictionary
for key,val in outpatient_charges_dict.items():
    #print(val[0])
    #print(val[-1])
    if patient_dict[str(key)] != None:
        patient_dict[str(key)].append(val) #
        outpatient_charges_present_ids.append(str(key))

#add missing values for the rest
outpatient_charges_present_ids.sort()
for key,val in patient_dict.items():
    if key not in outpatient_charges_present_ids:
        patient_dict[key].append(None)


output_columns = output_columns + ['outpatient_charges']


#_____________________________________________________________________________________________
# Output

os.chdir(os.getcwd())
output_file = open(output_folder + 'master_dataset_v1.csv', 'w')
writer = csv.writer(output_file)

writer.writerow(output_columns)
for key, val in patient_dict.items():
    writer.writerow(val)
    #print(val)

#It seems to be needed in Windows
output_file.close()

print("DONE")