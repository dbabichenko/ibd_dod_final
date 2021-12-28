import base64

from Crypto import Random
from Crypto.Cipher import AES
import csv, os
from utilities import stringutils, globals

source_files = ['06-19-2017 OP Only 2017.csv',
    'Harvey_Bradshaw_Questionnaires_2017-01-19.csv',
    'IBD Reg Pts Medications_2017-01-19.csv',
    'IBD Reg Pts w ER_OP_DC Reports_2017-01-19.csv',
    'IBD_Pts_w_Rad_Diagnostic_Tests_2017-01-23.csv',
    'IBD_Reg_Pts_Family_Hx_2017-01-19.csv',
    'IBD_Reg_Pts_Insurance_2017-01-19.csv',
    'IBD_Reg_Pts_Social_Hx_2017-01-19.csv',
    'IBD_Reg_Pts_Vitals_BP_Ht_Wt_2017-01-19.csv',
    'IBD_Reg_Pts_w_Labs_2017-01-19_part1.csv',
    'IBD_Reg_Pts_w_Labs_2017-01-19_part2.csv',
    'IBD_Reg_Pts_w_Office_Phone_Email_2017-01-19.csv',
    'IBD_Reg_Pts_w_Problem_List_2017-01-19.csv',
    'IBD_Registry_Participant_List_2017-01-19.csv',
    'SIBDQ_Pain_Questionaire_2017-01-19.csv',
    '06-19-2016 IP cost.csv']

#source_files = ['06-19-2017 OP Only 2017.csv']
mrn_dict = {}

mrn_source = "IBD_Registry_Participant_List_2017-01-19.csv"
base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/"
filename = base_path + mrn_source


seq_id = 0
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
            if mrn not in mrn_dict.keys():
                mrn_dict[mrn] = seq_id
                seq_id = seq_id + 1
            #print(str(row).replace('[', '').replace('[', ''))
            #writer.writerow(row)
            #print(row)


#print(mrn_dict)
for key in mrn_dict.keys():
    print(key + ": " + str(mrn_dict[key]))


for orig_filename in source_files:
    output_file = base_path + "deid_" + orig_filename
    writer = csv.writer(open(output_file, 'w'))
    print("Working with: " + orig_filename)
    with open(os.path.abspath(base_path + orig_filename), 'rU') as f:
        for row in csv.reader(f):
            if any(row):
                mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
                if mrn in mrn_dict.keys():
                    row.insert(0, mrn_dict[mrn])
                else:
                    row.insert(0, 'NO_MATCH')
                writer.writerow(row)


