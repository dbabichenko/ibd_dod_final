import csv, os
from dictionaries import meds_dictionary



def lookup_row(row):
    #print(row[3])
    #print(row[5])
    if row[5] in meds_dictionary.meds_dict.keys():
        if len(meds_dictionary.meds_dict[row[5]].med_generic_name) == 0:
            return meds_dictionary.meds_dict[row[5]].group
        else:
            if row[3] in meds_dictionary.meds_dict[row[5]].med_names:
                return meds_dictionary.meds_dict[row[5]].group
    return ''


source_file = 'deid_IBD Reg Pts Medications_2017-01-19.csv'


base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/De-identified with linkers 11-24-2017/"




output_file = base_path + "filtered_meds.csv"
writer = csv.writer(open(output_file, 'w'))

cnt = 0
data_file = open(base_path + source_file, 'rU')
for row in csv.reader(data_file):
    if cnt == 0:
        row.append("GROUP")
        writer.writerow(row)
    else:

        group = lookup_row(row)
        if group != '':
            row.append(group)
            writer.writerow(row)

        #print(row)
    cnt = cnt + 1

data_file.close()



