'''
* Separate UC from CR
* CR & UC are not mutually exclusive
* Flag rows with missing total score for both UC & CR
* Remove non-numerical labels - leave numbers only
'''

import csv, os


def has_values(row, col_range):
    for i in col_range:
        if row[i] != '':
            return True


    return False


source_file_name = 'deid_HBI-UCAI.csv'


base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/De-identified with linkers 11-24-2017/"




cr_output_file = base_path + "cleaned_deid_CR.csv"
uc_ouptut_file = base_path + "cleanded_deid_UC.csv"
cr_writer = csv.writer(open(cr_output_file, 'w'))
uc_writer = csv.writer(open(uc_ouptut_file, 'w'))

cnt = 0


# Columns 0 - 2: Both HBI & UCIA
# Columns 3 - 15: CR
# Columns 16 + : UC
data_file = open(os.path.abspath(base_path + source_file_name), 'rU')
for row in csv.reader(data_file):
    if any(row):
        if cnt == 0:
            cr_row = row[0:16]
            uc_row = row[0:4] + row[16:]
            cr_writer.writerow(cr_row)
            uc_writer.writerow(uc_row)
        else:
            cr_row = row[0:16]
            for i in range(3, len(cr_row)):
                cr_row[i] = str(cr_row[i])[0:1]
            uc_row = row[0:4] + row[16:]
            for i in range(3, len(uc_row)):
                uc_row[i] = str(uc_row[i])[0:1]

            if has_values(cr_row, range(3, len(cr_row))):
                cr_writer.writerow(cr_row)

            if has_values(uc_row, range(3, len(uc_row))):
                uc_writer.writerow(uc_row)

        #print(row)
    cnt = cnt + 1

    
data_file.close()



