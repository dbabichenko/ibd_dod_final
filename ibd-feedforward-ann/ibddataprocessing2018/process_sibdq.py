'''
Remove text labels - leave numeric data only
Remove rows with less then 5 responses
Remove empty rows
For rows with complete responses but missing sum, add responses' numerical values
Flag rows with with missing responses
'''

import csv, os


def is_complete_row(row):
    cnt_blank = 0
    for i in range(4, 14):
        if row[i] == '':
            cnt_blank = cnt_blank + 1

    if cnt_blank >= 5:
        return False
    else:
        return True


source_file_name = 'deid_SIBDQ.csv'


base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/De-identified with linkers 11-24-2017/"




output_file = base_path + "cleaned_deid_SIBDQ.csv"
writer = csv.writer(open(output_file, 'w'))

cnt = 0


data_file = open(os.path.abspath(base_path + source_file_name), 'rU')
for row in csv.reader(data_file):
    if is_complete_row(row):
        if cnt == 0:
            writer.writerow(row)
        else:
            total = 0
            has_missing_vals = False
            for i in range(4,14):
                '''
                num_arr = str(row[i]).split("=")
                if len(num_arr) == 2:
                    num_val = num_arr[0].strip()
                    row[i] = num_val
                else:
                    row[i] = ""
                '''
                row[i] = str(row[i])[0:1]
                if row[i] != '':
                    total = total + int(row[i])
                else:
                    has_missing_vals = True
            row.append(str(has_missing_vals))
            row[14] = total
            print(row)
            print(total)
            writer.writerow(row)

        #print(row)
    cnt = cnt + 1


data_file.close()



