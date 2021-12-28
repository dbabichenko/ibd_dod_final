import csv, os
from dictionaries import labs_dictionary



def lookup_row(row):
    if row[7] in labs_dictionary.labs_dict.keys():
        if len(labs_dictionary.labs_dict[row[7]].proc_names) == 0:
            return labs_dictionary.labs_dict[row[7]].group
        else:
            if row[3] in labs_dictionary.labs_dict[row[7]].proc_names:
                return labs_dictionary.labs_dict[row[7]].group
    return ''


source_files = ['deid_IBD_Reg_Pts_w_Labs_2017-01-19_part1.csv','deid_IBD_Reg_Pts_w_Labs_2017-01-19_part2.csv']


base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/De-identified with linkers 11-24-2017/"




output_file = base_path + "filtered_labs.csv"
writer = csv.writer(open(output_file, 'w'))

cnt = 0
for source_file_name in source_files:
    print(source_file_name)
    print("_______________________________")

    data_file = open(os.path.abspath(base_path + source_file_name), 'rU')
    for row in csv.reader(data_file):
        if cnt == 0:
            row.append("GROUP")
            writer.writerow(row)
        else:
            group = lookup_row(row)
            if group != '':
                row.append(group)
                writer.writerow(row)

        cnt = cnt + 1

    data_file.close()



