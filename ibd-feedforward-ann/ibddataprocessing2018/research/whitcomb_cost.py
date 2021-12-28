import csv
import os
import uuid

from utilities import sqlutils, stringutils, globals

sql = "SELECT procCode, category FROM ibd2015dw.procedure_costs GROUP BY procCode, category;"
categories = sqlutils.tableToDictionary(sql, 0, 1)


filename = "02-20-2015 Whitcomb Charge Data.csv"

os.chdir(globals.CSV_RESEARCH_SOURCE_FOLDER);

target_file = open('updated_' + filename + ".tsv",'w')
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            #print(len(row))
            proc_code = row[7]
            if proc_code in categories.keys():
                #print(proc_code + ": " + categories[proc_code])
                row.append(categories[proc_code])
            else:
                row.append("")

            str_row = ""
            for i, val in enumerate(row):
                if i == 1:
                    str_row = str_row + stringutils.prefixZeros(val, globals.MAX_MRN_LENGTH) + "\t"
                else:
                    str_row = str_row + str(val) + "\t"

            print(str_row)
            target_file.write(str_row + '\n')






#UPDATE procedure_costs a JOIN procedure_categories b ON a.procCode = b.procCode SET a.category = b.category;