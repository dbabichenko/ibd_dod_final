import csv
import os

from utilities import sqlutils, stringutils, globals

filename = "Eosinophilia time to hospitalization 3.17.15.csv"

os.chdir(globals.CSV_RESEARCH_SOURCE_FOLDER);

with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
            dt = stringutils.convertDateToMySQL(row[1])
            print(dt)
            print(mrn)


            sql = "INSERT INTO ibd_research.eos_hospitalizations (mrn, eosDate) VALUES (%s, %s);"
            sqlutils.execMysqlQuery(sql, (mrn, dt))




