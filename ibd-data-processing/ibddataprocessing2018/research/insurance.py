import csv
import os

from utilities import sqlutils, stringutils, globals

filename = "Medical Insurance.csv"

os.chdir(globals.CSV_RESEARCH_SOURCE_FOLDER);

with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
            insurance = row[1]
            sql = "INSERT INTO ibd_research.insurance(mrn,insurance) VALUES (%s, %s);"

            sqlutils.execMysqlQuery(sql, (mrn, insurance))




