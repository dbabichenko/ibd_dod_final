import csv
import os

from utilities import sqlutils, stringutils, globals

filename = "PBE + CRP vs CRP 4.27.15.csv"

os.chdir(globals.CSV_RESEARCH_SOURCE_FOLDER);

with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
            _id = row[1]
            print(_id)
            _dt = stringutils.convertDateToMySQL(row[2])
            print(_dt)
            _group = row[3]
            sql = "INSERT INTO ibd_research.pbe_crp(`mrn`,`id`,`dt`,`group`) "
            sql+= " VALUES (%s, %s, %s, %s);"

            sqlutils.execMysqlQuery(sql, (mrn, _id, _dt, _group))




