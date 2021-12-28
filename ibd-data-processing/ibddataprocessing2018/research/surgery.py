import csv
import os

from utilities import sqlutils, stringutils, globals

filename = "Surgery 2009-2014 MASTER.csv"

os.chdir(globals.CSV_RESEARCH_SOURCE_FOLDER);

with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
            before_2009 = row[1]
            y2009 = row[2]
            y2010 = row[3]
            y2011 = row[4]
            y2012 = row[5]
            y2013 = row[6]
            y2014 = row[7]


            sql = "INSERT IGNORE INTO ibd_research.surgery(mrn,before_2009,y2009,y2010,y2011,y2012,y2013,y2014) "
            sql+= " VALUES (%s, %s, %s, %s, %s, %s, %s, %s); ";
            sqlutils.execMysqlQuery(sql, (mrn, before_2009,y2009,y2010,y2011,y2012,y2013,y2014))




