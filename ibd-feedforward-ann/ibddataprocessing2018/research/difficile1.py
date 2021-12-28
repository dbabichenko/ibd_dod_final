import csv
import os

from utilities import sqlutils, stringutils, globals


sql = " SELECT mrn, notes FROM ibd2015dw.patient_chart_notes "
sql += " WHERE (notes LIKE '%c.diff%' OR notes LIKE '%difficile%' "
sql += "OR notes LIKE '%c diff%') "

sql = "SELECT mrn, notes FROM patient_chart_notes WHERE mrn = '075799055' AND (notes LIKE '%small bowel%' AND notes LIKE '%3/16/2009%');"
#sql += "AND (notes LIKE '%not detected%' OR notes LIKE '%negative%'); "
conn = sqlutils.getConnection()
cur = conn.cursor()
cur.execute(sql)



for row in cur:
    #stringToExaming =
    note = row[1].lower()

    print('||||' + row[0] + '|||' + note)
    print('____________________________________________________________________')


cur.close()
conn.close()