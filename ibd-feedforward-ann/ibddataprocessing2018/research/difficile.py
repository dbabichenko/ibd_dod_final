import csv
import os

from utilities import sqlutils, stringutils, globals


sql = " SELECT DISTINCT mrn FROM ibd2015dw.patient_chart_notes "
sql += " WHERE (notes LIKE '%c.diff%' OR notes LIKE '%difficile%' "
sql += "OR notes LIKE '%c diff%') "
#sql += "AND (notes LIKE '%not detected%' OR notes LIKE '%negative%'); "
conn = sqlutils.getConnection()
cur = conn.cursor()
cur.execute(sql)
print(cur.description)
print()


for row in cur:
    #stringToExaming =
    note = row[2].lower()
    idx1 = -1
    idx2 = -1
    try:
        idx1 = note.index('c.diff')
    except Exception as inst:
        #print("Did not find c.diff")
        print("")

    try:
        idx2 = note.index('difficile')
    except Exception as inst:
        #print("Did not find difficile")
        print("")

    sentence = ""
    if idx1 != -1:
        sentence = note[(idx1 - 200) : (idx1 + 200)]
    if idx2 != -1:
        sentence = note[(idx2 - 200) : (idx2 + 200)]
    #print(row[1] + " - " + str(idx1) + " - " + str(idx2))
    print(sentence)
    print('____________________________________________________________________')


cur.close()
conn.close()