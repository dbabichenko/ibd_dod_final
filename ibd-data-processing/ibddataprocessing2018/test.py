import pymysql

from utilities import reportutils
from utilities import stringutils
import md5

list = ['low', 'grade', 'dysplasia']
if all(word in 'low-grade dysplasia found in patient' for word in list):
    print("TRUE")
else:
    print("FALSE")

'''

colListSql = "SELECT DISTINCT category FROM `procedure` WHERE category <> '' ORDER BY category;"

sql = "SELECT fk_patientID, category, SUM(amount) FROM `procedure` "
sql += "JOIN procedure_cost ON procCode = fk_procCode "
sql += "WHERE category <> '' AND amount >= 0  "
sql += "GROUP BY fk_patientID, category ORDER BY fk_patientID, category; "

outputFilePath = '/Users/dmitriyb/Dropbox/Research/IBD Research/pyimport/ouptut/costpivot.csv'

reportutils.pivotFromQuery(colListSql, sql, outputFilePath)

'''

'''
colListSql = "SELECT DISTINCT fk_hospitalID FROM patient_hospital_cost;"
dataSQL = "SELECT fk_patientID, fk_hospitalID, SUM(totalCharges) FROM patient_hospital_cost "
dataSQL += "GROUP BY fk_patientID, fk_hospitalID ORDER BY fk_patientID, fk_hospitalID; "

outputFilePath = '/Users/dmitriyb/Dropbox/Research/IBD Research/pyimport/ouptut/hospitalcostpivot.csv'
reportutils.pivotFromQuery(colListSql, dataSQL, outputFilePath)

'''

'''
colListSql = "SELECT DISTINCT YEAR(contactDate) FROM patient_encounters WHERE fk_encTypeID = 52 ORDER BY YEAR(contactDate);"

sql = "SELECT fk_patientID, YEAR(contactDate), COUNT(*)  "
sql += "FROM patient_encounters WHERE fk_encTypeID = 52 "
sql += "GROUP BY fk_patientID, YEAR(contactDate) "
sql += "ORDER BY fk_patientID, YEAR(contactDate); "

outputFilePath = '/Users/dmitriyb/Dropbox/Research/IBD Research/pyimport/ouptut/phonecallpivot.csv'

reportutils.pivotFromQuery(colListSql, sql, outputFilePath)
'''
'''
colListSql = "SELECT DISTINCT YEAR(contactDate) FROM patient_encounters WHERE fk_encTypeID = 53 ORDER BY YEAR(contactDate);"

sql = "SELECT fk_patientID, YEAR(contactDate), COUNT(*)  "
sql += "FROM patient_encounters WHERE fk_encTypeID = 53 "
sql += "GROUP BY fk_patientID, YEAR(contactDate) "
sql += "ORDER BY fk_patientID, YEAR(contactDate); "

outputFilePath = '/Users/dmitriyb/Dropbox/Research/IBD Research/pyimport/ouptut/proc_visit_pivot.csv'

reportutils.pivotFromQuery(colListSql, sql, outputFilePath)

'''
