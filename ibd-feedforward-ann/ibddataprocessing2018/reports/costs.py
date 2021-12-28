import pymysql

from utilities import reportutils
from utilities import stringutils
from utilities import globals





colListSql = "SELECT DISTINCT category FROM procedure_costs WHERE category <> '' ORDER BY category;"

sql = "SELECT fk_patientID, category, SUM(amount) FROM procedure_costs "
sql += "WHERE category <> '' AND amount >= 0  "
sql += "GROUP BY fk_patientID, category ORDER BY fk_patientID, category; "

outputFilePath = outputFilePath = globals.OUTPUT_FOLDER_PATH + 'ProcedureCostByCategory.csv'

reportutils.pivotFromQuery(colListSql, sql, outputFilePath)

'''
colListSql = "SELECT DISTINCT fk_hospitalID FROM patient_hospital_cost;"
dataSQL = "SELECT fk_patientID, fk_hospitalID, SUM(totalCharges) FROM patient_hospital_cost "
dataSQL += "GROUP BY fk_patientID, fk_hospitalID ORDER BY fk_patientID, fk_hospitalID; "

outputFilePath = outputFilePath = globals.OUTPUT_FOLDER_PATH + 'HospitalCost.csv'
reportutils.pivotFromQuery(colListSql, dataSQL, outputFilePath)
'''

