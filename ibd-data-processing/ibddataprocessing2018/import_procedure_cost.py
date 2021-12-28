import csv
import os
import uuid

from utilities import sqlutils, stringutils, globals

# This script imports procedure cost data for existing study patients
# Data is imported from "New Outpt Cost Categorized.csv"
#       MRN
#       ORIG_SERVICE_DATE
#       AMOUNT
#       PROC_CODE
#       PROC_NAME
#       TYPE_OF_SERVICE
#       PROC_GROUP_NAME
#       category
# This script populates the following table in ibd_import database:
#       procedure
#       procedure_cost
#       procedure_dx
# Excel to database field mappings are defined in mappings/2014/procedure_cost.json


# Get a list of mapped fields
map = stringutils.getJsonMapping(globals.CURRENT_YEAR, 'procedure_cost')


filename = map["FILENAME"] # Source filename (defined in mappings JSON file)

os.chdir(globals.CSV_SOURCE_FOLDER);


# Fill dictionaries
sql = "SELECT patientID, encryptedPatientID FROM patient;"
patients = sqlutils.tableToDictionary(sql, 1, 0)

counter = 0

# Start procedure_cost data import
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row) and counter > 0:
            rawMrn = stringutils.prefixZeros(row[map['MRN']], globals.MAX_MRN_LENGTH)
            mrn = stringutils.computeMD5hash(rawMrn)
            if mrn in patients.keys():
                patientID = patients[mrn]
                procCode = row[map['PROC_CODE']]
                procName = row[map['PROC_NAME']]
                procGroupName = row[map['PROC_GROUP_NAME']]
                category = row[map['CATEGORY']]
                serviceType = row[map['TYPE_OF_SERVICE']]
                serviceDate = stringutils.convertDateToMySQL(row[map['ORIG_SERVICE_DATE']])
                amount = row[map['AMOUNT']]
                #components = row[map['COMPONENTS']]
                costID = uuid.uuid4()
                #sql = "INSERT INTO procedure_costs(costID, fk_patientID,serviceDate,amount,procCode,procName,procGroupName,category,serviceType,components) "
                #sql += "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
                sql = "INSERT INTO procedure_costs(costID, fk_patientID,serviceDate,amount,procCode,procName,procGroupName,category,serviceType) "
                sql += "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
                sqlutils.execMysqlQuery(sql, (str(costID), str(patientID), serviceDate, str(amount), procCode, procName, procGroupName, category, serviceType))

                # Import diagnosis codes (ICD9 codes) associated with each cost item
                dxList = row[map['DX']].split(',')
                for dx in dxList:
                    sql = "INSERT IGNORE INTO procedure_costs_diagnosis(costID,icd9) VALUES (%s, %s);"
                    sqlutils.execMysqlQuery(sql, (str(costID), str(dx)))



        counter = counter + 1  # Note - counter is only used to ignore the first row of the spreadsheet (contains column headings)


#UPDATE procedure_costs SET category = 'inpatient' WHERE category = 'intpatient';
#UPDATE procedure_costs SET category = 'medication' WHERE category = 'medications';

#UPDATE procedure_costs a JOIN procedure_categories b ON a.procCode = b.procCode SET a.category = b.category;