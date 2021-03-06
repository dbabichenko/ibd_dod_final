import csv
import os

from utilities import sqlutils, stringutils, globals

# This script imports pathology data for existing study patients
# Data is imported from IBD_Reg_Pts_Pathology_2014-03-04.csv
# The following columns are imported:
#       PAT_ID (MRN)
#       ORDER_DATE
#       ORDER_STATUS
#       CPT_CODE
#       RESULT_DATE
#       NARRATIVE
# This script populates the following two tables in ibd_import database:
#       pathology
# Excel to database field mappings are defined in mappings/2014/pathology.json


# Get a list of mapped fields
map = stringutils.getJsonMapping(globals.CURRENT_YEAR, 'pathology')


filename = map["FILENAME"] # Source filename (defined in mappings JSON file)

os.chdir(globals.CSV_SOURCE_FOLDER);


# Populate a dictionary object with values from medication table.  This greatly improves performance over
# looking up values for each INSERT or UPDATE statement.  Also helps prevent duplicates.
sql = "SELECT patientID, encryptedPatientID FROM patient;"
patients = sqlutils.tableToDictionary(sql, 1, 0)

# Start pathology data import
counter = 0
narrative = ""
orderDate = ""
orderStatus = ""
resultDate = ""
narrative = ""

pathology = []
pathRow = ['MRN','ORDER_DATE','ORDER_STATUS','RESULT_DATE','NARRATIVE']
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row) and counter > 0:
            # Even though the primary key used for patients' identifiers is auto-generated by MySQL
            # (sequential number), we still store the original MRN encrypted with MD5 hash
            # to link back to external records and to identify new patient records.
            mrn = row[map['MRN']]

            if mrn != "": # and mrn in patients.keys():
                #patientID = patients[mrn]
                pathology.append(pathRow)
                orderDate = stringutils.convertDateToMySQL(row[map['ORDER_DATE']].strip())
                orderStatus = row[map['ORDER_STATUS']]
                resultDate = stringutils.convertDateToMySQL(row[map['RESULT_DATE']].strip())
                narrative = row[map['NARRATIVE']]

                pathRow[0] = mrn
                pathRow[1] = orderDate
                pathRow[2] = orderStatus
                pathRow[3] = resultDate
                pathRow[4] = narrative

            else:
                narrative = row[map['NARRATIVE']]
                #print(narrative)
                pathRow[4] = pathRow[4] + "\n" + narrative
            #    sql = "UPDATE pathology SET narrative = %s"
            #    print(sql)

            #print(narrative)
            #if counter > 5000:
            #    break





        counter = counter + 1  # Note - counter is only used to ignore the first row of the spreadsheet (contains column headings)


for row in pathology:
    rawMrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
    mrn = stringutils.computeMD5hash(rawMrn)
    if mrn in patients.keys():
        patientID = patients[mrn]
        orderDate = row[1]
        orderStatus = row[2]
        resultDate = row[3]
        narrative = row[4]

        #print(narrative)

        sql = "INSERT INTO pathology_reports (fk_patientID, orderDate, orderStatus, resultDate, NARRATIVE) "
        sql += "VALUES (%s,%s,%s,%s,%s);"
        sqlutils.execMysqlQuery(sql, (patientID, orderDate, resultDate, resultDate, narrative))