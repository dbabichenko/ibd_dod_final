import csv
import os


from utilities import sqlutils, stringutils, globals


# This script imports diagnostic test data for existing study patients
# Data is imported from IBD_Reg_Pts_w_Office_Phone_Email_2014-03-04.csv
# The following columns are imported:
#
# This script populates the following two tables in ibd_import database:
#
# Excel to database field mappings are defined in mappings/2014/encounters.json


# Get a list of mapped fields
map = stringutils.getJsonMapping(globals.CURRENT_YEAR, 'diagnostictests')

# Populate dictionary objects.  This greatly improves performance over
# looking up values for each INSERT or UPDATE statement.  Also helps prevent duplicates.
sql = "SELECT patientID, encryptedPatientID FROM patient;"
patients = sqlutils.tableToDictionary(sql, 1, 0)


filename = map["FILENAME"] # Source filename (defined in mappings JSON file)

os.chdir(globals.CSV_SOURCE_FOLDER);


# Start diagnostic (radiology) test data import
counter = 0
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row) and counter > 0:
            # Even though the primary key used for patients' identifiers is auto-generated by MySQL
            # (sequential number), we still store the original MRN encrypted with MD5 hash
            # to link back to external records and to identify new patient records.
            rawMrn = stringutils.prefixZeros(row[map['MRN']], globals.MAX_MRN_LENGTH)
            mrn = stringutils.computeMD5hash(rawMrn)

            if mrn in patients.keys():
                patientID = patients[mrn]
                visitDate = stringutils.convertDateToMySQL(row[map['VISIT_DATE']].strip())
                orderingDate = stringutils.convertDateToMySQL(row[map['ORDERING_DATE']].strip())
                cptCode = row[map['CPT_CODE']]
                descr = row[map['DESCRIPTION']]
                quantity = row[map['QUANTITY']]
                if str(quantity).isdigit() == False:
                    quantity = 0
                procStartDate = stringutils.convertDateToMySQL(row[map['PROC_START_DATE']].strip())

                isAlive = 1 if row[map['PAT_STATUS']] == 'Alive' else 0

                proCatName = row[map['PROC_CAT_NAME']]
                orderType = row[map['ORDER_TYPE']]

                orderClass = row[map['ORDER_CLASS']]
                orderStatus = row[map['ORDER_STATUS']]
                updateDate = stringutils.convertDateToMySQL(row[map['UPDATE_DATE']].strip())
                resultDate = stringutils.convertDateToMySQL(row[map['RESULT_DATE']].strip())
                reviewDate = stringutils.convertDateToMySQL(row[map['REVIEW_DATE']].strip())



                sql = "INSERT INTO radiology_diagnostictests(fk_patientID,visitDepartment,visitDate,orderingDate,"
                sql+= "cptCode,description,procCatName,quantity,procStartDate,isAlive,orderType,orderClass,"
                sql+= "orderStatus,updateDate,resultDate,reviewDate)"
                sql += "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
                sqlutils.execMysqlQuery(sql, (patientID, '', visitDate, orderingDate, cptCode, descr, proCatName, quantity, procStartDate, isAlive, orderType, orderClass, orderStatus, updateDate, resultDate, reviewDate))


        counter = counter + 1  # Note - counter is only used to ignore the first row of the spreadsheet (contains column headings)

