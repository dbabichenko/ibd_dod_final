import csv
import os


from utilities import sqlutils, stringutils, globals

# This script imports laboratory data for existing study patients
# Data is imported from IBD_Reg_Pts_w_Labs_2014-02-21.csv
#       PATIENT_ID (MRN)
#       DESCRIPTION
#       SEVERITY
#       REACTION
# This script populates the following table in ibd_import database:
#       lab
#       patient_lab
# Excel to database field mappings are defined in mappings/2014/labs.json


# Get a list of mapped fields
map = stringutils.getJsonMapping(2017, 'labs')


filename = map["FILENAME"] # Source filename (defined in mappings JSON file)

os.chdir(globals.CSV_SOURCE_FOLDER);




# Fill dictionaries

sql = "SELECT patientID, encryptedPatientID FROM patient;"
patients = sqlutils.tableToDictionary(sql, 1, 0)



# Start patient_lab data import
counter = 0
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row) and counter > 0:
            # Lab test ID
            labID = row[map['PROC_CODE']].strip()

            # MRN
            rawMrn = stringutils.prefixZeros(row[map['MRN']], globals.MAX_MRN_LENGTH)
            mrn = stringutils.computeMD5hash(rawMrn)
            #print("MRN: " + mrn)
            #print("Lab ID: " + str(labID))
            # Insert into patient_lab
            if mrn in patients.keys():
                patientID = patients[mrn]
                # Comp ID
                compID = row[map['LAB_COMP_ID']].strip()
                # Comp Name
                compName = row[map['LAB_COMP_NAME']].strip()

                # Lab test ID
                labID = row[map['PROC_CODE']].strip()
                # Lab test name
                labName = row[map['PROC_NAME']].strip()
                # CPT code
                cptCode = row[map['CPT_CODE']].strip()

                orderDate = stringutils.convertDateToMySQL(row[map['ORDER_DATE']].strip())
                orderStatus = row[map['ORDER_STATUS']].strip()
                resultDate = stringutils.convertDateToMySQL(row[map['RESULT_DATE']].strip())
                orderValue = row[map['ORD_VALUE']].strip()
                orderNumValue = stringutils.stripNonNumeric(row[map['ORD_NUM_VALUE']].strip())
                refLow = stringutils.stripNonNumeric(row[map['REF_LOW']].strip())
                refHigh = stringutils.stripNonNumeric(row[map['REF_HIGH']].strip())
                refUnit = row[map['REF_UNIT']].strip()
                resultFlag = row[map['RESULT_FLAG']].strip()
                refNormalVals = row[map['REF_NORMAL_VALS']].strip();

                if refHigh == "":
                    refHigh = "0"
                if refLow == "":
                    refLow = "0"
                if orderNumValue == "":
                    orderNumValue = "0"
                if compID == "":
                    compID = "0"

                #sql = "INSERT IGNORE INTO patient_lab (fk_patientID, fk_labID, orderDate, orderStatus, resultDate, ordValue, ordNumValue, refLow, refHigh, refUnit, resultFlag) "
                #sql += "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                
                sql = "INSERT INTO labs "
                sql += "(labID, labName,cptCode,labCompID,labCompName,fk_patientID,orderDate,orderStatus,resultDate,ordValue, "
                sql += "ordNumValue,refLow,refHigh,refUnit,resultFlag, refNormalVals) "
                sql += "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "

                sqlutils.execMysqlQuery(sql, (labID, labName, cptCode, compID, compName, str(patientID), orderDate, orderStatus, resultDate, orderValue, orderNumValue, refLow, refHigh, refUnit, resultFlag, refNormalVals))


        counter = counter + 1  # Note - counter is only used to ignore the first row of the spreadsheet (contains column headings)







