import csv
import os

from utilities import sqlutils, stringutils, globals


# This script imports immunization data for existing study patients
# Data is imported from IBD_Reg_Pts_Flu_Pneumovax_2014-03-04.csv
# The following columns are imported:
#       PAT_ID (MRN)
#       GENDER
#       IMMUNE_DATE
#       IMMUNIZATION_NAME
# This script populates the following two tables in ibd_import database:
#       immunization
# Excel to database field mappings are defined in mappings/2014/immunization.json


# Get a list of mapped fields
map = stringutils.getJsonMapping(globals.CURRENT_YEAR, 'immunization')


filename = map["FILENAME"] # Source filename (defined in mappings JSON file)

os.chdir(globals.CSV_SOURCE_FOLDER);


# Populate a dictionary object with values from medication table.  This greatly improves performance over
# looking up values for each INSERT or UPDATE statement.  Also helps prevent duplicates.
sql = "SELECT patientID, encryptedPatientID FROM patient;"
patients = sqlutils.tableToDictionary(sql, 1, 0)

# Start pathology data import
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

                immunizationDate = stringutils.convertDateToMySQL(row[map['IMMUNE_DATE']].strip())
                immunizationName = row[map['IMMUNIZATION_NAME']]


                sql = "INSERT INTO immunizations (fk_patientID, immuneDate, immunizationName) "
                sql += "VALUES (%s,%s,%s);"
                sqlutils.execMysqlQuery(sql, (patientID, immunizationDate, immunizationName))


        counter = counter + 1  # Note - counter is only used to ignore the first row of the spreadsheet (contains column headings)


