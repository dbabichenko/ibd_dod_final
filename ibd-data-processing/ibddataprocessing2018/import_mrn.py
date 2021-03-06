import csv
import os

from utilities import sqlutils, stringutils, globals

# This script imports data from patients who agreed to participate in the IBD registry
# Data is imported from IBD_Registry_Participant_List_2014-02-21.csv
# Only the following columns are imported:
#       MRN (Patient ID)
#       Date of Birth (converted to year of birth)
#       Race
#       Ethnic group
#       Employment Status
# This script populates the following three table in ibd2015dw database:
#       patient
# Excel to database field mappings are defined in mappings/2015/patients.json


# Get a list of mapped fields
map = stringutils.getJsonMapping(2015, 'patients')


#Import patient information
filename = map["FILENAME"] # Source filename (defined in mappings JSON file)

os.chdir(globals.CSV_SOURCE_FOLDER);


# Start patient data import
counter = 0
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row) and counter > 0:
            # Even though the primary key used for patients' identifiers is auto-generated by MySQL
            # (sequential number), we still store the original MRN encrypted with MD5 hash
            # to link back to external records and to identify new patient records.
            rawMrn = stringutils.prefixZeros(row[map['MRN']], globals.MAX_MRN_LENGTH)
            #print(rawMrn)
            mrn = stringutils.computeMD5hash(rawMrn)



            sql = "INSERT IGNORE INTO mrn_map (mrn, encryptedMrn) VALUES (%s,%s)"
            sqlutils.execMysqlQuery(sql, (rawMrn, mrn))


        counter = counter + 1  # Note - counter is only used to ignore the first row of the spreadsheet (contains column headings)



