import csv
import os

from utilities import sqlutils, stringutils, globals

# This script imports hospitalization cost data for existing study patients
# Data is imported from 09-08-2014 Hospital admission cost data.csv
#       "HOSPITAL"
#       "MRN" (Note that this is an inpatient MRN - everything else uses outpatient MRN)
#       "ADMISSION_DATE"
#       "DISCHARGE_DATE"
#       "TOTAL_CHARGES"
# This script populates the following table in ibd_import database:
#       hospital_cost
# Excel to database field mappings are defined in mappings/2014/hospitals.json


# Get a list of mapped fields
map = stringutils.getJsonMapping(globals.CURRENT_YEAR, 'hospitals')

# Fill dictionaries
sql = "SELECT hospitalID, hospitalName FROM hospital;"
hospital = sqlutils.tableToDictionary(sql, 0, 1)

sql = "SELECT patientID, iMrn FROM inpatient_outpatient JOIN patient ON oMrn = encryptedPatientID;"
patients = sqlutils.tableToDictionary(sql, 1, 0)

filename = map["FILENAME"] # Source filename (defined in mappings JSON file)

os.chdir(globals.CSV_SOURCE_FOLDER);


# Start hospital_cost and lab data import
counter = 0
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row) and counter > 0:
            rawMrn = stringutils.prefixZeros(row[map['MRN']], globals.MAX_MRN_LENGTH)
            mrn = stringutils.computeMD5hash(rawMrn)
            hospitalID = row[map['HOSPITAL']].strip()
            print(mrn)
            print(hospitalID)
            if mrn in patients.keys() and hospitalID in hospital.keys():
                patientID = patients[mrn]

                admissionDate = stringutils.convertDateToMySQL(row[map['ADMISSION_DATE']])
                dischargeDate = stringutils.convertDateToMySQL(row[map['DISCHARGE_DATE']])
                totalCharges = row[map['TOTAL_CHARGES']]
                insuranceAdjustments = row[map['TOTAL_INSURANCE_ADJUSTMENTS']]
                insurancePayments = row[map['TOTAL_INSURANCE_PAYMENTS']]
                patientAdjustments = row[map['TOTAL_PATIENT_ADJUSTMENTS']]
                patientPayments =  row[map['TOTAL_PATIENT_PAYMENTS']]
                sql = "INSERT INTO hospital_costs (fk_patientID,fk_hospitalID,admissionDate,dischargeDate,totalCharges,insuranceAdjustments,insurancePayments,patientAdjustments,patientPayments) VALUES  (%s, %s, %s, %s, %s, %s, %s, %s, %s);"
                sqlutils.execMysqlQuery(sql, (str(patientID), hospitalID, admissionDate, dischargeDate, totalCharges,insuranceAdjustments,insurancePayments,patientAdjustments,patientPayments))

        counter = counter + 1  # Note - counter is only used to ignore the first row of the spreadsheet (contains column headings)


