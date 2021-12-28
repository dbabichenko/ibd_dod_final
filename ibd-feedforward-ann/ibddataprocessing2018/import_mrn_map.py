import csv
import os
import uuid
import md5
import utilities.stringutils

from utilities import sqlutils, stringutils, globals



def readFileContents(filepath, filename):
    f = open(filepath,"r")
    contents = f.read()
    recordID = filename.replace(".txt", "")
    idx = recordID.index("_")
    fileMatchID = recordID[0:idx]
    print(fileMatchID)
    patientID = 0
    if fileMatchID.lower() in matches.keys():
        encryptedMrn = matches[fileMatchID.lower()]
        if encryptedMrn in patients.keys():
            patientID = patients[encryptedMrn]
            print(patientID)

    sql = "INSERT INTO patient_chart_notes (recordID, fileMatchID, fk_patientID, notes) VALUES (%s, %s, %s, %s);"
    sqlutils.execMysqlQuery(sql, (recordID, fileMatchID, patientID, contents))




#os.chdir(dir);

# Fill dictionaries

sql = "SELECT patientID, encryptedPatientID FROM patient;"
patients = sqlutils.tableToDictionary(sql, 1, 0)
matches = {}
filename = "cerner_ids.csv"
os.chdir(globals.ROOT_FOLDER);

with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[1], globals.MAX_MRN_LENGTH)
            print(mrn + " : " + row[0])
            matchID = str(row[0]).lower()
            matches[matchID] = stringutils.computeMD5hash(mrn)



dir = globals.CERNER_SOURCE_FOLDER
subdirs = [x[0] for x in os.walk(dir)]
for subdir in subdirs:
    files = os.walk(subdir).next()[2]
    if (len(files) > 0):
        for file in files:
            if str(file).find(".txt", 0) != -1:
                print(subdir + "/" + file)
                readFileContents(subdir + "/" + file, str(file))





#UPDATE procedure_costs a JOIN procedure_categories b ON a.procCode = b.procCode SET a.category = b.category;