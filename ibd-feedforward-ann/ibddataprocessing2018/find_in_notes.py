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
    #print(fileMatchID)
    patientID = 0
    mrn = ""

    if fileMatchID.lower() in matches.keys():
        encryptedMrn = matches[fileMatchID.lower()]
        mrn = mrns[fileMatchID.lower()]
        #print(mrn)
        contents = contents.lower()
        if (contents.find("difficile", 0) != -1 or contents.find("c.diff", 0) != -1) and contents.find("not detected") != -1:
            return mrn
        else:
            return "not found"








#os.chdir(dir);

# Fill dictionaries

sql = "SELECT patientID, encryptedPatientID FROM patient;"
patients = sqlutils.tableToDictionary(sql, 1, 0)
matches = {}
mrns = {}
filename = "cerner_ids.csv"
os.chdir(globals.ROOT_FOLDER);

with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[1], globals.MAX_MRN_LENGTH)
            #print(mrn + " : " + row[0])
            matchID = str(row[0]).lower()
            matches[matchID] = stringutils.computeMD5hash(mrn)
            mrns[matchID] = mrn



dir = globals.CERNER_SOURCE_FOLDER
cnt = 0
subdirs = [x[0] for x in os.walk(dir)]
for subdir in subdirs:
    files = os.walk(subdir).next()[2]
    if (len(files) > 0):
        for file in files:
            if str(file).find(".txt", 0) != -1:
                #print(subdir + "/" + file)
                found = readFileContents(subdir + "/" + file, str(file))
                if found != "not found" and found != None:
                    cnt = cnt + 1
                    print(str(cnt) + ": " + found)




