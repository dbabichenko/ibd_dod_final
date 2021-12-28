__author__ = 'dmitriyb'
import csv, sys, os, uuid


filearray = ['lemann.csv']

csvsourcefolder = "/Users/dmitriyb/Dropbox/Research/IBD Research/pyimport/csv/"
sqltargetfolder = "/Users/dmitriyb/Dropbox/Research/IBD Research/pyimport/sql/"
targetFileName = sqltargetfolder + "lemannimport.sql"
targetFile = open(os.path.abspath(targetFileName), 'w')
for filename in filearray:
    os.chdir(csvsourcefolder);
    with open(os.path.abspath(filename), 'rU') as f:
        headers = csv.reader(f).next()
        #for x in range(1, 164):
        #    print(str(x) + ": " + headers[x]);

        i = 0
        for row in csv.reader(f):
            if any(row):
                if(i > 0):
                    for x in range(1, 164):
                        if("Total" not in headers[x]):
                            sql = "INSERT INTO ibd_normalized.lemann (patientID, item, score, sequence) VALUES "
                            sql += "((SELECT patientID FROM ibd_normalized.patient WHERE encryptedPatientID = MD5('" + row[0] + "')), '" + headers[x] + "', " + row[x] + ", " + str(x) + ");\n"
                            print(sql);
                            targetFile.write(sql)

                i = i + 1

