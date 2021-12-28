import pymysql

from utilities import reportutils
from utilities import globals



def eosEncountersPivotByYear(outputFileName, compareSign = "<"):

    colListSql = "SELECT DISTINCT encounterType FROM encounters ORDER BY encounterType;"

    sql = "SELECT mrn, encounterType, COUNT(encounterType) AS encounterCount "
    sql+= "FROM ibd_research.eos_hospitalizations a "
    sql+= "JOIN ibd2015dw.patient b ON MD5(a.mrn) = b.encryptedPatientID "
    sql+= "JOIN ibd2015dw.encounters c ON b.patientID = c.fk_patientID "
    sql+= "WHERE contactDate " + compareSign + " eosDate "
    sql+= "GROUP BY mrn, eosDate, encounterType; "
    print(sql)
    outputFilePath = globals.OUTPUT_FOLDER_PATH + outputFileName

    reportutils.pivotFromQuery(colListSql, sql, outputFilePath)

def eosMedsPivotByYear(outputFileName, compareSign = "<"):

    colListSql = "SELECT DISTINCT pharmClass FROM medications ORDER BY pharmClass;"

    sql = "SELECT mrn, pharmClass, COUNT(pharmClass) AS pharmCount "
    sql+= "FROM ibd_research.eos_hospitalizations a "
    sql+= "JOIN ibd2015dw.patient b ON MD5(a.mrn) = b.encryptedPatientID "
    sql+= "JOIN ibd2015dw.medications c ON b.patientID = c.fk_patientID "
    sql+= "WHERE LEAST(LEAST(orderingDate, startDate), endDate) " + compareSign + " eosDate "
    sql+= "GROUP BY mrn, eosDate, pharmClass;  "
    print(sql)
    outputFilePath = globals.OUTPUT_FOLDER_PATH + outputFileName

    reportutils.pivotFromQuery(colListSql, sql, outputFilePath)


eosEncountersPivotByYear("Encounters_Before_EOS.csv", "<")
eosEncountersPivotByYear("Encounters_After_EOS.csv", ">=")
eosMedsPivotByYear("Medications_Before_EOS.csv", "<")
eosMedsPivotByYear("Medications_After_EOS.csv", ">=")