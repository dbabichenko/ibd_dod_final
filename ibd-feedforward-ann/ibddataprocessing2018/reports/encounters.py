import pymysql

from utilities import reportutils
from utilities import globals



def encountersReportPivotByYear(categoryList, outputFileName, subcategoryClause = ""):

    colListSql = "SELECT DISTINCT YEAR(contactDate) FROM encounters WHERE encounterType IN (" + categoryList + ") ORDER BY YEAR(contactDate);"

    sql = "SELECT fk_patientID, YEAR(contactDate), COUNT(*)  "
    sql += "FROM encounters WHERE encounterType IN (" + categoryList + ") " + subcategoryClause + " "
    sql += "GROUP BY fk_patientID, YEAR(contactDate) "
    sql += "ORDER BY fk_patientID, YEAR(contactDate); "
    print(sql)
    outputFilePath = globals.OUTPUT_FOLDER_PATH + outputFileName

    reportutils.pivotFromQuery(colListSql, sql, outputFilePath)


encountersReportPivotByYear("'Telephone'", 'PhoneEncountersByYear.csv')
encountersReportPivotByYear("'ER Report'", 'ErVisitsByYear.csv')
encountersReportPivotByYear("'Discharge Summary'", 'DischargeSummaryByYear.csv')


def encountersReportByCategory(categoryList, outputFileName, subcategoryClause = ""):
    colList = ["Patient ID", "Encounter Type Name", "Contact Date", "Dept. ID", "Dept. Name"]
    sql = "SELECT fk_patientID, encounterType, DATE_FORMAT(contactDate, '%Y-%m-%d %T') AS contactDate, deptID, deptName "
    sql += "FROM encounters "
    sql += "WHERE encounterType IN (" + categoryList + ") " + subcategoryClause + "; ";
    print(sql)

    outputFilePath = globals.OUTPUT_FOLDER_PATH + outputFileName

    reportutils.spreadsheetFromQuery(colList, sql, outputFilePath)


encountersReportByCategory("'Telephone'", 'PhoneEncountersAll.csv')
encountersReportByCategory("'ER Report'", 'ErVisitsAll.csv')
encountersReportByCategory("'Discharge Summary'", 'DischargeSummaryAll.csv')

encountersReportByCategory("'New Patient Visit'", "NewPatientVisit_GAS_HBC_OAKLAND.csv", "AND deptID = 1045102")
encountersReportByCategory("'New Patient Visit'", "NewPatientVisit_COLON_RECTAL_DDC.csv", "AND deptID IN (10440009, 10168109)")
encountersReportByCategory("'New Patient Visit'", "NewPatientVisit_OTHER.csv", "AND deptID NOT IN (1045102, 10440009, 10168109)")

encountersReportByCategory("'Office Visit','Procedure Visit'", "OtherVisit_GAS_HBC_OAKLAND.csv", "AND deptID = 1045102")
encountersReportByCategory("'Office Visit','Procedure Visit'", "OtherVisit_COLON_RECTAL_DDC.csv", "AND deptID IN (10440009, 10168109)")
encountersReportByCategory("'Office Visit','Procedure Visit'", "OtherVisit_OTHER.csv", "AND deptID NOT IN (1045102, 10440009, 10168109)")

