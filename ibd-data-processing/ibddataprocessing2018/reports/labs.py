import pymysql

from utilities import reportutils
from utilities import globals
from datetime import date

def labsReportPivotByCategoryByYear(categoryList, outputFileName, compareType):
    colList = ["Patient ID"]
    for yr in range(2001, date.today().year):
        colList.append("count" + str(yr))
        colList.append("abnormal" + str(yr))


    sql = "SELECT fk_patientID "
    for yr in range(2001, date.today().year):
        sql+= ", SUM(count" + str(yr) + ") count" + str(yr) + ", SUM(abnormal" + str(yr) + ") abnormal" + str(yr) + " "
    sql += "FROM "
    sql += "("
    sql += "SELECT fk_patientID "
    for yr in range(2001, date.today().year):
        sql += ", CASE WHEN YEAR(resultDate) = " + str(yr) + " THEN 1 ELSE 0 END count" + str(yr) + " "
        if compareType == 1:
            sql += ", CASE WHEN (YEAR(resultDate) = " + str(yr) + " AND (ordNumValue < refLow OR ordNumValue > refHigh)) THEN 1 ELSE 0 END abnormal" + str(yr) + " "
        elif compareType == 2:
            sql += ", CASE WHEN (YEAR(resultDate) = " + str(yr) + " AND ordNumValue > refLow) THEN 1 ELSE 0 END abnormal" + str(yr) + " "
    sql += "FROM labs "
    sql += "WHERE ordValue <> '' AND ordNumValue <> '9999999' "
    sql += "AND (labName IN (" + categoryList + ") "
    sql += "OR labCompName IN (" + categoryList + ")) "
    sql += ") t "
    sql += "GROUP BY fk_patientID order by fk_patientID; "

    print(sql)


    outputFilePath = globals.OUTPUT_FOLDER_PATH + outputFileName

    reportutils.spreadsheetFromQuery(colList, sql, outputFilePath)


def labsReportByCategory(categoryList, outputFileName, compareType):
    colList = ["Patient ID", "Gender", "Lab Name", "Lab Component Name", "Result Date", "Order Value", "Order Number Value", "Ref. Low", "Ref. High", "Ref. Unit", "Ref. Status"]


    sql = "SELECT fk_patientID, gender, labName, labCompName, resultDate, ordValue, ordNumValue, refLow, refHigh, refUnit, "
    if compareType == 1:
        sql += "CASE WHEN ordNumValue < refLow THEN 'low' WHEN ordNumValue > refHigh THEN 'high' ELSE 'normal' END as refStatus "
    elif compareType == 2:
        sql += "CASE WHEN ordNumValue <= refLow THEN 'normal' ELSE 'high' END as refStatus "
    sql += "FROM labs JOIN patient ON fk_patientID = patientID "
    sql += "WHERE ordValue <> '' and ordNumValue <> '9999999' "
    sql += "AND (labName IN (" + categoryList + ") "
    sql += "OR labCompName IN (" + categoryList + ")) "
    sql += "ORDER BY fk_patientID;"

    print(sql)

    outputFilePath = globals.OUTPUT_FOLDER_PATH + outputFileName

    reportutils.spreadsheetFromQuery(colList, sql, outputFilePath)


labCategories = "'ERYTHROCYTE SEDIMENTATION RATE', 'ESR', 'SEDRATE - AUTOMATED', 'WESTERGREN SED RATE', 'WESTERGREN ESR', 'SED RATE'"
labsReportByCategory(labCategories, 'Lab01-ERYTHROCYTE.csv', 1)
labsReportPivotByCategoryByYear(labCategories, 'Lab01-ERYTHROCYTE-Pivot.csv', 1);

labCategories = "'ALBUMIN'"
labsReportByCategory(labCategories, 'Lab02-ALBUMIN.csv', 1)
labsReportPivotByCategoryByYear(labCategories, 'Lab02-ALBUMIN-Pivot.csv', 1);

labCategories = "'C-REACTIVE PROTEIN','HIGH SENSITIVITY CRP','CRP QUANTITATION','CARDIO CRP','CRP, INFLAMMATORY','CRP TITER'"
labsReportByCategory(labCategories, 'Lab03-CRP.csv', 2)
labsReportPivotByCategoryByYear(labCategories, 'Lab03-CRP-Pivot.csv', 2);


labCategories = "'ABS EOSINOPHILS','AUTO. ABSOL. EOSIN','ABSOLUTE EOS'"
labsReportByCategory(labCategories, 'Lab04-EOSINOPHILS.csv', 1)
labsReportPivotByCategoryByYear(labCategories, 'Lab04-EOSINOPHILS-Pivot.csv', 1);

labCategories = "'PLATELETS','PLATELET COUNT'"
labsReportByCategory(labCategories, 'Lab05-PLATELETS.csv', 1)
labsReportPivotByCategoryByYear(labCategories, 'Lab05-PLATELETS-Pivot.csv', 1);

labCategories = "'HEMOGLOBIN','HGB','HEMOGLOBIN CAPILLARY (POC)','HEMOGLOBIN (POCT)','HEMOGLOBIN (BEDSIDE)'"
labsReportByCategory(labCategories, 'Lab06-HEMOGLOBIN.csv', 1)
labsReportPivotByCategoryByYear(labCategories, 'Lab06-HEMOGLOBIN-Pivot.csv', 1);

