import pymysql
import json

dbhost = 'localhost'
dbport = 3306
dbuser = 'root'
dbpasswd = ''
dbname = 'ibd2015dw'

def get_event_json(eventName, sql):
    conn = pymysql.connect(host=dbhost, port=dbport, user=dbuser, passwd=dbpasswd, db=dbname)
    cur = conn.cursor()
    cur.execute(sql)
    #print(cur.description)


    temp = ""
    for row in cur:
        if temp != "":
            temp += ","
        temp += '{"eventType": "' + str(row[0]) + '",'
        temp += '"eventName": "' + str(row[1]) + '",'
        temp += '"eventValue": "' + str(row[2]) + '",'
        temp += '"contactDate": "' + str(row[3]) + '",'
        temp += '"rangeLow": "' + str(row[4]) + '",'
        temp += '"rangeHigh": "' + str(row[5]) + '"}'

        json = '"' + eventName + '":[' + temp + ']'
    cur.close()
    conn.close()

    return(json)

def get_medication_json():
    sql = "SELECT 'Medication' as eventType, medName, pharmClass, SCRAMBLE_DATE(orderingDate), SCRAMBLE_DATE(startDate), SCRAMBLE_DATE(endDate) FROM medications WHERE fk_patientID IN ('4204') LIMIT 10;"
    conn = pymysql.connect(host=dbhost, port=dbport, user=dbuser, passwd=dbpasswd, db=dbname)
    cur = conn.cursor()
    cur.execute(sql)
    # print(cur.description)

    medList = [];

    temp = {}
    for row in cur:
        #temp["eventType"] = str(row[0])
        temp["medicationName"] = str(row[1])
        temp["pharmClass"] = str(row[2])
        temp["orderingDate"] = str(row[3])
        temp["startDate"] = str(row[4])
        temp["endDate"] = str(row[5])
        medList.append(temp)
    cur.close()
    conn.close()

    return medList


def get_labs_json():
    sql = "SELECT 'Lab' as eventType, labName, cptCode, labCompName, SCRAMBLE_DATE(orderDate), SCRAMBLE_DATE(resultDate), ordValue, refLow, refHigh, refUnit, resultFlag FROM labs WHERE fk_patientID IN ('4204') LIMIT 10;"
    conn = pymysql.connect(host=dbhost, port=dbport, user=dbuser, passwd=dbpasswd, db=dbname)
    cur = conn.cursor()
    cur.execute(sql)
    # print(cur.description)

    labList = [];

    temp = {}
    for row in cur:
        #temp["eventType"] = str(row[0])
        temp["labName"] = str(row[1])
        temp["cptCode"] = str(row[2])
        temp["labComponentName"] = str(row[3])
        temp["orderDate"] = str(row[4])
        temp["resultDate"] = str(row[5])
        temp["orderValue"] = str(row[6])
        temp["referenceLowRange"] = str(row[7])
        temp["referenceHighRange"] = str(row[8])
        temp["referenceUnit"] = str(row[9])
        temp["resultFlag"] = str(row[10])
        labList.append(temp)
    cur.close()
    conn.close()

    return labList

def get_encounters_json():
    sql = "SELECT 'Encounter' as eventType, encounterType, SCRAMBLE_DATE(contactDate), deptName FROM encounters  WHERE fk_patientID IN ('4204')  LIMIT 10;"
    conn = pymysql.connect(host=dbhost, port=dbport, user=dbuser, passwd=dbpasswd, db=dbname)
    cur = conn.cursor()
    cur.execute(sql)
    # print(cur.description)

    tempList = [];

    temp = {}
    for row in cur:
        #temp["eventType"] = str(row[0])
        temp["encounterType"] = str(row[1])
        temp["contactDate"] = str(row[2])
        temp["deptartmentName"] = str(row[3])
        tempList.append(temp)
    cur.close()
    conn.close()

    return tempList

def get_vitals_json():
    sql = "(SELECT 'Vital' as eventType, 'Height' as vitalName, height, SCRAMBLE_DATE(contactDate), 80 AS rangeLow, 120 as rangeHigh  FROM vitals WHERE fk_patientID IN ('4204') LIMIT 10) "
    sql += " UNION "
    sql += " (SELECT 'Vital' as eventType, 'Weight' as vitalName, weight, SCRAMBLE_DATE(contactDate), 80 AS rangeLow, 120 as rangeHigh  FROM vitals WHERE fk_patientID IN ('4204') LIMIT 10) "
    sql += " UNION "
    sql += " (SELECT 'Vital' as eventType, 'BMI' as vitalName, bmi, SCRAMBLE_DATE(contactDate), 80 AS rangeLow, 120 as rangeHigh  FROM vitals WHERE fk_patientID IN ('4204') LIMIT 10) "
    sql += " UNION "
    sql += " (SELECT 'Vital' as eventType, 'Pulse' as vitalName, pulse, SCRAMBLE_DATE(contactDate), 80 AS rangeLow, 120 as rangeHigh  FROM vitals WHERE fk_patientID IN ('4204') LIMIT 10) "
    sql += " UNION "
    sql += " (SELECT 'Vital' as eventType, 'Systolic BP' as vitalName, SUBSTR(bp, 1, INSTR(bp, '/') - 1) AS eventValue, SCRAMBLE_DATE(contactDate), 80 AS rangeLow, 120 as rangeHigh FROM vitals WHERE fk_patientID IN ('4204') LIMIT 10) "
    sql += " UNION "
    sql += " (SELECT 'Vital' as eventType, 'Diastolic BP' as vitalName, SUBSTR(bp, INSTR(bp, '/') + 1) AS eventValue, SCRAMBLE_DATE(contactDate), 80 AS rangeLow, 120 as rangeHigh FROM vitals WHERE fk_patientID IN ('4204') LIMIT 10)"
    conn = pymysql.connect(host=dbhost, port=dbport, user=dbuser, passwd=dbpasswd, db=dbname)
    cur = conn.cursor()
    cur.execute(sql)
    # print(cur.description)

    tempList = [];

    temp = {}
    for row in cur:
        #temp["eventType"] = str(row[0])
        temp["vitalName"] = str(row[1])
        temp["vitalValue"] = str(row[2])
        temp["contactDate"] = str(row[3])
        temp["rangeLow"] = str(row[4])
        temp["rangeHigh"] = str(row[5])
        tempList.append(temp)
    cur.close()
    conn.close()

    return tempList
'''
json = '{'

sql = "SELECT 'Vitals' as eventType, 'Heart Rate' AS eventName, pulse AS eventValue, SCRAMBLE_DATE(contactDate), 60 AS rangeLow, 100 as rangeHigh FROM vitals WHERE fk_patientID IN ('4204');"
pulse = get_event_json("pulse", sql)
if pulse != "":
    json += pulse

sql = "SELECT 'Vitals bp' as eventType, 'Systolic BP' AS eventName, SUBSTR(bp, 1, INSTR(bp, '/') - 1) AS eventValue, SCRAMBLE_DATE(contactDate), 80 AS rangeLow, 120 as rangeHigh FROM vitals WHERE fk_patientID IN ('4204');"
sbp = get_event_json("sbp", sql)
json += ',' + sbp

sql = "SELECT 'Vitals' as eventType, 'Diastolic BP' AS eventName, SUBSTR(bp, INSTR(bp, '/') + 1) AS eventValue, contactDate, SCRAMBLE_DATE(contactDate), 60 AS rangeLow, 80 as rangeHigh FROM vitals WHERE fk_patientID IN ('4204');"
dbp = get_event_json("dbp", sql)
json += ',' + dbp

sql = "SELECT 'Lab' as eventType, 'ALANINE AMINOTRANS(ALT)' AS eventName, ordNumValue AS eventValue, SCRAMBLE_DATE(orderDate), refLow, refHigh  FROM labs WHERE fk_patientID IN ('4204') AND labName = 'ALANINE AMINOTRANS(ALT)';"
lab = get_event_json("alanine_alt", sql)
json += ',' + lab

sql = "SELECT 'Lab' as eventType, 'ALKALINE PHOSPHATASE' AS eventName, ordNumValue AS eventValue, SCRAMBLE_DATE(orderDate), refLow, refHigh  FROM labs WHERE fk_patientID IN ('4204') AND labName = 'ALKALINE PHOSPHATASE';"
lab = get_event_json("ALKALINE_PHOSPHATASE", sql)
json += ',' + lab

json += '}'

print(json)

'''

jsonObj = {}
jsonObj["medications"] = get_medication_json()
jsonObj["labs"] = get_labs_json()
jsonObj["encounters"] = get_encounters_json()
jsonObj["vitals"] = get_vitals_json()


print(json.dumps(jsonObj))