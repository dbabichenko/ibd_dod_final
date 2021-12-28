import datetime, hashlib, time, json
from dateutil.parser import parse
import re

def stringToDate(datestring):
    arr = datestring.split("/")
    finaldate = ""
    i = 0
    for i in range(0, len(arr)):
        if i < len(arr) - 1:
            arr[i] = "0" + str(arr[i])
            arr[i] = arr[i][-2:]

        finaldate += arr[i] + "/"
    return finaldate[:-1]



def computeMD5hash(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()

def yearFromStringDate(datestring):
    #datestring = stringToDate(datestring)
    #dt = time.strptime(datestring.strip(), '%m/%d/%Y')
    arr = datestring.split("/")
    return arr[2]


def getJsonMapping(year, mappingName):
    jsonFileName = 'mappings/' + str(year) + '/' + mappingName + '.json'
    json_data=open(jsonFileName)
    data = json.load(json_data)
    json_data.close()

    return data

def convertDateToMySQL(strDate):
    print(strDate)
    try:
        return str(parse(strDate).strftime("%Y-%m-%d"))
    except ValueError:
        return ""


def convertDateTimeToMySQL(strDateTime):
    try:
        return str(parse(strDateTime).strftime("%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return ""


def stripNonNumeric(strNum):
    return re.sub("[^0-9]", "", strNum)


def prefixZeros(input, maxLen):
    prefix = ""
    input = str(input).strip();
    for i in range(0, maxLen - len(input)):
        prefix = prefix + "0"

    return prefix + input


def clean_num_vals(val):
    val = val.replace(",", "")
    if val == "":
        return -99
    else:
        temp = val.split("=")
        return temp[0]