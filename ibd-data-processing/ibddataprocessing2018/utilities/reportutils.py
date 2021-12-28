__author__ = 'dmitriyb'

import sqlutils
import pymysql

def pivotFromQuery(colListSQL, dataSQL, outputFilePath):
    # I am making an assumption that every pivot table will use
    # patientID (MRN) as it's key column.  Might need to change this later
    columns = ['mrn']

    # Get a list of columns for the report
    columns = sqlutils.queryToList(colListSQL, columns)

    # Get data set to pivot
    data = sqlutils.queryToDataset(dataSQL, 0)

    # Prepare output file
    f = open(outputFilePath,'w')
    strRow = str(columns)
    f.write(strRow.replace('[', '').replace(']', '') + '\n')

    # Pivoting data directly on the database level is very inefficient.  It is easier and faster
    # to copy desired data set into a Python dictionaries and pivot it on Python level.
    d = {} # blank data set

    # Pivot data set from the database and store pivoted data in a temp data set d.
    for mrn in data:
        row = [0] * len(columns) # This will initialize all cells to 0 (zero)
        rowIndex = columns.index(data[mrn][1])
        row[0] = data[mrn][0] # Patient's MRN
        row[rowIndex] = data[mrn][2]
        d[data[mrn][0]] = row

    # Write results to file
    for key in d:
        strRow = str(d[key])
        f.write(strRow.replace('[', '').replace(']', '') + '\n')

    f.close()


def spreadsheetFromQuery(colList, dataSQL, outputFilePath):

    # Prepare output file
    f = open(outputFilePath,'w')
    strRow = str(colList)
    f.write(strRow.replace('[', '').replace(']', '') + '\n')



    # Write results to file
    conn = pymysql.connect(host=sqlutils.dbhost, port=sqlutils.dbport, user=sqlutils.dbuser, passwd=sqlutils.dbpasswd, db=sqlutils.dbname)
    cur = conn.cursor()
    cur.execute(dataSQL)



    for row in cur:
        strRow = ""
        for i in range(0, len(row)):
            if strRow == "":
                strRow = str(row[i])
            else:
                strRow += "," + str(row[i])
        strRow += "\n"
        f.write(strRow)

    cur.close()
    conn.close()

    f.close()