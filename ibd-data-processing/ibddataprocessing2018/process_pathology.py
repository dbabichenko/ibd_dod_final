import csv
import utilities.sqlutils
import hashlib
from hashlib import md5
import os


path1 = "/Users/dmitriyb/Dropbox/Research/IBD Research/pathology/path1.csv"
path2 = "/Users/dmitriyb/Dropbox/Research/IBD Research/pathology/path2.csv"

outputpath = "/Users/dmitriyb/Dropbox/Research/IBD Research/pathology/out/"

path_dict = {}


def process_rows(reader):
    cnt = 0
    for row in reader:
        if cnt > 0:
            mrn = str(row[0])
            if mrn in path_dict.keys():
                path_dict[mrn] = path_dict[mrn] + row[1] + os.linesep
            else:
                path_dict[mrn] = row[1] + os.linesep
        cnt = cnt + 1


with open(path1, 'rU') as f:
    reader = csv.reader(f)
    process_rows(reader)

with open(path2, 'rU') as f:
    reader = csv.reader(f)
    process_rows(reader)


#print(path_dict["841083101"])
#print(len(path_dict))
for k, v in path_dict.items():
    fo = open(outputpath + k + ".txt", "w")
    fo.write(v)
    fo.close()
    #sql = "INSERT INTO ibd_research.pathology VALUES (%s,%s);"
    #utilities.sqlutils.execMysqlQuery(sql, (k, v))
