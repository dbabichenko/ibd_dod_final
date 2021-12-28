import base64

from Crypto import Random
from Crypto.Cipher import AES
import csv, os
from utilities import stringutils, globals


source_files = ['SIBDQ.csv', 'HBI- UCAI.csv']
mrn_dict = {}

mrn_source = "deid_IBD_Registry_Participant_List_2017-01-19.csv"
base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/"
filename = base_path + "MRN with linkers 11-24-2017/" + mrn_source


with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[1], globals.MAX_MRN_LENGTH)

            mrn_dict[mrn] = row[0]
            # print(row[1] + ": " + str(row[0]))



for sf in source_files:
    filename = base_path + sf
    output_filename = base_path + "deid_" + sf
    writer = csv.writer(open(output_filename, 'w'))

    data_file = open(os.path.abspath(filename), 'rU')
    for row in csv.reader(data_file):
        if any(row):
            mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
            if mrn in mrn_dict.keys():
                row.insert(0, mrn_dict[mrn])
            else:
                row.insert(0, "NO_MATCH")

            writer.writerow(row)