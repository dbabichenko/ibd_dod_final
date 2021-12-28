import base64

from Crypto import Random
from Crypto.Cipher import AES
import csv, os
from utilities import stringutils, globals


#source_files = ['06-19-2017 OP Only 2017.csv']
mrn_dict = {}

mrn_source = "deid_IBD_Registry_Participant_List_2017-01-19.csv"
base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/"
filename = base_path + "MRN with linkers 11-24-2017/" + mrn_source


seq_id = 0
with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[1], globals.MAX_MRN_LENGTH)

            mrn_dict[mrn] = row[0]
            seq_id = seq_id + 1
            print(row[1] + ": " + str(row[0]))


