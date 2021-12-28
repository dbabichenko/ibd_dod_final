import base64

from Crypto import Random
from Crypto.Cipher import AES
import csv, os
from utilities import stringutils, globals

BS = 16
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
unpad = lambda s : s[0:-ord(s[-1])]


class AESCipher:

    def __init__( self, key ):
        self.key = key

    def encrypt( self, raw ):
        raw = pad(raw)
        iv = Random.new().read( AES.block_size )
        cipher = AES.new( self.key, AES.MODE_CBC, iv )
        return base64.b64encode( iv + cipher.encrypt( raw ) )

    def decrypt( self, enc ):
        enc = base64.b64decode(enc)
        iv = enc[:16]
        cipher = AES.new(self.key, AES.MODE_CBC, iv )
        return unpad(cipher.decrypt( enc[16:] ))


cipher = AESCipher('Ra1D5@A!w@ysRul3')
#encrypted = cipher.encrypt('Secret Message A')
#decrypted = cipher.decrypt(encrypted)
#print encrypted
#print decrypted

orig_filename = "SIBDQ_Pain_Questionaire_2017-01-19.csv"
base_path = "/Users/dmitriyb/Desktop/Data/ibd_did/"
filename = base_path + orig_filename
output_file = base_path + "deid_" + orig_filename
writer = csv.writer(open(output_file, 'w'))

with open(os.path.abspath(filename), 'rU') as f:
    for row in csv.reader(f):
        if any(row):
            mrn = stringutils.prefixZeros(row[0], globals.MAX_MRN_LENGTH)
            encryptedMrn = cipher.encrypt(mrn)
            row[0] = encryptedMrn
            #print(str(row).replace('[', '').replace('[', ''))
            writer.writerow(row)
            #print(row)


