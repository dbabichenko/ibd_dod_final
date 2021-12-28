#http://stackoverflow.com/questions/4576077/python-split-text-on-sentences
#https://www.robincamille.com/2012-02-18-nltk-sentence-tokenizer/
#http://www.nltk.org/data.html

import nltk.data
import os
import unicodedata
import datetime



negation_terms = ['r/o', 'ro', 'rule out', 'negative', 'no evidence', 'no']

search_terms = [
    ['high', 'grade', 'dysplasia']
    ,['low', 'grade', 'dysplasia']
    ,['mild', 'dysplasia']
    ,['history', 'of', 'dysplasia']
    ,['positive', 'dysplasia']
    ]

potential_positives = []
actual_positives = []
matched_mrns = []

dir_path = '/Users/dmitriyb/Desktop/ibd_deidentified/pathology/out/'
output_path = dir_path + "processed/"
log_file_path = output_path + "log_" + str(datetime.datetime.now()).replace('-', '_').replace(' ', '_').replace(':', '_').replace('.', '_') + ".csv"


if os.path.isdir(output_path) == False:
    os.makedirs(output_path)

log_file = open(log_file_path, "w")


def match_potential_positives(sentence):
    sentence = sentence.lower()

    for term_list in search_terms:
        #print(term_list)
        if all(word in sentence for word in term_list):
            potential_positives.append(sentence)

def match_actual_positives(filename, sentence):

    match_found = False
    for phrase in negation_terms:
        if sentence.find(phrase) != -1:
            match_found = True
    if match_found == False:
        actual_positives.append(sentence)
        log_file.write(filename.replace('.txt', '_') + ', ' + sentence.replace(', ', ' ').replace('\n', '') + os.linesep)
        if filename not in matched_mrns:
            matched_mrns.append(filename)



def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])




#filelist = ['842351700.txt']
for filename in os.listdir(dir_path):
    filepath = dir_path + filename
    #print(filepath)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    if os.path.isfile(filepath) == True:
        fp = open(filepath)
        data = fp.read()
        data = remove_non_ascii(data)
        sentences = tokenizer.tokenize(data)
        #print '\n-----\n'.join(tokenizer.tokenize(data))

        # Clear lists
        del potential_positives[:]
        del actual_positives[:]

        for sentence in sentences:
            match_potential_positives(sentence)

        for sentence in potential_positives:
            match_actual_positives(filename, sentence)

        #if len(actual_positives) > 0:
            #print(actual_positives)

log_file.close()

print(len(matched_mrns))
