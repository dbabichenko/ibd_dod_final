import torch
import pprint
from torch.utils.data import DataLoader
from data_utils2 import PatientDataset2
import gru, utils, model
import argparse, os, shutil
import pandas as pd, numpy as np
from scipy.special import softmax
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modelname", help="model & logfile name")
parser.add_argument("-T", "--OP_tgt", help="Name of OP target")
parser.add_argument('-d', '--dataset', help='Dataset name')
args = parser.parse_args()


def get_results(test_ds, type):
    dl = DataLoader(test_ds, batch_size=len(test_ds), num_workers = 5)
    retdict = gru.eval_model(model, dl, args.OP_tgt, 3)

    model_dir = os.path.join('models', args.modelname)
    try:
        os.makedirs(os.path.join(model_dir, f'{type}_eval_results'))
    except:
        shutil.rmtree(os.path.join(model_dir, f'{type}_eval_results'))
        os.makedirs(os.path.join(model_dir, f'{type}_eval_results'))

    retdict['modelname'] = args.modelname

    # Print
    pprint.pprint(retdict)
    utils.pkl_dump(retdict, os.path.join('models', args.modelname,f'{type}_report.dict'))
    


datadir=args.dataset
valid_ds, test_ds = PatientDataset2(datadir, 'valid'), PatientDataset2(datadir, 'test') 
X_Mean = utils.pkl_load(os.path.join('data', datadir, 'x_mean.pkl'))


input_size = X_Mean.size
model = model.IBDModel(input_size, 3, X_Mean, [8,6,1])
model.load_state_dict(torch.load(os.path.join('models', args.modelname,'checkpoint.pt'))['state_dict'])

if os.path.exists(os.path.join('models', args.modelname, 'report.txt')):
    os.remove(os.path.join('models', args.modelname, 'report.txt'))

# print(get_results(valid_ds, 'valid'))
print(get_results(test_ds, 'test'))



