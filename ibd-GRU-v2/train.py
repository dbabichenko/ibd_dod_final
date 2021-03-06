import torch 
from torch.utils.data import DataLoader
from data_utils import PatientDataset2
import gru
import utils
import argparse
import time, os, shutil
import pandas as pd, numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss
from scipy.special import softmax
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", help="model & logfile name")
parser.add_argument("-b", "--batch_size")
parser.add_argument("-e", "--epochs")
parser.add_argument("-T", "--OP_tgt", help="Name of OP target")
parser.add_argument("-A", "--AUX_tgts", help="Comma-separated names of Aux targets")
parser.add_argument('-a', '--aux_alpha', help='Weight [0,1) for Aux Loss')
parser.add_argument('-t', '--tr_alpha', help='Weight [0,1) for TR Loss')
parser.add_argument('-lr', '--learning_rate')
parser.add_argument('-d', '--datadir')
args = parser.parse_args()


print(f"\n\n\nTrain model with args {args}\n\n")	
aux_cols = args.AUX_tgts.split(',')


# Load Datasets
datadir=args.datadir
train_ds, valid_ds, test_ds = PatientDataset2(datadir, 'train'), PatientDataset2(datadir, 'valid'), PatientDataset2(datadir, 'test') 
X_Mean = utils.pkl_load(os.path.join('data', datadir, 'x_mean.pkl'))


# Balanced sampling
class_probab = [0.9, 0.037, 0.06] 
reciprocal_weights = [class_probab[train_ds[index][1][args.OP_tgt]] for index in range(len(train_ds))]
weights = (1 / torch.Tensor(reciprocal_weights))
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_ds))

train_iter = DataLoader(train_ds, batch_size=int(args.batch_size), drop_last=True, num_workers=2, sampler=sampler)
test_iter = DataLoader(test_ds, batch_size=len(test_ds), drop_last=True, num_workers=2)


t0 = time.time()
model = gru.train_model(train_iter, test_iter, X_Mean, args.OP_tgt, aux_cols, int(args.epochs), args.model_name, 3, float(args.learning_rate), float(args.aux_alpha), float(args.tr_alpha), class_weights=None, l2=0)
print(f'Time taken to train: {time.time()-t0}')
print(f"Saving model to {os.path.join('models', args.model_name, 'model.mdl')}")
torch.save(model.state_dict(), os.path.join('models', args.model_name, 'model.mdl'))

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())




