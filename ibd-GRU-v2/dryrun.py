import torch 	
from torch.utils.data import DataLoader
import data_utils as du
import gru
import utils
import argparse
import time, os

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size")
parser.add_argument("-T", "--OP_tgt", help="Name of OP target") #annual_charges
parser.add_argument("-A", "--AUX_tgts", help="Comma-separated names of Aux targets") # abnormal_labs,diagnostics,surgery
parser.add_argument("-c", "--nb_tgt_classes", help="No. of target classes")
parser.add_argument('-a', '--aux_alpha', help='Weight [0,1) for Aux Loss')
parser.add_argument('-t', '--tr_alpha', help='Weight [0,1) for TR Loss')
parser.add_argument('-d', '--dataset', help='Dataset name')
parser.add_argument('-e', '--epochs', help='epochs')
args = parser.parse_args()


print(f"\n\n\nDry run model with args {args}\n\n")	


valid_iter = DataLoader(utils.pkl_load(os.path.join('datasets', args.dataset, 'valid.dataset')), shuffle=True, batch_size=int(args.batch_size), drop_last=True)
test_iter = DataLoader(utils.pkl_load(os.path.join('datasets', args.dataset, 'test.dataset')), shuffle=True, batch_size=int(args.batch_size), drop_last=True)
print("Loaded DataLoaders")
t0 = time.time()
aux_cols = args.AUX_tgts.split(',')
X_Mean = utils.pkl_load(os.path.join('datasets', args.dataset, 'x_mean.data'))

model = gru.train_model(valid_iter, test_iter, X_Mean, args.OP_tgt, aux_cols, int(args.epochs), 'test_model', int(args.nb_tgt_classes), float(args.aux_alpha), float(args.tr_alpha), print_every=10)

print(f'Time taken to train: {time.time()-t0}')
print(f"Saving model to {os.path.join('models', 'test_model', 'model.mdl')}")
torch.save(model.state_dict(), os.path.join('models', 'test_model', 'model.mdl'))

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())



print("\n\n===================================================================\n\n")



print("Evaluating")
v, te = utils.pkl_load(os.path.join('datasets',args.dataset, 'valid.dataset')), utils.pkl_load(os.path.join('datasets',args.dataset, 'test.dataset'))
print("VALID\n=============================================")
dl = DataLoader(v, batch_size=len(v))
retdict = gru.eval_model(model, dl, args.OP_tgt, int(args.nb_tgt_classes))
for metric,val in retdict.items():
    print(f"{metric}: {val}\n---------------------------------------------")

print("TEST\n=============================================")
dl = DataLoader(te, batch_size=len(te))
retdict = gru.eval_model(model, dl, args.OP_tgt, int(args.nb_tgt_classes))
for metric,val in retdict.items():
    print(f"{metric}: {val}\n---------------------------------------------")