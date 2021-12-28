import torch 
from torch.utils.data import DataLoader
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
parser.add_argument('-d', '--dataset', help='Dataset name')
parser.add_argument('-lr', '--learning_rate')
args = parser.parse_args()


print(f"\n\n\nTrain model with args {args}\n\n")	
aux_cols = args.AUX_tgts.split(',')

def get_results(test_ds, type):
    def data(ix):
        d = test_results.loc[ix:ix+32]
        d['0'] = 'TEST'
        d['0'].iloc[:24] = 'TRAIN'
        return d

    dl = DataLoader(test_ds, batch_size=len(test_ds))
    retdict = gru.eval_model(model, dl, args.OP_tgt, nb_tgt_classes)

    model_dir = os.path.join('models', args.modelname)
    try:
        os.makedirs(os.path.join(model_dir, f'{type}_eval_results'))
    except FileExistsError:
        shutil.rmtree(os.path.join(model_dir, f'{type}_eval_results'))
        os.makedirs(os.path.join(model_dir, f'{type}_eval_results'))
    
    # Pad predictions. logits to input sequences
    test_ids = pd.Series(test_ds.list_IDs)
    results = pd.DataFrame(np.array(retdict['preds']))
    results.columns = ['y', 'yhat'] +[f'logit{i}' for i in range(nb_tgt_classes)]
    results['seq_id'] = test_ids
    seq_data = pd.read_pickle('pickles/x_padded_inputs.pkl') 
    seq_data['seq_id'] = seq_data.AUTO_ID.astype(str)+'_'+seq_data.MONTH_TS.astype(str)
    test_results = results.merge(seq_data, how='outer', on='seq_id').sort_values('seq_id')
    test_results.index = range(len(test_results))

    # Save to file
    neg_class = 0
    pos_class = nb_tgt_classes-1
    test_results['tag'] = ''
    test_results.loc[(test_results['y']==pos_class) & (test_results['yhat']==neg_class), 'tag'] = 'fn'
    test_results.loc[(test_results.y==pos_class)&(test_results.yhat==pos_class), 'tag'] = 'tp'
    test_results.loc[(test_results.y==neg_class)&(test_results.yhat==neg_class), 'tag'] = 'tn'
    test_results.loc[(test_results.y==neg_class)&(test_results.yhat==pos_class), 'tag'] = 'fp'
    for i in ['fn', 'tp', 'tn', 'fp']:
        dfs = [data(ix) for ix in test_results[test_results.tag==i].index]
        pd.concat(dfs, axis=0).to_csv(f'{model_dir}/{type}_eval_results/{i}.csv')

    # Classification Report
    report = classification_report(results.y, results.yhat, labels=[0,1,2], target_names=['lt_7k', 'lt_53k', 'gt_53k'], output_dict=True)

    y_dum = pd.get_dummies(results.y).values
    y_hats = softmax(results[['logit0', 'logit1', 'logit2']].values, axis=0)

    # Multiclass ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    brier = dict()
    for i in range(nb_tgt_classes):
        fpr[i], tpr[i], _ = roc_curve(y_dum[:, i], y_hats[:, i])
        roc_auc[i] = round(auc(fpr[i], tpr[i]), 4)
    brier[0] = brier_score_loss(results.y.replace({1:2}), y_hats[:,0])
    brier[1] = brier_score_loss(results.y.replace({1:0}), y_hats[:,1])

    # Print
    with open(os.path.join(model_dir, 'report.txt'),'a') as f:
        print(f"\n\n======================== {type} ========================\n", file=f)
        print(retdict['conf_matrix'], file=f)
        print(f"\nMultilclass ROC:{roc_auc}\nLoss:{retdict['loss']}\n", file=f)
        print(f"Multiclass Brier Loss: {brier}\n", file=f)
        print(pd.DataFrame.from_dict(report), file=f)
    
    with open(os.path.join(model_dir, 'report.txt')) as f:
        return f.read()



# Load Datasets
train_ds = utils.pkl_load(os.path.join('datasets', args.dataset, 'train.dataset'))
valid_ds = utils.pkl_load(os.path.join('datasets',args.dataset, 'valid.dataset'))
test_ds = utils.pkl_load(os.path.join('datasets',args.dataset, 'test.dataset'))
X_Mean = utils.pkl_load(os.path.join('datasets', args.dataset, 'x_mean.data'))



meta_ds = utils.pkl_load(os.path.join('datasets',args.dataset, 'meta.data'))
class_sample_count = meta_ds['train_labels']
nb_tgt_classes = len(class_sample_count)
valid_ds_length = len(valid_ds)

class_probab = [0.8, 0.1, 0.1] 
reciprocal_weights = [class_probab[train_ds[index][1][args.OP_tgt]] for index in range(len(train_ds))]
weights = (1 / torch.Tensor(reciprocal_weights))
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_ds))
train_iter = DataLoader(train_ds, batch_size=int(args.batch_size), drop_last=True, num_workers=2, sampler=sampler)
valid_iter = DataLoader(valid_ds, batch_size=valid_ds_length, num_workers=2)


print(f"Loaded DataLoaders. Meta:{meta_ds}\nClass Counts: {class_sample_count}. Sampling proportionately in training.\n")
t0 = time.time()
model = gru.train_model(train_iter, valid_iter, X_Mean, args.OP_tgt, aux_cols, int(args.epochs), args.model_name, int(nb_tgt_classes), float(args.learning_rate), float(args.aux_alpha), float(args.tr_alpha), class_weights=None, l2=0)
print(f'Time taken to train: {time.time()-t0}')
print(f"Saving model to {os.path.join('models', args.model_name, 'model.mdl')}")
torch.save(model.state_dict(), os.path.join('models', args.model_name, 'model.mdl'))

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("\n\nEvaluating")
print("VALID\n=============================================")
print(get_results(valid_ds, 'valid'))

print("\n\nTEST\n=============================================")
print(get_results(test_ds, 'test'))