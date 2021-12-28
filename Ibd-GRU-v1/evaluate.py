from torch.utils.data import DataLoader
import gru, utils
import argparse, os, shutil
import pandas as pd, numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss
from scipy.special import softmax
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modelname", help="model & logfile name")
parser.add_argument("-T", "--OP_tgt", help="Name of OP target")
parser.add_argument('-d', '--dataset', help='Dataset name')
args = parser.parse_args()


def get_results(test_ds, type):
    def data(ix):
        d = test_results.loc[ix:ix+32]
        d['0'] = 'TEST'
        d['0'].iloc[:24] = 'TRAIN'
        return d

    dl = DataLoader(test_ds, batch_size=len(test_ds), num_workers = 5)
    retdict = gru.eval_model(model, dl, args.OP_tgt, 3)

    model_dir = os.path.join('models', args.modelname)
    try:
        os.makedirs(os.path.join(model_dir, f'{type}_eval_results'))
    except:
        shutil.rmtree(os.path.join(model_dir, f'{type}_eval_results'))
        os.makedirs(os.path.join(model_dir, f'{type}_eval_results'))
    
    # Pad predictions. logits to input sequences
    test_ids = pd.Series(test_ds.list_IDs)
    results = pd.DataFrame(np.array(retdict['preds']))
    results.columns = ['y', 'yhat'] +[f'logit0', 'logit1', 'logit2']
    results['seq_id'] = test_ids

    seq_data = pd.read_pickle('pickles/x_padded_inputs.pkl') 
    seq_data['seq_id'] = seq_data.AUTO_ID.astype(str)+'_'+seq_data.MONTH_TS.astype(str)
    test_results = results.merge(seq_data, how='outer', on='seq_id').sort_values('seq_id')
    test_results.index = range(len(test_results))

    # Save to file
    neg_class = 0
    pos_class = 2
    test_results['tag'] = ''
    test_results.loc[(test_results['y']==pos_class) & (test_results['yhat']==neg_class), 'tag'] = 'fn'
    test_results.loc[(test_results.y==pos_class)&(test_results.yhat==pos_class), 'tag'] = 'tp'
    test_results.loc[(test_results.y==neg_class)&(test_results.yhat==neg_class), 'tag'] = 'tn'
    test_results.loc[(test_results.y==neg_class)&(test_results.yhat==pos_class), 'tag'] = 'fp'
    for i in ['fn', 'tp', 'tn', 'fp']:
        dfs = [data(ix) for ix in test_results[test_results.tag==i].index]
        pd.concat(dfs, axis=0).to_csv(f'{model_dir}/{type}_eval_results/{i}.csv')

    # Classification Report
    report = classification_report(results.y, results.yhat, labels=[0,1,2], target_names=['Low', 'Mid', 'High'], output_dict=True)

    # Multiclass ROC
    y_dum = pd.get_dummies(results.y).values
    y_hats = results[['logit0','logit1', 'logit2']].values    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    brier = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_dum[:, i], y_hats[:, i])
        roc_auc[i] = round(auc(fpr[i], tpr[i]), 4)

    # Pos vs Neg Brier Loss
    bin_results = results[results.y!=1]
    y_hats_bin = bin_results[['logit0', 'logit2']].values
    brier[0] = brier_score_loss((1-bin_results.y).abs(), y_hats_bin[:,0])
    brier[2] = brier_score_loss(bin_results.y, y_hats_bin[:,1])

    # Print
    with open(os.path.join(model_dir, 'report.txt'),'a') as f:
        print(f"\n\n======================== {type} ========================\n", file=f)
        print(retdict['conf_matrix'], file=f)
        print(f"\nMultilclass ROC:{roc_auc}\nLoss:{retdict['loss']}\n", file=f)
        print(f"Multiclass Brier Loss: {brier}\n", file=f)
        print(pd.DataFrame.from_dict(report), file=f)
    

print(f"\n\n\nEvaluate latest checkpoint with args {args}\n\n")
valid_ds = utils.pkl_load(os.path.join('datasets',args.dataset, 'valid.dataset'))
test_ds = utils.pkl_load(os.path.join('datasets',args.dataset, 'test.dataset'))
X_Mean = utils.pkl_load(os.path.join('datasets', args.dataset, 'x_mean.data'))
nb_tgt_classes = len(utils.pkl_load(os.path.join('datasets',args.dataset, 'meta.data'))['train_labels'])
model = gru.GRUD(53, 53*6, nb_tgt_classes, X_Mean, [8,6,1])
model.load_state_dict(utils.pkl_load(os.path.join('models', args.modelname,'trainmeta.dict'))['best_model'])

os.remove(os.path.join('models', args.modelname, 'report.txt'))

print(get_results(valid_ds, 'valid'))
print(get_results(test_ds, 'test'))



