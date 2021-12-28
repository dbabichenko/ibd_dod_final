import torch
import torch.nn as nn
import utils, os
import pandas as pd, numpy as np
import gru
from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss, precision_recall_curve
pd.options.mode.chained_assignment = None


# Hyper Parameters 
input_size = 52
hidden_size = 312
num_classes = 3
num_epochs = 50
batch_size = 128
learning_rate = 0.0001

class BaselineDataset(torch.utils.data.Dataset):  
    def __init__(self, datadir, type): 
        self.datadir = os.path.join('data',datadir+'_baseline', type) # will contain train/test/valid and {tr,v,t}_label_dict.pkl
        self.label_dict = utils.pkl_load(os.path.join(self.datadir,'label_dict.pkl'))
        self.list_IDs = list(self.label_dict.keys())

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = utils.pkl_load(os.path.join(self.datadir, ID+'.npy')) # Numpy seq
        y = self.label_dict[ID]['annual_charges'] 
        return X, y

class BaselineDataset2(torch.utils.data.Dataset):  
    def __init__(self, datadir, list_IDs): 
        self.datadir = os.path.join('data',datadir+'_baseline')
        self.list_IDs = list_IDs
        self.label_dict = utils.pkl_load(os.path.join(self.datadir,'annual_charges_label_dict.pkl'))
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = utils.pkl_load(os.path.join(self.datadir, 'sequences', ID+'.npy')) # Numpy seq
        y = self.label_dict[ID] # Dict of available target values
        return X, y





if __name__=="__main__":

    datadir = 'm24p8_nolap'
    train = BaselineDataset(datadir, 'train')
    valid = BaselineDataset(datadir, 'valid')
    test = BaselineDataset(datadir, 'test')

    # datadir = 'm24p8'
    # train = BaselineDataset2(datadir, utils.pkl_load('datasets/intra60_split/train.dataset').list_IDs)
    # valid = BaselineDataset2(datadir, utils.pkl_load('datasets/intra60_split/valid.dataset').list_IDs)
    # test = BaselineDataset2(datadir, utils.pkl_load('datasets/intra60_split/test.dataset').list_IDs)

    print("loading all class probab stuff")
    class_probab = [0.9, 0.07, 0.03] 
    reciprocal_weights = list(train.label_dict.values())
    # reciprocal_weights = [class_probab[i] for i in reciprocal_weights]  #olap
    reciprocal_weights = [class_probab[dic['annual_charges']] for dic in reciprocal_weights] # nolap
    weights = (1 / torch.Tensor(reciprocal_weights))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train))


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test, 
                                            batch_size=len(test), 
                                            shuffle=False, num_workers=2)

    # Neural Network Model (1 hidden layer)
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Net, self).__init__()
            self.hidden_dim = hidden_size
            self.op_act = nn.LeakyReLU()
            self.mlp = nn.Sequential(
                nn.Linear(input_size, self.hidden_dim),
                self.op_act,
                nn.Linear(self.hidden_dim, self.hidden_dim//9),
                nn.Dropout(0.3),
                self.op_act,
                nn.Linear(self.hidden_dim//9, 3)
            )
            self.logistic = nn.Linear(input_size, 3)

        def forward(self, x):
            out = self.mlp(x)
            # out = self.logistic(x)
            return out
        
    net = Net(input_size, hidden_size, num_classes).cuda()
        
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

    # Train the Model
    print("starting to train")
    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(train_loader):  
            if y.unique(return_counts=True)[1][1]>0:
                print(f"=", end='')
            # print(y.unique(return_counts=True))


            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(x.float().cuda())
            loss = criterion(outputs, y.cuda())
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, len(train)//batch_size, loss.item()))

    # Test the Model
    correct = 0
    total = 0
    test_loss = 0
    accuracy=0
    conf_matrix = torch.zeros(3, 3)
    for batch, (x,y) in enumerate(test_loader):
        yhat = net(x.float().cuda()).cpu()
        softmaxed = torch.softmax(yhat, 1)
        _, labels = torch.max(softmaxed, 1)
        test_loss += criterion(yhat,y).item()
        accuracy += (labels.long()==y.long()).float().mean()

        preds = torch.cat((torch.unsqueeze(y.float(), 1), 
                        torch.unsqueeze(labels.float(), 1),
                        torch.softmax(yhat,1).float()),  1)
        
        for t,p in zip(y.view(-1), labels.view(-1)):
            conf_matrix[t.long(), p.long()] += 1

    conf_matrix = conf_matrix.detach()
    accuracy = (accuracy/(batch+1)).item()
    q = conf_matrix.sum(0)/conf_matrix.sum()
    p = conf_matrix.sum(1)/conf_matrix.sum()
    pe = sum(p*q).item()
    kappa = (accuracy-pe)/(1-pe)

    eval_scores = {'loss':test_loss/(batch+1), 'accuracy':accuracy, 'conf_matrix':conf_matrix.tolist(), 'kappa':kappa}
    eval_scores.update(gru.eval_report(preds.detach()))
    utils.pkl_dump(eval_scores, os.path.join('models', 'baselines', f'{datadir}_nolap_report.dict'))


