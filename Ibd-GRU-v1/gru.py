# **********************************
# 	Author: Suraj Subramanian
# 	2nd January 2020
# **********************************

import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_utils as du
import utils
import random
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from matplotlib.lines import Line2D

class GRUD(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, x_mean, aux_op_dims=[], use_decay=True, op_act=None):
        super(GRUD, self).__init__()
        self.device = utils.try_gpu()
        
        # Assign input and hidden dim
        mask_dim = delta_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_decay = use_decay
        self.x_mean = torch.tensor(x_mean, device=self.device).float()

        self.op_act = op_act or nn.LeakyReLU()
        
        # 2 FC Layers for Aux output (Single/Multi-label binary classification)
        self.aux_fc_layers=nn.ModuleList()
        for aux in aux_op_dims:
            self.aux_fc_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim//3),
                    nn.Linear(hidden_dim//3, aux),
                    self.op_act
                )
            )
            nn.init.kaiming_normal_(self.aux_fc_layers[-1][0].weight, 0.01)
            nn.init.kaiming_normal_(self.aux_fc_layers[-1][1].weight, 0.01)


        # Output - 2 FC layers
        self.fc_op = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//3),
            nn.Linear(hidden_dim//3, output_dim),
            self.op_act
        )
        nn.init.kaiming_normal_(self.fc_op[0].weight, 0.01)
        nn.init.kaiming_normal_(self.fc_op[1].weight, 0.01)
        

        
        # Linear Combinators
        if use_decay:
            self.R_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)
            self.Z_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)
            self.tilde_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)
            self.gamma_x_lin = nn.Linear(delta_dim, delta_dim)
            self.gamma_h_lin = nn.Linear(delta_dim, hidden_dim)
        else:
            self.R_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.Z_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.tilde_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.R_lin.weight)
        nn.init.xavier_normal_(self.Z_lin.weight)
        nn.init.xavier_normal_(self.tilde_lin.weight)
            
            
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, device=self.device, dtype=torch.float)
    

    def forward(self, inputs, hidden):
        inputs = inputs.float()
        batch_size = inputs.size(0)
        step_size = inputs.size(2)
        outputs = None
        tr_out = torch.zeros(batch_size, step_size, self.output_dim)
                
        # TR prediction
        for t in range(step_size):
            hidden = self.step(inputs[:,:,t:t+1,:], hidden)
            tr_out[:, t:t+1, :] = torch.unsqueeze(self.fc_op(hidden), 1)
            
        # OP prediction
        outputs = self.fc_op(hidden) # GRU -> FC -> FC -> RELU -> (Softmax)
        
        # Aux prediction
        aux_out = []
        for aux_layer in self.aux_fc_layers:
            a_o = aux_layer(hidden) # GRU -> FC -> FC -> RELU -> (Sigmoid)
            aux_out.append(a_o)

        return outputs, tr_out, aux_out, hidden


    def predict(self, inputs, hidden):
        step_size = inputs.size(2)
        outputs = None
            
        for t in range(step_size):
            hidden = self.step(inputs[:,:,t:t+1,:], hidden)
        
        outputs = self.fc_op(hidden)
        return outputs, hidden
        
    
    def step(self, inputs, h): # inputs = (batch_size x type x dim)
        inputs = inputs.float()
        x, obs_mask, delta, x_tm1 = torch.squeeze(inputs[:,0,:,:]), \
                            torch.squeeze(inputs[:,1,:,:]), \
                            torch.squeeze(inputs[:,2,:,:]), \
                            torch.squeeze(inputs[:,3,:,:])
        
        if self.use_decay:
            gamma_x = torch.exp(-torch.max(torch.zeros(delta.size(), device=self.device), self.gamma_x_lin(delta)))
            gamma_h = torch.exp(-torch.max(torch.zeros(h.size(), device=self.device), self.gamma_h_lin(delta)))
            x = (obs_mask * x) + (1-obs_mask)*( (gamma_x*x_tm1) + (1-gamma_x)*self.x_mean ) 
            h = torch.squeeze(gamma_h*h)
            
        gate_in = torch.cat((x,h,obs_mask), -1)
        z = torch.sigmoid(self.Z_lin(gate_in))
        r = torch.sigmoid(self.R_lin(gate_in))
        tilde_in = torch.cat((x, r*h, obs_mask), -1)
        tilde = torch.tanh(self.tilde_lin(tilde_in))
        h = (1-z)*h + z*tilde
        return h


class TrainPlot:
    def __init__(self, modelname):
        self.modelname = modelname
        self.call_count = 0
    
    def plot_grad_flow(self, named_parameters):
        '''
        https://github.com/alwynmathew/gradflow-check
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        
        plt.figure(0)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join('models', self.modelname, 'gradflow', f'{self.call_count}.png'), bbox_inches='tight')

        self.call_count +=1


def train_epoch(model, train_iter, tgt_col, aux_cols, criterion, optimizer, aux_alpha, tr_alpha,\
                scheduler=None, print_every=10, plotter=None):

    def get_aux_loss(aux_y_hat, aux_y):
        aux_criterion = nn.BCEWithLogitsLoss() # For multi-label classification BCELoss(sigmoid(logit))
        combined_aux_loss = 0
        for ix in range(len(aux_y)):
            truth = aux_y[ix].to(device)
            pred = aux_y_hat[ix]
            if len(truth.size())==1: truth = torch.unsqueeze(truth, 1)
            combined_aux_loss += aux_criterion(pred, truth)
        return combined_aux_loss

    device = utils.try_gpu()
    metrics = utils.Accumulator(5) #batch, loss, outputloss, trloss, auxloss
    loss_steps = []
    batch_size = train_iter.batch_size

    denom = 1+aux_alpha+tr_alpha
    loss_weights = [1/denom, aux_alpha/denom, tr_alpha/denom]

    for batch, (X, y_dict) in enumerate(train_iter):
        X = X.to(device)
        y = y_dict[tgt_col].to(device)   # OP target tensor      
        aux_y = [y_dict[ac] for ac in aux_cols] # List of Aux target tensors
        
        state = model.init_hidden(batch_size)
        y_hat, y_tr_hat, aux_y_hat, state = model(X, state)
        
        optimizer.zero_grad()   
        op_loss = criterion(y_hat, y) # Output Loss
        metrics.add(0,0,op_loss.item(),0,0)
        
        tr_loss, aux_loss = 0, 0
        if tr_alpha>0:
            seq_len = X.size(2)
            # Reshape replicated targets and predictions for loss compute. No linear scaling.
            y_tr = torch.unsqueeze(y, 1).repeat(1, seq_len).view(batch_size*seq_len)
            y_tr_hat = y_tr_hat.view(batch_size*seq_len, -1) # [batch_size*seq_len, C]
            # Get loss
            tr_loss = criterion(y_tr_hat.to(device), y_tr)
            metrics.add(0,0,0,tr_loss.item(),0)
        if aux_alpha>0:
            aux_loss = get_aux_loss(aux_y_hat, aux_y)
            metrics.add(0,0,0,0,aux_loss.item())
        
        loss = loss_weights[0]*op_loss + loss_weights[1]*aux_loss + loss_weights[2]*tr_loss # Weighted combination of OP, Aux, TR loss
        loss.backward()
        optimizer.step()
         
        metrics.add(1, loss.item(), 0, 0,0)
        loss_steps.append(loss.item())
        if batch%print_every == 0:
            plotter.plot_grad_flow(model.named_parameters())
            print(f"Minibatch:{batch}    OPLoss:{metrics[2]/metrics[0]}    TRLoss:{metrics[3]/metrics[0]}    AuxLoss:{metrics[4]/metrics[0]}    AggLoss:{metrics[1]/metrics[0]}        Examples seen: {metrics[0]*batch_size}")
        
    return metrics[1]/metrics[0]


    
def train_model(train_iter, valid_iter, X_Mean, tgt_col, aux_cols, epochs, modelname, nb_classes, \
                lr=0.001, aux_alpha=0, tr_alpha=0, class_weights=None, model=None, print_every=100):
    """
    Train a GRUD model

    :param train_iter: Train DataLoader
    :param valid_iter: Valid DataLoader
    :param X_Mean: Empirical Mean values for each dimension in the input (only important for variables with missing data)
    :param tgt_col: (str) Name of OP target
    :param aux_cols: list(str) of names of Aux targets. 
    :param epochs: Int of epochs to run
    :param modelname: Unique name for this model
    :param nb_classes: Number of OP target classes
    :param aux_alpha: Weight for Aux Loss
    :param tr_alpha: Weight for TR Loss
    :param class_weights (optional): Weights to scale OP Loss (for skewed datasets)
    """
    device = utils.try_gpu()
    try:
        os.makedirs(os.path.join('models',modelname))
    except FileExistsError:
        pass
    os.makedirs(os.path.join('models',modelname, 'gradflow'))
    plotter = TrainPlot(modelname)

    for X,y in train_iter: break
    input_dim = X.size(-1)
    hidden_dim = input_dim*6  # Arbitrary
    aux_dim = [ (y[aux_c].size(-1) if len(y[aux_c].size())>1 else 1) for aux_c in aux_cols] # if-else for targets with single dimennsion. their size(-1) will be batchsize
    op_act = nn.LeakyReLU()
    class_weights = class_weights or [1]*nb_classes 

    print(f"\n\n Training model with input_dim={input_dim}, hidden_dim={hidden_dim}. Aux dimensions={aux_dim}")

    model = GRUD(input_dim, hidden_dim, nb_classes, X_Mean, aux_dim, True, op_act).to(device)
    model.float()
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, verbose=True, threshold=0.001)

    train_meta = {}
    train_meta['loss_steps'] = []
    train_meta['valid_loss'] = []
    train_meta['min_valid_loss'] = sys.maxsize
    train_meta['best_model'] = None
    
    
    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_iter, tgt_col, aux_cols, criterion, optimizer, aux_alpha, tr_alpha, scheduler, print_every=print_every, plotter=plotter)
        # validation loss / early stopping
        retdict = eval_model(model, valid_iter, tgt_col, nb_classes)
        loss, acc, confmat, _ = retdict.values()       
        if epoch<125:scheduler.step()

        if loss < train_meta['min_valid_loss']:
            train_meta['min_valid_loss'] = loss
            train_meta['best_epoch'] = epoch+1
            train_meta['best_model'] = model.state_dict()
            train_meta['optimizer_state'] = optimizer.state_dict()
            utils.pkl_dump(train_meta, os.path.join('models', modelname, 'trainmeta.dict'))
            print(f"Checkpoint created  ValidLoss: {loss}    ValidAcc:{acc}          WallTime: {datetime.datetime.now()}\n CP::  {confmat}")

        print("\n\n================================================================================================================\n")
        print(f"Epoch: {epoch+1}    TrainLoss: {avg_loss}    ValidLoss: {loss}    ValidAcc:{acc}       WallTime: {datetime.datetime.now()}\n")
        print(confmat)
        print("\n\n================================================================================================================\n")
        train_meta['loss_steps'].append(avg_loss)
        train_meta['valid_loss'].append(loss)

        # Plot losses
        plt.figure(1)
        plt.plot(train_meta['loss_steps'])
        plt.plot(train_meta['valid_loss'])
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join('models', modelname, 'lossPlot.png'), bbox_inches='tight')
    return model



def predict(model, X,y, state):
    device = utils.try_gpu()
    yhat, _  = model.predict(X.to(device).float(), state.to(device).float()) # We don't need cell state
    softmaxed = torch.softmax(yhat, 1)
    _, pred_label = torch.max(softmaxed, 1)
    return yhat, pred_label
   

### TO-DO Find out why is this taking so long. How lazy is dataloader, does it slow down parallel model training?
def eval_model(model, test_iter, tgt_col, nb_classes):
    device = utils.try_gpu()
    test_loss = 0
    accuracy = 0
    loss_criterion = nn.CrossEntropyLoss()
    conf_matrix = torch.zeros(nb_classes, nb_classes, device=device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        for batch, (X,y_dict) in enumerate(test_iter):
            y = y_dict[tgt_col]
            state = model.init_hidden(test_iter.batch_size)
            yhat, labels = predict(model, X,y, state)
            test_loss += loss_criterion(yhat.to(device),y.to(device)).item()
            accuracy += (labels.to(device).long()==y.to(device).long()).float().mean()

            preds = torch.cat((torch.unsqueeze(y.to(device).float(), 1), 
                            torch.unsqueeze(labels.float(), 1),
                            torch.softmax(yhat,1).float()),  1).to('cpu')
            if nb_classes>1:
                for t,p in zip(y.view(-1), labels.view(-1)):
                    conf_matrix[t.long(), p.long()] += 1
    model.train()

    # TODO classification_report
    retdict = {'loss':test_loss/(batch+1), 'accuracy':accuracy/(batch+1), 'conf_matrix':conf_matrix.to('cpu'), 'preds':preds}
    return retdict
    
