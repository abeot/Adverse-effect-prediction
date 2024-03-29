import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pickle
import numpy as np
from dgllife.utils import EarlyStopping
from utils import *

class Classifier(nn.Module):
    def __init__(self, in_dim, h_dims:list):
        super(Classifier, self).__init__()

        neurons = [in_dim, *h_dims]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) \
                         for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        # self.emb = nn.GRU(h_dims[-1], h_dims[-1])
        self.final = nn.Linear(h_dims[-1], 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        # x = torch.squeeze(self.output(self.final(x)))
        x = torch.squeeze(self.output(self.final(x)))
        return x

class tox_dataset(Dataset):
    def __init__(self, df, ae_name):
        self.len = len(df)
        self.df = df
        self.ic_start_ind = df.columns.get_loc("appendix endocrine cells")
        self.ae_start_ind = df.columns.get_loc(ae_name)

    def __getitem__(self, idx):
        """
        OUTPUT
        :param fp: fingerprint, should be 167 dim
        :param ic: drug tissue concentration
        :param ae: adverse events
        """
        # header = ['bit' + str(i) for i in range(167)]
        # fp = self.df[header]
        # fp = torch.tensor([float(b) for b in fp.iloc[idx]], dtype=torch.float32)
        ic = self.df.iloc[:, self.ic_start_ind:self.ae_start_ind]
        ic = torch.tensor(ic.values.astype(np.float32))[idx]
        ae = self.df.iloc[:, self.ae_start_ind:]
        ae = torch.tensor(ae.values.astype(np.float32))[idx]
        # ae = onehot(5)(ae) # use onehot 
        # return fp, ic, ae.float()
        return ic, ae.float()
    def __len__(self): return self.len

def loss_func(output, target, weight):  

    target = target.to(dtype=torch.float32)
    
    output.requires_grad_(True)
    target.requires_grad_(True)
    
    log_output = torch.log(torch.clamp(output, min=1e-10, max=1.0 - 1e-10))
    log_1_minus_output = torch.log(torch.clamp(1 - output, 
                                    min=1e-10, max=1.0 - 1e-10))

    loss = -torch.sum(target * log_output + \
            weight * (1 - target) * log_1_minus_output)

    return loss

def train_epoch(model, loader, device='cpu', epoch=None, optimizer=None,
                MASK=-100, model_type='MLP', weight_loss=None, ver=False, ae_name=""):
    if optimizer==None: # no optimizer, either valid or test
        model.eval()
        if epoch != None: train_type = 'Valid'
        else: train_type = 'Test'
    else: model.train(); train_type='Train'

    if weight_loss == None: 
        weight_loss = 1.0
    total_loss, y_probs, y_label = 0, {}, {}
    
    for idx, batch_data in enumerate(loader): 
        # fp, ic, ae = batch_data
        # fp, ic, ae = fp.to(device), ic.to(device), ae.to(device)
        ic, ae = batch_data
        mask = ae == MASK 
        mask = mask.to(device)
        # pred = model(torch.cat((fp, ic), 1)) 
        pred = model(ic)
        # print('pred', pred)

        loss = loss_func(pred, ae, weight_loss)
        
        if train_type != 'Train': # valid or test, output probs and labels
            probs = pred.cpu().detach().numpy().tolist()
            label = ae.cpu().detach().numpy().tolist()

            if idx == 0: y_probs, y_label = probs, label
            else: y_probs += probs; y_label += label
        
        total_loss += loss.item() # sum up all loss for all AE in this batch
        # print(total_loss)
        if optimizer != None: 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    total_loss /= len(loader)
    if epoch != None: # train or valid
        if ver: print(f'Epoch:{epoch}, [{train_type}] loss: {total_loss:.3f}')
    elif epoch == None: # test
        if ver: print(f'[{train_type}] loss: {total_loss:.3f}')
        # print(y_probs, y_label)
        performance = eval_dict(y_probs, y_label, False, 
                                'MLP', draw_fig=True, ae_name=ae_name)
        performance = float(total_loss)

    if   train_type == 'Train': return total_loss, y_probs, y_label
    elif train_type == 'Valid': return total_loss, y_probs, y_label
    else: return performance, y_probs, y_label

def load_model(model, path, device='cpu'):
    print('load model from path: ', path)
    model.load_state_dict(torch.load(path, map_location=device))

def eval(model, loader, path=None):
    if path != None: load_model(model, path)
    performance, probs, label = train_epoch(model, loader)
    return performance, probs, label

lr = 1e-5 # learning rate, try 1e-5
wd = 1e-5 # weight decay try 1e-5
best_epoch = 0
MAX_EPOCH = 300
model_path = 'test.pt'
patience = 30
stopper = EarlyStopping(mode='lower', patience=patience)
verbose_freq = 10 # print out results every 10 epochs

def train(model, data_loader, val_loader, test_loader=None, weight_loss=None, 
          ver_freq=verbose_freq, optimizer=None, ae_name=""):
    train_dict = {}
    valid_dict = {}
    min_loss = np.inf
    for epoch in range(best_epoch, MAX_EPOCH): 
        score, _, _ = train_epoch(model, data_loader, epoch=epoch, 
                optimizer=optimizer, weight_loss=weight_loss)
        val_score, probs, labels = train_epoch(model, val_loader, 
                epoch=epoch, weight_loss=weight_loss)
        print(f'Epoch:{epoch} [Train] Loss:{score:.3f} | ',
              f'[Valid] Loss: {val_score:.3f}', end='\t')
        train_dict[epoch] = score
        valid_dict[epoch] = val_score
        
        early_stop = stopper.step(val_score, model)
        if val_score < min_loss: 
            print(f'SAVE MODEL: loss drop: {min_loss:.3f} -> {val_score:.3f}')
            min_loss = val_score
            torch.save(model.state_dict(), model_path)
        
        if epoch % ver_freq == 0 and epoch != 0:
            plot_loss(train_dict, valid_dict, name='valid', 
            title_name="loss during training MLP")
            eval_dict(probs, labels, False, 'MLP', ae_name=ae_name)
        
        if early_stop: print('early stop'); break
    
    print('Finished training \n')
    clean_files() # delete all .pth files, use with caution
    
    plot_loss(train_dict, valid_dict, name='valid', 
            title_name="loss during training MLP")
    
    if test_loader != None: 
        eval(model, test_loader, model_path)
# train_epoch(model, train_loader, AEs)
# p, a, b = train_epoch(model, test_loader, AEs,device='cpu', epoch=1, optimizer=optimizer,
#                 MASK=-100, model_type='MLP', weight_loss=None, ver=False)