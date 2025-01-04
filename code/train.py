"""
Last edited: 06-30-24
Author: Albert Cao, Yingzi Bu
Description: Code for generating final models and evaluating performance
"""
import pandas as pd
from utils import *
from ml_utils import *
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from dgllife.utils import EarlyStopping
import torch.optim as optim
from sklearn.model_selection import KFold
import pickle


## CONSTANTS
lr = 1e-5 # learning rate, try 1e-5
wd = 1e-5 # weight decay try 1e-5
best_epoch = 0
MAX_EPOCH = 300

# BELOW IS NOT CHANGEABLE
in_dim = 215
k_folds = 5
patience = 15
verbose_freq = 100 # print out results every 10 epochs
batch_size = 64

best_cohen_dict  = {
    'diarrhoea': 0.2,
    'dizziness': 0.2,
    'headache': 0.2,
    'nausea': 0.2,
    'vomiting': 0.2
}

# h_dims = [800, 512, 216, 128, 64]
model_file = 'best_models'
with open(f'{model_file}/h_dims.pkl', 'rb') as f: h_dims = pickle.load(f)
    
aes = [
    'diarrhoea',
    'dizziness',
    'headache',
    'nausea',
    'vomiting'
]

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        try:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        except:
            print(f'cannot normalize {feature_name}')
    return result


def get_data(ae_name, negative_sampling=None):
    # df = pd.read_csv(f'data_normalized_{ae_name}.csv')
    df = pd.read_csv(f'{ae_name}_new.csv')
    df = df.drop(columns=['drug', 'SMILES'])
    # df = normalize(df)
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=5, test_size=0.15, random_state=42)
    for train_index, test_index in split.split(df, df[ae_name]):
        train_df = df.loc[train_index]
        test_df = df.loc[test_index]

    if negative_sampling != None:
        counts = train_df[ae_name].value_counts()
        print('previous:', counts)
        count_0 = counts[0]
        count_1 = counts[1]
        negative_df = train_df.loc[train_df[ae_name] == 0]
        while count_0 < count_1*negative_sampling:
            train_df = pd.concat([train_df, negative_df], ignore_index=True)
            counts = train_df[ae_name].value_counts()
            count_0 = counts[0]
            count_1 = counts[1]
        print('After adding negative samples', counts)
    return df, train_df, test_df

# gpu version of function: train_epoch, train
# slight modification, different from version in ml_utils.py
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
        ic, ae = ic.to(device), ae.to(device)
        mask = ae == MASK
        mask = mask.to(device)
        pred = model(ic)
        
        loss = loss_func(pred, ae, weight_loss)

        if train_type != 'Train': # valid or test, output probs and labels
            probs = pred.cpu().detach().numpy().tolist()
            label = ae.cpu().detach().numpy().tolist()
            # print(probs, type(probs), label, type(label))
            if isinstance(probs, float): probs = [probs]
            if isinstance(label, float): label = [label]
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
        # performance = float(total_loss)

    if   train_type == 'Train': return total_loss, y_probs, y_label
    elif train_type == 'Valid': return total_loss, y_probs, y_label
    else: return performance, y_probs, y_label

def eval(model, loader, path=None, ae_name="", device='cpu'):
    if path != None: load_model(model, path)
    performance, probs, label = train_epoch(model, loader, device=device, ae_name=ae_name)
    return performance, probs, label

def train(model, data_loader, val_loader, test_loader=None, weight_loss=None,
          ver_freq=verbose_freq, optimizer=None, ae_name="", device='cuda', model_path=None):
    train_dict = {}
    valid_dict = {}
    min_loss = np.inf
    if model_path == None: model_path = f'test_{ae_name}.pt'

    ### MOD:
    # stopper was put inside function train instead of outside.
    # Otherwise you are using the same stopper for 5-fold validation
    stopper = EarlyStopping(mode='lower', patience=patience)

    for epoch in range(best_epoch, MAX_EPOCH):
        score, _, _ = train_epoch(model, data_loader, epoch=epoch,
                optimizer=optimizer, weight_loss=weight_loss, device=device)
        val_score, probs, labels = train_epoch(model, val_loader,
                epoch=epoch, weight_loss=weight_loss, device=device)
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
    # clean_files() # delete all .pth files, use with caution

    plot_loss(train_dict, valid_dict, name='valid',
            title_name="loss during training MLP")

    if test_loader != None:
        performance, _, _ = eval(model, test_loader, model_path,
                                 device=device, ae_name=ae_name)
        return performance

def get_max(d:dict):
    max_key = next(iter(d))
    for key in d:
        if d[key] > d[max_key]: max_key = key
    return max_key, d[max_key]

def batch_train(ae_name, model_file, k_folds,
                min_weight, max_weight, weight_interval,
                negative_sampling=None, best_cohen=0.15, batch_size=64):

    _, train_df, test_df = get_data(ae_name, negative_sampling=negative_sampling)

    with open(f'{model_file}/h_dims.pkl', 'rb') as f: h_dims = pickle.load(f)

    temp_min = int(min_weight/weight_interval)
    temp_max = int(max_weight/weight_interval)
    kf = KFold(n_splits=k_folds, shuffle=True)

    model_path = f'{model_file}/test_{ae_name}.pt'

    train_dataset = tox_dataset(train_df, ae_name)
    params = {'batch_size':batch_size, 'shuffle':False,
              'drop_last':False, 'num_workers': 0}
    test_loader = DataLoader(tox_dataset(test_df, ae_name), **params)

    result_dict = {}
    for idx_here in range(temp_min, temp_max):
        print('\n\n')
        weight_loss_here = idx_here * weight_interval
        print('*'*40, weight_loss_here, '*'*40)

        result_list = []

        for repeat_time in range(3):

            results = {
                "acc": [],
                "precision": [],
                "recall": [],
                "F1": [],
                "TP": [],
                "TN": [],
                "FP": [],
                "FN": [],
                "cohen": []
            }
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
                print(f"Fold {fold + 1}/{k_folds}, batch_size={batch_size},", end="")
                print(f"negative sampling={negative_sampling}, weight {weight_loss_here}, {ae_name}, {model_file}")
                print("-------")

                stopper = EarlyStopping(mode='lower', patience=patience)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                )
                val_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(val_idx),
                )

                model = Classifier(in_dim, h_dims)
                if torch.cuda.is_available(): model = model.cuda()
                optimizer = optim.AdamW(params=model.parameters(),
                                        lr=lr, weight_decay=wd)

                cls_results = train(model, train_loader, val_loader, test_loader,
                        weight_loss=weight_loss_here, optimizer=optimizer,
                        ae_name=ae_name, device='cuda', model_path=model_path)

                cohen_here = cls_results['cohen']
                if cohen_here > best_cohen:
                    try: file_old_path = file_new_path
                    except: file_old_path = 'not_find_this_file'
                    print('save model, best cohen by now =', cohen_here)
                    file_new_path = f'{model_file}/{ae_name}_cohen_{cohen_here}.pt'
                    torch.save(model.state_dict(), file_new_path)
                    import os
                    file_path = f'{ae_name}_cohen_{best_cohen}.pt'
                    if os.path. exists(file_old_path):
                        os.remove(file_old_path)
                        print(f"The file {file_old_path} has been deleted.")
                    best_cohen = cohen_here

                for key in cls_results:
                    results[key].append(cls_results[key])

            for key in results:
                results[key].append(np.mean(results[key]))

            results_df = pd.DataFrame.from_dict(results)
            cohen_here = results_df['cohen'].tolist()[-1]
            result_list.append(cohen_here)
            if cohen_here == 0: break

        result_dict[weight_loss_here] = sum(result_list) / len(result_list)
        if result_dict[weight_loss_here] < 0: break

    return result_dict

### TRAIN 
for i in tqdm(range(9,1,-1), desc=f'current'):
    max_weight = min(int(10/i) + 3, 9)
    min_weight = max(min(max(int(10/i) - 3, 1), max_weight-4), 0.2)
    # print('min max:', min_weight, max_weig)ht)
    for _, ae_name in tqdm(enumerate(['headache', 'nausea',
                                      'vomiting', 'dizziness',
                                      'diarrhoea']),
                           total=5, desc=f'current ae, i= {i}'):

        result_dict = batch_train(ae_name, model_file, k_folds,
                            min_weight=min_weight, max_weight=max_weight,
                            weight_interval=0.2,
                            negative_sampling=0.1*i, best_cohen=0.2)


### Evaluation 

# model_folder = 'best_models'
# ae_name = 'dizziness'
# _, _, test_df = get_data(ae_name)
# test_loader = DataLoader(tox_dataset(test_df, ae_name), **params)
# with open(f'{model_folder}/h_dims.pkl', 'rb') as f: h_dims = pickle.load(f)
# model = Classifier(in_dim, h_dims)
# if torch.cuda.is_available(): model = model.cuda()
# model_path_best_cohen = '/content/drive/MyDrive/Adverse-effect-prediction-main/best_models/dizziness_cohen_0.3109869646182495.pt'
# _ = eval(model, test_loader, path=model_path_best_cohen, device='cuda')


# model_folder = 'best_models'
# ae_name = 'headache'
# _, _, test_df = get_data(ae_name)
# test_loader = DataLoader(tox_dataset(test_df, ae_name), **params)
# with open(f'{model_folder}/h_dims.pkl', 'rb') as f: h_dims = pickle.load(f)
# model = Classifier(in_dim, h_dims)
# if torch.cuda.is_available(): model = model.cuda()
# model_path_best_cohen = '/content/drive/MyDrive/Adverse-effect-prediction-main/best_models/headache_cohen_0.4108761329305136.pt'
# _ = eval(model, test_loader, path=model_path_best_cohen, device='cuda')


# model_folder = 'best_models'
# ae_name = 'nausea'
# _, _, test_df = get_data(ae_name)
# test_loader = DataLoader(tox_dataset(test_df, ae_name), **params)
# with open(f'{model_folder}/h_dims.pkl', 'rb') as f: h_dims = pickle.load(f)
# model = Classifier(in_dim, h_dims)
# if torch.cuda.is_available(): model = model.cuda()
# model_path_best_cohen = '/content/drive/MyDrive/Adverse-effect-prediction-main/best_models/nausea_cohen_0.4134078212290503.pt'
# _ = eval(model, test_loader, path=model_path_best_cohen, device='cuda')


# model_folder = 'best_models'
# ae_name = 'vomiting'
# _, _, test_df = get_data(ae_name)
# test_loader = DataLoader(tox_dataset(test_df, ae_name), **params)
# with open(f'{model_folder}/h_dims.pkl', 'rb') as f: h_dims = pickle.load(f)
# model = Classifier(in_dim, h_dims)
# if torch.cuda.is_available(): model = model.cuda()
# model_path_best_cohen = '/content/drive/MyDrive/Adverse-effect-prediction-main/best_models/vomiting_cohen_0.29735234215885953.pt'
# _ = eval(model, test_loader, path=model_path_best_cohen, device='cuda')

