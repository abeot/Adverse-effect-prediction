"""
Last edited: 04-27-24
Author: Yingzi Bu, Albert Cao
Description: Functions for generating graphs and evaluating performance
"""

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, roc_auc_score
import math
import sklearn.metrics as metrics
import numpy as np
from mycolorpy import colorlist as mcp
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem

def smile_list_to_MACCS(smi_list):
    MACCS_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        maccs = list(MACCSkeys.GenMACCSKeys(mol).ToBitString())
        MACCS_list.append(maccs)
    return MACCS_list

def process(df):
    MACCS_list = smile_list_to_MACCS(df['SMILES'].tolist())
    header = ['bit' + str(i) for i in range(167)]
    new_df = pd.DataFrame(MACCS_list, columns=header)
    new_df['SMILES'] = df['SMILES']
    new_df = new_df.merge(df)
    return new_df

def pairwise_similarity(fp_list):
    num = len(fp_list)
    similarities = np.zeros((num, num))
    for i in range(num):
        similarity = DataStructs.BulkTanimotoSimilarity(
            fp_list[i], fp_list[i:])
        # print(type(similarity), len(similarity))
        similarities[i, i:] = similarity
        similarities[i:, i] = similarity
    for i in range(num): assert similarities[i, i] == 1
    return similarities


def plot_tanimoto_2(feature_list: list, title=None, savepath=None):
    similarities = pairwise_similarity(feature_list)
    fig = plt.figure(figsize = (8,8))
    heatmap = sns.heatmap(similarities, cmap='Blues', square=True)
    # Get the color bar axes and adjust its position and size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_aspect(20)  # Adjust this value as needed to match the figure size

    # Adjust the color bar's position and size
    cax = plt.gcf().axes[-1]
    cax.set_position([0.78, 0.1, 0.03, 0.8])  # Adjust these values as needed

    if title == None: title = 'Tanimoto Demo'
    plt.title(title, fontsize = 16)
    make_path('Tanimoto', False)
    if savepath == None: savepath = f'Tanimoto/{title}.png'
    plt.savefig(savepath, format='png', transparent=True)
    print('figure saved at ', savepath)
    plt.show(); plt.close()

def plot_tanimoto(df, title=None, savepath=None):
    smiles = df['SMILES']
    maccs_list = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_list.append(maccs)
    plot_tanimoto_2(maccs_list, title=title, savepath=savepath)

def plot_dim_reduced(mol_info, label, task_type, dim_reduct='PCA', title=None):
    """
    param mol_info: could be MACCS Fingerprint
    param label: label of data
    param task_type: [True, False], True:regression; False: classification
    param dim_reduct : ['PCA', 't-SNE']
    param title: None or string, the name of the plot
    Return figure.png saved at dim_reduct/title.png
    """
    features, labels = mol_info.copy(), label.copy()
    n_components = 2
    if dim_reduct == 'PCA':
        pca = PCA(n_components=n_components)
        pca.fit(features)
        features = StandardScaler().fit_transform(features)
        features = pd.DataFrame(data = pca.transform(features))
        ax_label = 'principle component'
    elif dim_reduct=='t-SNE':
        features = TSNE(n_components=n_components).fit_transform(features)
        features = MinMaxScaler().fit_transform(features)
        features = pd.DataFrame(np.transpose((features[:,0],features[:,1])))
        ax_label = 't-SNE'
    else: print("""Error! dim_reduct should be 'PCA' or 't-SNE'"""); return

    columns = [f'{ax_label} {i+1}' for i in range(n_components)]
    # features = pd.DataFrame(data = pca.transform(features), columns=columns)
    features.columns = columns
    features['label'] = labels

    sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots(figsize=(6, 6))
    f, ax = plt.subplots()
    custom_palette = sns.color_palette('husl', n_colors=6)
    param_dict = {'x': columns[0],
                'y': columns[1],
                'hue':'label',
                'palette': custom_palette,
                'data': features,
                's': 10,
                'ax':ax}

    # sns.despine(f, left=True, bottom=False)
    sns.scatterplot(**param_dict)

    if task_type == True: # regression task, color bar for labels
        norm = plt.Normalize(labels.min(), labels.max())
        scalarmap = plt.cm.ScalarMappable(cmap=param_dict['palette'], norm=norm)
        scalarmap.set_array([])
        ax.figure.colorbar(scalarmap)
        ax.get_legend().remove()
    else: sns.move_legend(ax, 'upper right') # for classification, label box

    ax = plt.gca()
    # Set the border or outline color and width
    border_color = 'black'
    border_width = 0.6  # Adjust this as needed

    # Add a rectangular border around the plot
    for i in ['top', 'right', 'bottom', 'left']: ax.spines[i].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(border_width); spine.set_color(border_color)
    # move the legend if has that:

    if title == None: title = f'{dim_reduct}_demo'
    plt.title(title); make_path(dim_reduct, False)
    plt.savefig(f'{dim_reduct}/{title}.png', format='png', transparent=True)
    print(f'figure saved at {dim_reduct}/{title}.png')
    # plt.show(); plt.close()


def get_morgan(df):
    smiles = df['SMILES']
    morgan_list = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        features = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048
        )
        morgan_list.append(features)
    return morgan_list

def plot_morgan(df, title=None, savepath=None):
    morgan_list = get_morgan(df)
    plot_tanimoto_2(morgan_list, title=title, savepath=savepath)

def get_min(d:dict):
    min_key = next(iter(d))
    for key in d:
        if d[key] < d[min_key]: min_key = key
    return min_key, d[min_key]

def plot_loss(train_dict, test_dict, name='test', title_name=None):
    fig = plt.figure()
    # fig.grid(False)
    plt.plot(list(train_dict.keys()), list(train_dict.values()), label='train')
    plt.plot(list(test_dict.keys()), list(test_dict.values()), label=name)
    argmin, min = get_min(test_dict)
    plt.plot(argmin, min, '*', label=f'min epoch {argmin}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if title_name == None: title_name = 'loss during training'
    plt.title(title_name); plt.grid(False)
    plt.legend()
    plt.show(); plt.close()

def make_path(path_name, verbose=True):
    import os
    if os.path.exists(path_name):
        if verbose: print('path:', path_name, 'already exists')
    else: os.makedirs(path_name); print('path:', path_name, 'is created')
    
def clean_files(path=None, file_types=['pth'], ver=False):
    import os
    from os import walk
    if path == None: path = os.getcwd()
    # delete all early stopping related files
    files = next(walk(path), (None, None, []))[2]
    for f in files:
        if isinstance(f, str):
            file_type = f.split('.')[-1]
            if file_type in file_types:
                file_here = path + '/' + f 
                os.remove(file_here)
                if ver: print(f'removed from {path}: {f}')
evaluate_names = ['ROC-AUC', 'PR-AUC']

def get_preds(thres, prob):
    try: 
        if prob.shape[1] == 2: prob = prob[:, 1]
    except: pass
    return [1 if p > thres else 0 for p in prob]

# AUC, AP figure generating
# code reference: 
# https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py

def roc_curve(y_pred, y_label, method_name, figure_title=None, figure_file=None):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)	
    fpr = dict()
    tpr = dict() 
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 2
    plt.plot(fpr[0], tpr[0],
            lw=lw, label= method_name + ' (area = %0.3f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    title = 'AUROC'
    if figure_title != None: title += ' on ' + figure_title + ' test set'
    plt.title(title, fontsize=fontsize)
    # plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    if figure_file != None: 
        plt.savefig(figure_file)
    plt.show(); plt.close()
    return 

# code reference: 
# https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py
def prc_curve(y_pred, y_label, method_name, figure_title=None, figure_file=None):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        reference: 
            https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    '''	
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
    #	plt.plot([0,1], [no_skill, no_skill], linestyle='--')
    label_name = ' (area = %0.3f)' % average_precision_score(y_label, y_pred)
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name+label_name)
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    # plt.title('Precision Recall Curve')
    title = 'PRAUC'
    if figure_title != None: title += ' on ' + figure_title + ' test set'
    plt.title(title, fontsize=fontsize)
    plt.xlim(0,1)
    plt.ylim(0,1)
    # plt.title('PRC', fontsize=fontsize)
    plt.legend()
    if figure_file != None:
        plt.savefig(figure_file)
    plt.show(); plt.close()
    return 

def reg_evaluate(label_clean, preds_clean):
    mae = metrics.mean_absolute_error(label_clean, preds_clean)
    mse = metrics.mean_squared_error(label_clean, preds_clean)
    rmse = np.sqrt(mse) #mse**(0.5)
    r2 = metrics.r2_score(label_clean, preds_clean)

    print('  MAE     MSE     RMSE    R2')
    print("&%5.3f" % (mae), " &%5.3f" % (mse), " &%5.3f" % (rmse),
      " &%5.3f" % (r2))
    # return r2, mae, rmse
    return mae, mse, rmse, r2

def evaluate(y_real, y_hat, y_prob): # for classification 
    # print('y_real', y_real)
    # print('y_hat', y_hat)
    try: 
        TN, FP, FN, TP = confusion_matrix(y_real, y_hat).ravel()

    except: # the label are all the same
        if y_real == y_hat: 
            if   y_real[0] == 1: TN, FP, FN, TP = 0, 0, 0, len(y_real)
            elif y_real[0] == 0: TN, FP, FN, TP = len(y_real), 0, 0, 0
    
    print(f'TN: {TN}; FP: {FP}; FN: {FN}; TP: {TP}')
    
    ACCURACY = (TP + TN) / (TP + FP + TN + FN)
    
    try: SE = TP / (TP + FN)
    except: SE = np.nan 
    recall = SE
    
    try: SP = TN / (TN + FP)
    except: SP = np.nan
    
    try: weighted_accuracy = (SE + SP) / 2
    except: weighted_accuracy = np.nan 

    try: precision = TP / (TP + FP)
    except: precision = np.nan

    try: F1 = 2 * precision * recall /(precision + recall)
    except: F1 = np.nan

    temp = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if temp != 0: MCC = (TP * TN - FP * FN) * 1.0 / (math.sqrt(temp))
    else: MCC = np.nan

    try:
        if y_prob.shape[1] == 2: proba = y_prob[:, 1]
        else: proba = y_prob
    except: proba = y_prob
    
    try: AP  = average_precision_score(y_real, proba)
    except: AP = np.nan
    try: AUC = roc_auc_score(y_real, proba)
    except: AUC = np.nan
    try: cohen = cohen_kappa_score(y_real, proba)
    except: cohen = np.nan
    # print(f'Accuracy, w_acc,   prec, recall/SE,   SP,   ',
    #       f'F1,     AUC,     MCC,     AP')
    print(f'  Acc,  w_acc,   prec,  recall,   SP,   ',
          f' F1,    AUC,   MCC,   AP')
    print("&%5.3f"%(ACCURACY), " &%5.3f"%(weighted_accuracy), 
          " &%5.3f"%(precision), " &%5.3f"%(SE), " &%5.3f"%(SP), 
    " &%5.3f"%(F1), "&%5.3f"%(AUC), "&%5.3f"%(MCC), "&%5.3f"%(AP))
    # print(type(F1))
    results_dict = {
        "acc": ACCURACY,
        "precision": precision,
        "recall": SE,
        "F1": F1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "cohen": cohen
    }
    return results_dict

def eval_dict(y_probs:dict, y_label:dict, IS_R, model_type='model',
              draw_fig=False, fig_title=None, fig_path=None, ae_name=""):
    """
    Return a dictionary of name: performance
    IS_R == True: regression task, returns R2
    IS_R == False: classific task, returns accuracy
    """
    name = ae_name 
    # IS_R = task_list[i]
    print('*'*15, name, '*'*15)
    # print('Regression task', IS_R)
    # print(y_probs)
    probs = y_probs
    label = y_label
    assert len(probs) == len(label)
    if IS_R == False: # classification task
        preds = get_preds(0.5, probs)
        cls_results = evaluate(label, preds, probs)
        if draw_fig:
            plt.grid(False)
            roc_curve(probs, label, model_type, figure_title=name, figure_file=f"roc_curves/{name}.png")
            prc_curve(probs, label, model_type, figure_title=name, figure_file=f"prc_curves/{name}.png")
        # performances[name] = float(cls_results[0]) # accuracy 
        performances = cls_results
        for key in performances:
            performances[key] = float(performances[key])

    else: # regression task
        mae, mse, rmse, r2 = reg_evaluate(label, probs)
        # performances[name] = float(r2) # r2 
        performances[name]=[float(mae), float(mse), float(rmse), float(r2)]
        if draw_fig:
            plt.grid(False)
            color = mcp.gen_color_normalized(cmap='viridis', data_arr=label)
            plt.scatter(label, probs, cmap='viridis', marker='.',
                        s=10, alpha=0.5, edgecolors='none', c=color)
            plt.xlabel(f'True value'); plt.ylabel(f'Predicted value')
            if fig_title == None: 
                title = f'{name} test set performance of {model_type}'
            else: title = f'{name} {fig_title}'
            plt.title(title)
            x0, xmax = plt.xlim();  y0, ymax = plt.ylim()
            data_width = xmax - x0; data_height = ymax - y0
            # print(x0, xmax, y0, ymax, data_width, data_height)
            r2   = f'R2:     {r2:.3f}'
            mae  = f'MAE:   {mae:.3f}'
            rmse = f'RMSE: {rmse:.3f}'
            plt.text(x0 + 0.1*data_width, y0 + data_height * 0.8/0.95, r2)
            plt.text(x0 + 0.1*data_width, y0 + data_height * 0.8,  mae)
            plt.text(x0 + 0.1*data_width, y0 + data_height * 0.8*0.95, rmse)
            if fig_path != None: # save figure at fig_path
                make_path(fig_path, False); 
                fig_name = f'{fig_path}/{title}.png'
                plt.savefig(fig_name, format='png', transparent=False)

            plt.show(); plt.cla(); plt.clf(); plt.close()
    print()
    return performances

