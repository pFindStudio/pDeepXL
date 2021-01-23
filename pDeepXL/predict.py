# from .model import BiLSTMLinearPredictor # 相对路径导入
from pDeepXL.model import BiLSTMLinearPredictor # 绝对路径导入，两者都可以。
import pDeepXL.utils as utils

import configparser
import pickle
from tqdm import tqdm
import numpy as np
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnn_utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = configparser.ConfigParser()
import pkg_resources # https://stackoverflow.com/a/16892992
path_config = pkg_resources.resource_filename('pDeepXL', 'configs/config.ini')
# if os.path.isfile(path_config):
#     print('config file exist')
# else:
#     print('config file not exist')
config.read(path_config, encoding='UTF-8')

MIN_PREC_CHARGE=int(config['DEFAULT']['min_prec_charge'])
MAX_PREC_CHARGE=int(config['DEFAULT']['max_prec_charge'])
MIN_PEPTIDE_LEN=int(config['DEFAULT']['min_peptide_len'])
MAX_PEPTIDE_LEN=int(config['DEFAULT']['max_peptide_len'])

from pDeepXL.featurer import PeptideFeaturer
pf = PeptideFeaturer(config)


def GetPeptideFeatures(CSMs, consider_xlink_ion):
    T,L,X=[],[],[] # L表示总长度
    PT=[] # 肽段类型peptide type，0 for linear, 1 for non-clv, 2 for clv
    L1,L2,S1,S2=[],[],[],[] # α和β的长度，及交联位点，如果是单肽（P=0），则这几个值都是-1（无效）

    cur_pt=1 if consider_xlink_ion else 2

    num_invalid=0
    for csm in tqdm(CSMs):
        title,scan,prec_charge,instrument,NCE_low, NCE_medium, NCE_high,LinkerName,seq1,mods1,linksite1,seq2,mods2,linksite2=csm
        l1,l2 = len(seq1),len(seq2)

        if not (prec_charge >= MIN_PREC_CHARGE and prec_charge <= MAX_PREC_CHARGE \
                and l1 >= MIN_PEPTIDE_LEN and l1 <= MAX_PEPTIDE_LEN \
                and l2 >= MIN_PEPTIDE_LEN and l2 <= MAX_PEPTIDE_LEN): # 电荷与长度的约束
            num_invalid+=1
            continue

        seq1_vec,seq1_aa_mod_sum = pf.Sequence2Vec(seq1,mods1,prec_charge,instrument,NCE_low, NCE_medium, NCE_high,LinkerName,linksite1,False)
        seq2_vec,seq2_aa_mod_sum = pf.Sequence2Vec(seq2,mods2,prec_charge,instrument,NCE_low, NCE_medium, NCE_high,LinkerName,linksite2,False)

        # 对于不可碎裂交联剂，累加和需要考虑互补肽段
        if consider_xlink_ion:
            seq1_vec = pf.AddAnotherSeq(seq1_vec, linksite1, seq2_aa_mod_sum)
            seq2_vec = pf.AddAnotherSeq(seq2_vec, linksite2, seq1_aa_mod_sum)

        linker_symbol_vec = [pf.LinkerSymbol2Vec()]

        X.append(seq1_vec+linker_symbol_vec+seq2_vec)
        L.append(l1-1+l2-1+1)
        T.append(title)

        PT.append(cur_pt)
        L1.append(l1-1)
        L2.append(l2-1)
        S1.append(linksite1)
        S2.append(linksite2)

    print('collected %d valid samples, discard %d invalid samples'%(len(T),num_invalid))
    return T,PT,L,L1,L2,S1,S2,X


def LoadPredictDataSet(BATCH_SIZE, consider_xlink_ion, path_data_file):
    CSMs,mpTitleLines,header = utils.ReadpLink2PredictFile(path_data_file)
    T,PT,L,L1,L2,S1,S2,X=GetPeptideFeatures(CSMs, consider_xlink_ion)
    predict_data_loader=utils.ConvertPredictDataToDataLoader(BATCH_SIZE, X, T, PT, L1, L2, S1, S2)
    return predict_data_loader,mpTitleLines,header


def PredictMS2(model, iterator):
    model.eval()
    mpPeaks={}
    mpSpecInfo={}

    for batch_x, batch_length,batch_title,batch_pt,batch_l1,batch_l2,batch_s1,batch_s2 in iterator:

        batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, batch_length, batch_first=True)
        batch_x_pack = batch_x_pack.to(device)
        masked_y_pred,out_len = model.work(batch_x_pack)

        for title, pred, lseq, pt,l1,l2,s1,s2 in zip(batch_title,masked_y_pred,out_len, batch_pt,batch_l1,batch_l2,batch_s1,batch_s2):
            peaks=[]
            for i in range(lseq):
                peaks.append(pred[i].cpu().detach().numpy())
            peaks=np.array(peaks)
            peaks=np.transpose(peaks)
            mpPeaks[title]=peaks
            mpSpecInfo[title]=[pt,l1,l2,s1,s2]
    return mpPeaks, mpSpecInfo



def get_model(is_non_cleavable):
    INPUT_DIM = 130
    HIDDEN_DIM = 256
    OUTPUT_DIM = 8
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5

    path_non_clv_model = pkg_resources.resource_filename('pDeepXL', 'pt/non_clv_model.pt')
    path_clv_model = pkg_resources.resource_filename('pDeepXL', 'pt/clv_model.pt')
    path_model=path_non_clv_model if is_non_cleavable else path_clv_model

    print('loading pretrained model...')
    model = BiLSTMLinearPredictor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    model.load_state_dict(torch.load(path_model,map_location=device))
    model = model.to(device)

    return model


# ----- single predict -----------

def predict_single(prec_charge,instrument,NCE_low, NCE_medium, NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2):
    
    if crosslinker=='DSSO' or crosslinker=='DSBU':
        is_non_cleavable=False
    elif crosslinker=='DSS' or crosslinker=='Leiker':
        is_non_cleavable=True
    else:
        print('sorry, pDeepXL does not support %s cross-linker'%crosslinker)
        return

    #--- preprocess data ---
    title='anonymous.dta'
    scan='0'
    single_csm=[title,scan,prec_charge,instrument,NCE_low, NCE_medium, NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2]
    CSMs=[single_csm]

    T,PT,L,L1,L2,S1,S2,X=GetPeptideFeatures(CSMs, is_non_cleavable)
    predict_data_loader=utils.ConvertPredictDataToDataLoader(1, X, T, PT, L1, L2, S1, S2) # batchsize=1
    #----------------------

    model = get_model(is_non_cleavable)

    print('predicting...')
    mpPeaks, mpSpecInfo=PredictMS2(model, predict_data_loader)
    print('prediction done')
    
    # 把预测输出格式转换为plot的输入格式
    peaks = mpPeaks[title]
    pt,l1,l2,s1,s2=mpSpecInfo[title] # 这里记录的l1和l2是序列的真实长度-1
    alpha_peaks=peaks[:,:l1].tolist()
    beta_peaks=peaks[:,l1+1:].tolist() # 跳过中间的交联剂输出

    # 与dta的格式一致，补齐无效的b离子和y离子，使得离子长度和肽段长度一致
    for idx in range(8):
        if idx%4<=1: # b ion
            alpha_peaks[idx].append(0.0) 
            beta_peaks[idx].append(0.0)
        else: # y ion
            alpha_peaks[idx].insert(0,0.0)
            beta_peaks[idx].insert(0,0.0)

    pred_matrix=alpha_peaks,beta_peaks # plot_single接收的输入格式

    return mpPeaks, mpSpecInfo,pred_matrix

#----------------------------------------------------------------




# ----- batch predict and save to file -----------

def predict_batch(path_data_file,is_non_cleavable):

    BATCH_SIZE=1024
    print('loading predicting data...')
    predict_data_loader,mpTitleLines,header=LoadPredictDataSet(BATCH_SIZE, is_non_cleavable, path_data_file)

    model = get_model(is_non_cleavable)

    print('predicting...')
    mpPeaks, mpSpecInfo=PredictMS2(model, predict_data_loader)
    print('prediction done')
    
    return mpPeaks, mpTitleLines, header, mpSpecInfo


def save_result_batch(path, predicted_results):
    mpPeaks, mpTitleLines, header, mpSpecInfo=predicted_results
    fout=open(path,'w')
    fout.write(header+'\t')
    fout.write('seq1_pred_b1b2y1y2\tseq2_pred_b1b2y1y2\n')
    for title, peaks in mpPeaks.items():
        fout.write(mpTitleLines[title]+'\t')
        pt,l1,l2,s1,s2=mpSpecInfo[title] # 这里记录的l1和l2是序列的真实长度-1
        alpha_peaks=peaks[:,:l1].tolist()
        beta_peaks=peaks[:,l1+1:].tolist() # 跳过中间的交联剂输出

        # 与dta的格式一致，补齐无效的b离子和y离子，使得离子长度和肽段长度一致
        for idx in range(8):
            if idx%4<=1: # b ion
                alpha_peaks[idx].append(0.0) 
                beta_peaks[idx].append(0.0)
            else: # y ion
                alpha_peaks[idx].insert(0,0.0)
                beta_peaks[idx].insert(0,0.0)

        fout.write('\t'.join(map(str,[alpha_peaks]))+'\t')
        fout.write('\t'.join(map(str,[beta_peaks]))+'\n')
    fout.close()
    print('write done')

#----------------------------------------------------------------





#--------------------- generate spectra library ---------------------

def generate_mgf_library(path, predicted_results):
    pass


def generate_blib_library(path, predicted_results):
    pass


def generate_msp_library(path, predicted_results):
    pass


def generate_spectra_library(path, format, predicted_results):
    if format=='mgf':
        generate_mgf_library(path, predicted_results)
    elif format=='blib':
        generate_blib_library(path, predicted_results)
    elif format=='msp':
        generate_msp_library(path, predicted_results)
    else:
        print('sorry, spectra library format %s is not supported.'%format)
        return

#----------------------------------------------------------------
