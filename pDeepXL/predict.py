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


import pDeepXL.mass_util as mass_util
import sqlite3
import time
import zlib
import struct
import os
HIGHEST_INTENSITY = 10000


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

def peaks_post_process(peaks):
    """谱峰后处理，mz排序、近质量合并、强度重新归一化"""
    peaks = sorted(peaks, key=lambda x:(x[0],x[1]))

    tolerance_samemass = 1 # ppm
    def accu_samemass_peak(peaks):
        """累加质量相近的谱峰"""
        peaks_new = []
        for i, (mz, relat_inten, anno) in enumerate(peaks):
            if i!=0 and abs(mz-peaks[i-1][0])*1e6 < tolerance_samemass:
                peaks_new[i-1][1] += relat_inten
                peaks_new[i-1][2].append(anno)
                continue
            peaks_new.append([mz, relat_inten, [anno]])
        return peaks_new
    peaks = accu_samemass_peak(peaks)

    # 强度重新归一化
    intens = [x[1] for x in peaks]
    max_inten = max(intens)
    if max_inten != 0.0:
        for i, (mz, relat_inten, anno) in enumerate(peaks):
            peaks[i][1] = relat_inten/max_inten
    return peaks

def cal_non_clv_peaks(pep_pair,prec_charge,pred_matrix):
    """计算不可碎裂母离子质量，谱峰[mz, 强度, 注释]"""
    LinkerName,seq1,mods1,linksite1,seq2,mods2,linksite2=pep_pair
    alpha_preds,beta_preds=pred_matrix
    alpha_preds,beta_preds=alpha_preds[:4],beta_preds[:4]
    
    l1,l2=len(seq1),len(seq2)
    b1mass1,b2mass1,y1mass1,y2mass1,b1mass2,b2mass2,y1mass2,y2mass2=mass_util.calfragmass4xl(seq1,mods1,linksite1,seq2,mods2,linksite2,LinkerName)
    
    alpha_ions=[b1mass1,b2mass1,y1mass1,y2mass1]
    beta_ions=[b1mass2,b2mass2,y1mass2,y2mass2]
    ion_charges=(1,2,1,2)
    ion_names=('b+','b++','y+','y++')

    peaks = []
    # alpha
    for ion_mzs, ion_charge, ion_name, pred_intens in zip(alpha_ions, ion_charges, ion_names, alpha_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            relative_intensity=pred_inten
            ion_pos= pos+1 if ion_name[0]=='b' else l1-pos
            # msp格式离子注释
            ion_txt='a%s%d'%(ion_name[0],ion_pos)
            if ion_charge>=2:
                ion_txt='%s^%d'%(ion_txt, ion_charge)
            ion_txt='%s/0.0 1/1 0.0'%(ion_txt)
            peaks.append([theo_mz,relative_intensity,ion_txt])

    # beta
    for ion_mzs, ion_charge, ion_name, pred_intens in zip(beta_ions, ion_charges, ion_names, beta_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            relative_intensity=pred_inten
            ion_pos= pos+1 if ion_name[0]=='b' else l2-pos
            # msp格式离子注释
            ion_txt='b%s%d'%(ion_name[0],ion_pos)
            if ion_charge>=2:
                ion_txt='%s^%d'%(ion_txt, ion_charge)
            ion_txt='%s/0.0 1/1 0.0'%(ion_txt)
            peaks.append([theo_mz,relative_intensity,ion_txt])

    prec_mass=(alpha_ions[0][0]+alpha_ions[2][1]-2*mass_util.PMass+prec_charge*mass_util.PMass)/prec_charge

    peaks = peaks_post_process(peaks)
    return prec_mass,peaks

def cal_clv_peaks(pep_pair,prec_charge,pred_matrix):
    """计算可碎裂母离子质量，谱峰[mz, 强度, 注释]"""
    LinkerName,seq1,mods1,linksite1,seq2,mods2,linksite2=pep_pair
    alpha_preds,beta_preds=pred_matrix

    l1,l2=len(seq1),len(seq2)

    # 把8维拆开为12维，和下面的ion_names对应
    alpha_preds=np.array(alpha_preds)
    alpha_preds_expended=np.zeros([12,l1])
    alpha_preds_expended[0:2,0:linksite1]=alpha_preds[0:2,0:linksite1] # regular b
    alpha_preds_expended[2:4,linksite1+1:]=alpha_preds[2:4,linksite1+1:] # regular y
    alpha_preds_expended[4:6,linksite1:]=alpha_preds[0:2,linksite1:] # clv-long b
    alpha_preds_expended[6:8,0:linksite1+1]=alpha_preds[2:4,0:linksite1+1] # clv-long y
    alpha_preds_expended[8:12,:]=alpha_preds[4:8,:] # clv-short
    alpha_preds=alpha_preds_expended.tolist()

    beta_preds=np.array(beta_preds)
    beta_preds_expended=np.zeros([12,l2])
    beta_preds_expended[0:2,0:linksite2]=beta_preds[0:2,0:linksite2] # regular b
    beta_preds_expended[2:4,linksite2+1:]=beta_preds[2:4,linksite2+1:] # regular y
    beta_preds_expended[4:6,linksite2:]=beta_preds[0:2,linksite2:] # clv-long b
    beta_preds_expended[6:8,0:linksite2+1]=beta_preds[2:4,0:linksite2+1] # clv-long y
    beta_preds_expended[8:12,:]=beta_preds[4:8,:] # clv-short
    beta_preds=beta_preds_expended.tolist()
    #################################


    alpha_ions=mass_util.calfragmass4clv(seq1,LinkerName,linksite1,mods1)
    beta_ions=mass_util.calfragmass4clv(seq2,LinkerName,linksite2,mods2)

    ion_charges=(1,2,1,2,1,2,1,2,1,2,1,2)
    ion_names=('b+','b++','y+','y++','Lb+','Lb++','Ly+','Ly++','Sb+','Sb++','Sy+','Sy++')

    peaks = []
    # alpha
    for ion_mzs, ion_charge, ion_name, pred_intens in zip(alpha_ions, ion_charges, ion_names, alpha_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            relative_intensity=pred_inten
            ion_pos= pos+1 if 'b' in ion_name else l1-pos
            # msp格式离子注释
            ion_txt='a%s%d'%(ion_name.split('+')[0],ion_pos)
            if ion_charge>=2:
                ion_txt='%s^%d'%(ion_txt, ion_charge)
            ion_txt='%s/0.0 1/1 0.0'%(ion_txt)
            peaks.append([theo_mz,relative_intensity,ion_txt])

    # beta
    for ion_mzs, ion_charge, ion_name, pred_intens in zip(beta_ions, ion_charges, ion_names, beta_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            relative_intensity=pred_inten
            ion_pos= pos+1 if 'b' in ion_name else l2-pos
            # msp格式离子注释
            ion_txt='b%s%d'%(ion_name.split('+')[0],ion_pos)
            if ion_charge>=2:
                ion_txt='%s^%d'%(ion_txt, ion_charge)
            ion_txt='%s/0.0 1/1 0.0'%(ion_txt)
            peaks.append([theo_mz,relative_intensity,ion_txt])

    peaks = peaks_post_process(peaks)

    # linksite从0开始
    if linksite1 != 0:
        # 同一碎裂位点b+Ly
        prec_mass_alpah_long = alpha_ions[0][0]+alpha_ions[6][1]-2*mass_util.PMass # 中性质量
    else:
        # 同一碎裂位点Lb+y
        prec_mass_alpah_long = alpha_ions[4][0]+alpha_ions[2][1]-2*mass_util.PMass # 中性质量
    
    if linksite2 != 0:
        # 同一碎裂位点b+Sy
        prec_mass_beta_short = beta_ions[0][0]+beta_ions[10][1]-2*mass_util.PMass # 中性质量
    else:
        # 同一碎裂位点Sb+y
        prec_mass_beta_short = beta_ions[8][0]+beta_ions[2][1]-2*mass_util.PMass # 中性质量

    prec_mass = (prec_mass_alpah_long+prec_mass_beta_short-sum(mass_util.mpClvLinkerLongShortMass[LinkerName])+mass_util.mpLinkerXLMass[LinkerName])/prec_charge+mass_util.PMass

    # print(peaks)
    return prec_mass,peaks


def generate_mgf_library(path, predicted_results):
    """生成预测谱图的mgf格式文件"""

    begin_str = 'BEGIN IONS\n'
    end_str = 'END IONS\n'
    fout=open(path,'w')
    
    mpPeaks, mpTitleLines, header, mpSpecInfo=predicted_results
    for title, peaks in mpPeaks.items():
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

        segs = mpTitleLines[title].strip().split('\t')
        title_line = segs[0]
        scan = int(segs[1])
        charge = int(segs[2])
        instrument = segs[3]
        NCE_low = float(segs[4])
        NCE_medium = float(segs[5])
        NCE_high = float(segs[6])
        crosslinker = segs[7]
        seq1 = segs[8]
        mods1 = eval(segs[9]) # 不安全的
        linksite1 = int(segs[10])
        seq2 = segs[11]
        mods2 = eval(segs[12])
        linksite2 = int(segs[13])
        seq1_pred_b1b2y1y2 = alpha_peaks
        seq2_pred_b1b2y1y2 = beta_peaks

        pep_pair=[crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2]
        pred_matrix=seq1_pred_b1b2y1y2,seq2_pred_b1b2y1y2
        if crosslinker=='DSSO' or crosslinker=='DSBU':
            prec_mass,peaks = cal_clv_peaks(pep_pair,charge,pred_matrix)
        elif crosslinker=='DSS' or crosslinker=='Leiker':
            prec_mass,peaks = cal_non_clv_peaks(pep_pair,charge,pred_matrix)
        else:
            print('do not support %s cross-linker'%crosslinker)
            break

        rt = 0.0
        fout.write(begin_str)
        fout.write('TITLE=%s\n'%(title))
        fout.write('CHARGE=%d+\n'%(charge))
        fout.write('RTINSECONDS=%f\n'%(rt))
        fout.write('PEPMASS=%f\n'%(prec_mass))
        for mz, relat_inten, anno in peaks:
            fout.write('%f %f\n'%(mz, relat_inten*HIGHEST_INTENSITY))
        fout.write(end_str)

    fout.close()
    print('mgf write done')



def generate_blib_library(path, predicted_results):
    """生成预测谱库blib格式"""

    fname = os.path.split(path)[1]

    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path) # 链接数据库
    cur = conn.cursor() # 创建游标cur来执行SQL语句

    mpPeaks, mpTitleLines, header, mpSpecInfo=predicted_results

    # 库的头部信息
    cur.execute('''CREATE TABLE LibInfo
                (libLSID TEXT,
                createTime TEXT,
                numSpecs INTEGER,
                majorVersion INTEGER,
                minorVersion INTEGER);''')

    liblsid_str = 'urn:lsid:pfind.ict.ac.cn:spectral_library:pdeepxl:nr:'+fname
    createtime_str = time.asctime(time.localtime(time.time()))
    numspecs = len(mpPeaks)
    majorversion = 1
    minorversion = 1

    cur.execute("INSERT INTO LibInfo VALUES('%s', '%s', %d, %d, %d)"%(liblsid_str, createtime_str, numspecs, majorversion, minorversion))

    # 创建RefSpectra表
    cur.execute('''CREATE TABLE RefSpectra
                (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                peptideSeq VARCHAR(150),
                precursorMZ REAL,
                precursorCharge INTEGER,
                peptideModSeq VARCHAR(200),
                numPeaks INTEGER);''')

    # 创建Modifications表
    # 修饰从0开始，alpha与beta肽拼接后计算位点，中间只多一个交联剂修饰位点
    cur.execute('''CREATE TABLE Modifications
                (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                RefSpectraID INTEGER,
                position INTEGER,
                mass REAL);''')

    # 创建RefSpectraPeaks表
    # mz little-endian 64 bit doubles, zlib-compressed
    # 强度 little-endian 32 bit floats, zlib-compressed
    cur.execute('''CREATE TABLE RefSpectraPeaks
                (RefSpectraID INTEGER,
                peakMZ BLOB,
                peakIntensity BLOB);''')
    


    for spec_id, (title, peaks) in enumerate(mpPeaks.items()):
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

        segs = mpTitleLines[title].strip().split('\t')
        title_line = segs[0]
        scan = int(segs[1])
        charge = int(segs[2])
        instrument = segs[3]
        NCE_low = float(segs[4])
        NCE_medium = float(segs[5])
        NCE_high = float(segs[6])
        crosslinker = segs[7]
        seq1 = segs[8]
        mods1 = eval(segs[9]) # 不安全的
        linksite1 = int(segs[10])
        seq2 = segs[11]
        mods2 = eval(segs[12])
        linksite2 = int(segs[13])
        seq1_pred_b1b2y1y2 = alpha_peaks
        seq2_pred_b1b2y1y2 = beta_peaks

        pep_pair=[crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2]
        pred_matrix=seq1_pred_b1b2y1y2,seq2_pred_b1b2y1y2
        if crosslinker=='DSSO' or crosslinker=='DSBU':
            prec_mass,peaks = cal_clv_peaks(pep_pair,charge,pred_matrix)
        elif crosslinker=='DSS' or crosslinker=='Leiker':
            prec_mass,peaks = cal_non_clv_peaks(pep_pair,charge,pred_matrix)
        else:
            print('do not support %s cross-linker'%crosslinker)
            break


        # 谱图识别唯一编号
        spectra_id = spec_id + 1

        # === 序列字符串，带修饰
        seq1_mod=list(seq1)
        # 修饰按照位点重排序，方便在序列中添加修饰
        mods1 = dict(sorted(mods1.items(), key=lambda x:(x[0],x[1])))
        mod_num = 0
        for site, mod_str in mods1.items():
            mod_mass = mass_util.mpModMassN14[mod_str]
            if mod_mass >= 0:
                seq1_mod.insert(site+1+mod_num, '[+%.1f]'%(mod_mass))
            else:
                seq1_mod.insert(site+1+mod_num, '[%.1f]'%(mod_mass))
        seq1_mod=''.join(seq1_mod)
  
        seq2_mod=list(seq2)
        mods2 = dict(sorted(mods2.items(), key=lambda x:(x[0],x[1])))
        mod_num = 0
        for site, mod_str in mods2.items():
            mod_mass = mass_util.mpModMassN14[mod_str]
            if mod_mass >= 0:
                seq2_mod.insert(site+1+mod_num, '[+%.1f]'%(mod_mass))
            else:
                seq2_mod.insert(site+1+mod_num, '[%.1f]'%(mod_mass))
        seq2_mod=''.join(seq2_mod)

        seq_str = '%s(%d)-%s-%s(%d)'%(seq1,linksite1,crosslinker,seq2,linksite2)
        seq_mod_str = '%s(%d)-%s-%s(%d)'%(seq1_mod,linksite1,crosslinker,seq2_mod,linksite2)

        cur.execute("INSERT INTO RefSpectra (peptideSeq, precursorMZ, precursorCharge, peptideModSeq, numPeaks) VALUES('%s', %f, %d, '%s', %d)"%(seq_str, prec_mass, charge, seq_mod_str, len(peaks)))

        if len(mods1) != 0:
            for site, mod_str in mods1.items():
                cur.execute("INSERT INTO Modifications (RefSpectraID, position, mass) VALUES(%d, %d, %f)"%(spectra_id, site+1, mass_util.mpModMassN14[mod_str]))
        if len(mods2) != 0:
            for site, mod_str in mods2.items():
                cur.execute("INSERT INTO Modifications (RefSpectraID, position, mass) VALUES(%d, %d, %f)"%(spectra_id, site+1+len(seq1)+1, mass_util.mpModMassN14[mod_str]))

        peaks_mz = []
        peaks_inten = []
        for mz, relat_inten, anno in peaks:
            peaks_mz.append(mz)
            peaks_inten.append(relat_inten*HIGHEST_INTENSITY)
        peaks_num = len(peaks)

        mz_bin = struct.pack('<{0}d'.format(peaks_num), *peaks_mz)
        mz_bin_de = zlib.compress(mz_bin)
        inten_bin = struct.pack('<{0}f'.format(peaks_num), *peaks_inten)
        inten_bin_de = zlib.compress(inten_bin)

        cur.execute("INSERT INTO RefSpectraPeaks VALUES(%d, ?, ?)"%(spectra_id),(sqlite3.Binary(mz_bin_de), sqlite3.Binary(inten_bin_de)))

    conn.commit()
    conn.close()
    print('blib write done')



def generate_msp_library(path, predicted_results):
    """生成预测谱库msp格式"""

    fout=open(path,'w')
    
    mpPeaks, mpTitleLines, header, mpSpecInfo=predicted_results
    for title, peaks in mpPeaks.items():
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

        segs = mpTitleLines[title].strip().split('\t')
        title_line = segs[0]
        scan = int(segs[1])
        charge = int(segs[2])
        instrument = segs[3]
        NCE_low = float(segs[4])
        NCE_medium = float(segs[5])
        NCE_high = float(segs[6])
        crosslinker = segs[7]
        seq1 = segs[8]
        mods1 = eval(segs[9]) # 不安全的
        linksite1 = int(segs[10])
        seq2 = segs[11]
        mods2 = eval(segs[12])
        linksite2 = int(segs[13])
        seq1_pred_b1b2y1y2 = alpha_peaks
        seq2_pred_b1b2y1y2 = beta_peaks

        pep_pair=[crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2]
        pred_matrix=seq1_pred_b1b2y1y2,seq2_pred_b1b2y1y2
        if crosslinker=='DSSO' or crosslinker=='DSBU':
            prec_mass,peaks = cal_clv_peaks(pep_pair,charge,pred_matrix)
        elif crosslinker=='DSS' or crosslinker=='Leiker':
            prec_mass,peaks = cal_non_clv_peaks(pep_pair,charge,pred_matrix)
        else:
            print('do not support %s cross-linker'%crosslinker)
            break

        # === 序列字符串，带氧化修饰
        seq1_mod=list(seq1)
        # 修饰按照位点重排序，方便在序列中添加多个氧化
        mods1 = dict(sorted(mods1.items(), key=lambda x:(x[0],x[1])))
        ox_num = 0
        for site, mod_str in mods1.items():
            if 'Oxidation' in mod_str:
                seq1_mod.insert(site+1+ox_num, '(O)')
        seq1_mod=''.join(seq1_mod)
  
        seq2_mod=list(seq2)
        # 修饰按照位点重排序，方便在序列中添加多个氧化
        mods2 = dict(sorted(mods2.items(), key=lambda x:(x[0],x[1])))
        ox_num = 0
        for site, mod_str in mods2.items():
            if 'Oxidation' in mod_str:
                seq2_mod.insert(site+1+ox_num, '(O)')
        seq2_mod=''.join(seq2_mod)

        seq_str = '%s(%d)-%s-%s(%d)'%(seq1_mod,linksite1,crosslinker,seq2_mod,linksite2)
        fout.write('Name: %s/%d\n'%(seq_str,charge))

        fout.write('MW: %.3f\n'%(prec_mass*charge))

        # 修饰从0开始，alpha与beta肽拼接后计算位点，中间只多一个交联剂修饰位点
        mods_str = 'Mods='
        mods_num = len(mods1) + len(mods2)
        mods_str += str(mods_num)
        if mods_num != 0:
            for site, mod_str in mods1.items():
                mods_str += '/'
                mods_str += str(site)
                mods_str += ','
                mods_str += seq1[site]
                mods_str += ','
                mods_str += mod_str.split('[')[0]
            for site, mod_str in mods2.items():
                mods_str += '/'
                mods_str += str(site+len(seq1)+1)
                mods_str += ','
                mods_str += seq2[site]
                mods_str += ','
                mods_str += mod_str.split('[')[0]
        parent_str = 'Parent=%.3f'%(prec_mass)
        naa_str = 'Naa=%d'%(len(seq1)+len(seq2))
        fout.write('Comment: %s %s %s\n'%(mods_str, parent_str, naa_str))
        
        fout.write('Num peaks: %d\n'%(len(peaks)))
        for mz, relat_inten, anno in peaks:
            fout.write('%f %f "%s"\n'%(mz, relat_inten*HIGHEST_INTENSITY, ','.join(anno[:2])))
        
        fout.write('\n')

    fout.close()
    print('msp write done')


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
