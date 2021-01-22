
import os
import struct
import configparser

import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats.stats import pearsonr
import torch.utils.data
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnn_utils
import re
import logging
import bisect
import pkg_resources # https://stackoverflow.com/a/16892992
path_config = pkg_resources.resource_filename('pDeepXL', 'configs/config.ini')

# print('-1-')
# print(path_config)
# print('-2-')
config = configparser.ConfigParser()
# print(os.path.abspath(os.getcwd()))
# path_config=r'configs/config.ini' # 在当前项目下，绝对路径要从项目根目录pDeepXL开始
# if os.path.isfile(path_config):
#     print('config file exist')
# else:
#     print('config file not exist')
config.read(path_config, encoding='UTF-8')
good_mods=set(config['DEFAULT']['valid_mods'].split(','))
# print('good mods: ', good_mods)

# ScanHeadsman-1.2.20200730输出的仪器名称和对应的简称
mpInstrumentShortNames={'Orbitrap Fusion Lumos':'Lumos','Q Exactive HF-X Orbitrap':'QEHFX','Orbitrap Fusion':'Fusion','Q Exactive HF Orbitrap':'QEHF','Q Exactive Orbitrap':'QE','Q Exactive Plus Orbitrap':'QEPlus','Exactive Plus Orbitrap':'QEPlus'}

# 读取仪器类型
def read_instrument_name(instrument_summary_path):
    fin=open(instrument_summary_path)
    lines=fin.readlines()
    fin.close()

    instrument_name_idx=8
    for i, col in enumerate(lines[0].split(',')):
        if col.strip()=='InstrumentName':
            instrument_name_idx = i

    # ans=lines[1].split(',')[8]
    contents=re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', lines[1].strip()) # 防止有些单元格内部逗号
    ans=contents[instrument_name_idx]
    return mpInstrumentShortNames[ans]

# instrument_summary_path=r'/data/zlchen/pDeepXL/data/PXD012546/R1_A5.sum.csv'
# print(read_instrument_name(instrument_summary_path))


# 判断一个字符串是否是数字，int和float都可以
def IsFloat(x):
    try:
        v=float(x)
        return True
    except ValueError:
        return False


# 读取每个MS2 Scan的能量
# 目前只支持单能量或三能量
def read_ms2_energy(energy_info_path):
    fin=open(energy_info_path)
    lines=fin.readlines()
    fin.close()

    hcd_energy_idx=0 # 不同质谱设置，能量所在的列不同，根据header行找到能量列
    energy_idx1=0 # 如果energy_idx对应列为负数，则用这一列
    activation_idx1=0 # 必须要是HCD碎裂模式
    for i, col in enumerate(lines[0].split(',')):
        if col.strip()=='HCD Energy': # 如果是阶梯能量，这一列显示最完整
            hcd_energy_idx = i
        elif col.strip()=='Energy1': # 如果是非阶梯能量，这一列也可以
            energy_idx1 = i
        elif col.strip()=='Activation1':
            activation_idx1 = i

    mpMS2Energy={}
    for i in range(1,len(lines)):
        # contents=lines[i].split(',')
        contents=re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', lines[i].strip()) # 有些单元格内部还有逗号
        msorder=int(contents[1])
        if msorder != 2:
            continue
        ms2scan=int(contents[0])

        energy_str=contents[energy_idx1]
        if contents[activation_idx1].strip().upper() != 'HCD': # 只提取HCD能量，单能量
            continue
        if '"' in contents[hcd_energy_idx]: # 多能量
            energy_str=contents[hcd_energy_idx]

        eg_contents=energy_str.split(',')
        if len(eg_contents) == 1: # 单能量
            mpMS2Energy[ms2scan]=[0.0, float(energy_str), 0.0] # low, medium, high
        elif len(eg_contents) == 3: # 三能量
            eg_contents=energy_str.strip()[1:-1].split(',') # 去掉外层的双引号
            mpMS2Energy[ms2scan]=sorted(list(map(float,eg_contents)))

    return mpMS2Energy


# energy_info_path=r'/data/zlchen/pDeepXL/data/PXD017620/CV2_DSSO_CV5_20200512_stepHCD_8_30B_R1.csv'
# energy_info_path=r'/data/zlchen/pDeepXL/data/PXD017620/XL/pf2/C_Lee_010916_ymitos_BS3_XL_A13_A15_11_Rep2.csv'
# energy_info_path=r'/data/zlchen/pDeepXL/data/PXD017620/crude_fresh/pf2/A_Linden_131118_211118_ymitos_crude_fresh_1to1_12.csv'
# energy_info_path=r'/data/zlchen/pDeepXL/data/PXD012546/R1_A5.csv'
# mpMS2Energy=read_ms2_energy(energy_info_path)
# print(mpMS2Energy)
# print('hello')


# 提取文件夹内所有文件的绝对路径
def absoluteFilePaths(directory):
    ans=[]
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            ans.append(os.path.abspath(os.path.join(dirpath, f)))
    return ans



# 读取pf2文件
def load_pf2(pf2_path):

    try:
        f = open(pf2_path, "rb")
        spec_title = os.path.basename(pf2_path)[:-10]
        nSpec,lenTitle = struct.unpack("2i", f.read(8))
        pf2title = struct.unpack("%ds" %lenTitle, f.read(lenTitle))
    except BaseException as error:
        print('An exception occurred: {}'.format(error))
        print('error pf2=%s'%pf2_path)
    #print(pf2title)
    
    mpSpec = {}
    
    for __i__ in range(nSpec):
        scan_no, = struct.unpack("i",f.read(4))
    #    print(">>Spectrum NO = %d" %scan_no)
        nPeak, = struct.unpack("i", f.read(4))
        peaks = []
        mz_int = struct.unpack(str(nPeak*2)+"d", f.read(nPeak*2*8))
        for i in range(nPeak):
            mz = mz_int[i*2]
            inten = mz_int[i*2 + 1]
            peaks.append( (mz, inten) )
        
        peaks=sorted(peaks,key=lambda x:x[0]) # 按mz从小到大排序，也许pf2默认已经排过序了。
        max_inten=0
        if len(peaks)!=0: # 当谱图没有任何谱峰时，会出bug：B170414_02_Lumos_ML_IN_205_FOMixSEC29BS3_TrypGluC_pepSECFr16.33307.33307.3.0.dta
            max_inten=max(peaks,key=lambda x:x[1])[1]
        
        nMix, = struct.unpack("i", f.read(4))
        nMaxCharge = 0
        for i in range(nMix):
            precursor, = struct.unpack("d", f.read(8))
            nCharge, = struct.unpack("i", f.read(4))
            if nCharge > nMaxCharge: nMaxCharge = nCharge
            specname = "%s.%d.%d.%d.%d.dta" %(spec_title, scan_no, scan_no, nCharge, i)
            spec_info=(specname, nCharge, precursor,max_inten)
            mpSpec[specname]=[spec_info,peaks]
    f.close()
    
    return pf2title, mpSpec


# 读取pFind3已按FDR过滤的结果
def load_pfind3_filtered(pfind3_path):
    
    fin=open(pfind3_path)
    lines=fin.readlines()
    fin.close()
    
    psms=[]

    for i in range(1,len(lines)):
        contents=lines[i].split('\t')
        title=contents[0].strip()
        scan=int(contents[1])
        exp_mh=float(contents[2])
        charge=int(contents[3])
        qvalue=float(contents[4])
        
        seq=contents[5].strip()
        ms=contents[10].strip().split(';')
        mods2={}
        for m in ms:
            if m=='':
                continue
            pos,name=m.split(',')
            pos=max(0, int(pos)-1) # 转换，修饰的下标从0开始；如果是N端修饰，则本身是0，不能出现负数
            name=name.strip()
            mods2[pos]=name

        psms.append([title,scan,charge,exp_mh,seq,mods2])
        
    return psms


# 读取pLink2过滤后的CSM结果
def ReadpLinkFilteredPSM5(strPLinkFilteredPath):
    vpLinkPSM = [] # title,scan,charge,exp_mh,seq1,mods1,linksite1,seq2,mods2,linksite2
    fpLink = open(strPLinkFilteredPath)
    lines=fpLink.readlines()
    fpLink.close()
    for i in range(1,len(lines)):
        line = lines[i].strip()
        contents = line.split(',')

        title = contents[1].strip()
        scan = int(title.split('.')[1])
        charge = int(contents[2])
        exp_mh = float(contents[3])

        pep1,pep2 = contents[4].split('-')
        seq1 = ''.join([i for i in pep1 if i.isalpha()])
        seq2 = ''.join([i for i in pep2 if i.isalpha()])
        lseq1=len(seq1)

        linksite1 = ''.join([i for i in pep1 if i.isnumeric()])
        linksite2 = ''.join([i for i in pep2 if i.isnumeric()])

        linksite1=max(0,int(linksite1)-1)
        linksite2=max(0,int(linksite2)-1)

        LinkerName = contents[6].strip()
        if LinkerName=='SS':
            assert (seq1[linksite1]=='C') and (seq2[linksite2]=='C') # 交联位点没问题
        else:
            assert (linksite1==0 or seq1[linksite1]=='K') and (linksite2==0 or seq2[linksite2]=='K') # 交联位点没问题

        mods=contents[8].strip().split(';')
        mods1,mods2={},{}

        for m in mods:
            if m.strip() == '' or m.lower() == 'null':
                continue
            mname,msite=m.split('(')
            msite=int(msite[:-1])
            maa=mname[-2:-1]

            if msite > lseq1+1: # beta修饰
                msite=max(0,msite-lseq1-3-1)
                assert seq2[msite]==maa
                mods2[msite]=mname
            else:
                msite=max(0,msite-1)
                assert seq1[msite] == maa
                mods1[msite]=mname

        vpLinkPSM.append([title,scan,charge,exp_mh,seq1,mods1,linksite1,seq2,mods2,linksite2])
    return vpLinkPSM



# 读取aa.ini
def readaaini(path):
    fin=open(path)
    lines=fin.readlines()
    fin.close()
    
    mpaa={}
    for line in lines:
        if line[0]=='R':
            mpcomp={}
            aa,com=line.split('=')[1].strip().split('|')[0:2]
            eles=com.split(')')
            for i in range(len(eles)-1):
                e,n=eles[i].split('(')
                n=int(n)
                mpcomp[e]=n
            mpaa[aa]=mpcomp
    return mpaa


# 读取element.ini
def readeleini(path):
    fin=open(path)
    lines=fin.readlines()
    fin.close()
    
    mpele={}
    for line in lines:
        if line[0]=='E':
            ele,mass,prob=line.split('=')[1].strip().split('|')[0:3]
            ms=[float(i) for i in mass.split(',')[:-1]]
            ps=[float(i) for i in prob.split(',')[:-1]]
            comb=list(zip(ms,ps))
            high_prob_mass=sorted(comb,key=lambda x:x[1])[-1][0]
            mpele[ele]=high_prob_mass
    return mpele



# 计算ppm质量误差
def calppm(theo,exp):
    return abs(1e6*(exp-theo)/theo) # 一定要加abs！！！！！！！！！！！

# 线性查找，太慢了！！！
def ismatch(spec,theo_mz,charge):
    if theo_mz<0:
        return 0.0
    spec_info,peaks = spec
    for exp_mz,intensity in peaks:
        if calppm(theo_mz*charge,exp_mz*charge)<=float(config['DEFAULT']['ppm_ms2']):
            return intensity
    return 0.0

# 二分查找，相比线性查找，提速至少30倍
# 经过单肽和交联数据集验证，结果和线性查找完全一样
def ismatch_bs(spec,theo_mz,charge):
    if theo_mz<0:
        return 0.0
    spec_info,peaks = spec
    n=len(peaks)
    exp_mzs=[p[0] for p in peaks] # 看load_pf2函数，已经是从小到大排序过的
    theo_mz_low,theo_mz_high=theo_mz-1,theo_mz+1 # 1Da的误差肯定比20ppm大
    low_idx,high_idx=bisect.bisect_left(exp_mzs,theo_mz_low),bisect.bisect_right(exp_mzs,theo_mz_high)
    for i in range(low_idx, high_idx+1):
        if i >=0 and i <n:
            exp_mz,intensity=peaks[i]
            if calppm(theo_mz*charge,exp_mz*charge)<=float(config['DEFAULT']['ppm_ms2']):
                return intensity
    return 0.0


def matchbatch(spec,ion_mzs,ion_charge):
    n=len(ion_mzs)
    mints=[0.0]*n
    for i,ion_mz in enumerate(ion_mzs):
        curint=ismatch_bs(spec,ion_mz,ion_charge)
        mints[i]=curint
    return mints


# 目前版本仅考虑C+57和O+16修饰
def is_mod_valid(mods):
    for msite,mname in mods.items():
        if mname not in good_mods:
            return False
    return True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def PlotWithEpoch(train_values, valid_values, path_result_home, name):
    epochs = [i + 1 for i in range(len(train_values))]

    plt.figure()
    plt.plot(epochs, train_values,'b',label='Training %s'%name)
    plt.plot(epochs, valid_values,'r',label='Validation %s'%name)
    plt.title('Training and Validation %s'%name)
    plt.legend()
    plt.xlim(1, len(epochs) + 1)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel('epochs')
    plt.savefig(r'%s/%s.png'%(path_result_home,name))

def CountAccuPercentage(sims):
    s=sorted(sims)
    n=len(s)
    x=[s[0]] # 最小pcc
    y=[1.0] # yi=pcc>=xi的比例
    for i,pcc in enumerate(s):
        x.append(pcc)
        y.append(1-i/n)
    x.append(s[-1])
    y.append(1/len(sims)) # 最后一个
    return x,y

# input is sorted pccs and the corresponding perncentages
def BatchBSWantedPccs(pccs, percentages, wanted_pccs):
    wanted_percentages=[]
    for wpcc in wanted_pccs:
        idx=bisect.bisect_left(pccs, wpcc)
        if idx==len(pccs):
            wanted_percentages.append(0) # 所有pcc都没超过wpcc
        else:
            wanted_percentages.append(percentages[idx])
    return wanted_percentages


def PlotSimPercentage(train_sims, valid_sims, path_result_home, name):
    train_pcc,train_percentage=CountAccuPercentage(train_sims)
    valid_pcc,valid_percentage=CountAccuPercentage(valid_sims)

    plt.figure()
    plt.plot(train_pcc, train_percentage,'b',label='Training %s'%name)
    plt.plot(valid_pcc, valid_percentage,'r',label='Validation %s'%name)

    plt.title('Traing and Validation %s'%name)
    plt.legend()

    plt.xlabel('PCC = x')
    plt.ylabel('Percentage of PCCs >= x (%)')
    plt.savefig(r'%s/%s.png'%(path_result_home,name))


def PloSingleSimPercentage(pccs, percentages, path_result_home, name):
    plt.figure()
    plt.plot(pccs, percentages,'b',label=name)

    plt.title(name)
    plt.legend()

    plt.xlabel('PCC = x')
    plt.ylabel('Percentage of PCCs >= x (%)')
    plt.savefig(r'%s/%s.png'%(path_result_home,name))


# 输入numpy二维矩阵，shape：len*16
# 提取单肽离子，展开成一维向量
def flat_linear_pep_ions(input_matrix):
    return input_matrix[:,:4].flatten()

# # 输入numpy二维矩阵，shape：len*16
# # 提取不可碎裂交联肽段产生的离子
# # 包括单肽离子和xlink离子，展开成一维向量
# def flat_non_clv_pep_ions(input_matrix,l1,l2,s1,s2):
#     # 把交联输出拷贝到对应单肽位置
#     input_matrix[s1:l1,0:2]=input_matrix[s1:l1,4:6] # alpha cp xlink b
#     input_matrix[0:s1,2:4]=input_matrix[0:s1,6:8] # alpha cp xlink y
#     input_matrix[l1+1+s2:,0:2]=input_matrix[l1+1+s2:,4:6] # beta cp xlink b
#     input_matrix[l1+1:l1+1+s2,2:4]=input_matrix[l1+1:l1+1+s2,6:8] # beta cp xlink y
#     # 截取粘贴了交联离子的单肽区域出来算PCC
#     return flat_linear_pep_ions(input_matrix)

# 输入numpy二维矩阵，shape：len*16
# 提取可碎裂交联肽段产生的离子
# 包括单肽离子和clv-long、clv-short离子，展开成一维向量
def flat_clv_pep_ions(input_matrix,l1,l2,s1,s2):
    # 先把clv-long交联输出拷贝到对应单肽位置
    # input_matrix[s1:l1,0:2]=input_matrix[s1:l1,8:10] # alpha cp clv-long b
    # input_matrix[0:s1,2:4]=input_matrix[0:s1,10:12] # alpha cp clv-long y
    # input_matrix[l1+1+s2:,0:2]=input_matrix[l1+1+s2:,8:10] # beta cp clv-long b
    # input_matrix[l1+1:l1+1+s2,2:4]=input_matrix[l1+1:l1+1+s2,10:12] # beta cp clv-long y

    # 截取粘贴了交联离子的单肽区域出来，包括linear+clv_long
    linear_clv_long=flat_linear_pep_ions(input_matrix)

    # 然后提取clv-short匹配强度，把clv-short的y放到对应b的位置上，因为b位置无法匹配
    input_matrix[0:s1,4:6]=input_matrix[0:s1,6:8] # alpha cp clv-short y
    input_matrix[l1+1:l1+1+s2,4:6]=input_matrix[l1+1:l1+1+s2,6:8] # beta cp clv-short y
    # 把xlink的b区域提取出来
    clv_short=input_matrix[:,4:6].flatten()

    return np.append(linear_clv_long,clv_short)


# total_lens, len1s, len2s表示输入向量的总长度、pep1向量长度、pep2向量长度
# 分别=(l1-1)+(l2-1)+1、l1-1、l2-1，即比肽段实际长度短1
# 而linksite1s, linksite2s里面存储的是根据肽段实际长度确定的交联位点（下标从0开始）
def CalSim(predictions, truths, total_lens, pep_types, len1s, len2s, linksite1s, linksite2s):
    sims=[]
    for y_pred,y_truth,total_len,pep_type,l1,l2,s1,s2 in zip(predictions, truths, total_lens, pep_types, len1s, len2s, linksite1s, linksite2s):

        packed_y_pred=y_pred[:total_len].detach().cpu().numpy()
        packed_y_truth=y_truth[:total_len].detach().cpu().numpy()

        if pep_type==0 or pep_type==1: # linear pep or non-clv pep pair
            packed_y_pred=flat_linear_pep_ions(packed_y_pred)
            packed_y_truth=flat_linear_pep_ions(packed_y_truth)
        # elif pep_type==1: # non-clv pep pair
        #     packed_y_pred=flat_non_clv_pep_ions(packed_y_pred,l1,l2,s1,s2)
        #     packed_y_truth=flat_non_clv_pep_ions(packed_y_truth,l1,l2,s1,s2)
        else: # clv pep pair
            packed_y_pred=flat_clv_pep_ions(packed_y_pred,l1,l2,s1,s2)
            packed_y_truth=flat_clv_pep_ions(packed_y_truth,l1,l2,s1,s2)

        cursim=pearsonr(packed_y_pred,packed_y_truth)[0]
        sims.append(cursim)
    return sims


# https://vodkazy.cn/2019/12/12/%E5%B0%8F%E8%AE%B0%EF%BC%9A%E5%A4%84%E7%90%86LSTM-embedding%E5%8F%98%E9%95%BF%E5%BA%8F%E5%88%97/
class MyData(torch.utils.data.Dataset):
    def __init__(self, data_seq, data_label, data_title, data_type, data_l1, data_l2, data_s1, data_s2):
        self.data_seq = data_seq
        self.data_label = data_label
        self.data_title = data_title
        self.data_type = data_type
        self.data_l1 = data_l1
        self.data_l2 = data_l2
        self.data_s1 = data_s1
        self.data_s2 = data_s2

    def __len__(self):
        return len(self.data_seq)
    def __getitem__(self, idx):
        return self.data_seq[idx], self.data_label[idx], self.data_title[idx], self.data_type[idx], self.data_l1[idx], self.data_l2[idx], self.data_s1[idx], self.data_s2[idx]

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True) # pack_padded_sequence 要求要按照序列的长度倒序排列
    x = [torch.Tensor(i[0]) for i in data]
    y = [torch.Tensor(i[1]) for i in data]
    x_padded = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = rnn_utils.pad_sequence(y, batch_first=True, padding_value=0)

    xy_length,t_vec,pt_vec,l1_vec,l2_vec,s1_vec,s2_vec=[],[],[],[],[],[],[]
    for d in data:
        xy_length.append(len(d[0]))
        t_vec.append(d[2])
        pt_vec.append(d[3])
        l1_vec.append(d[4])
        l2_vec.append(d[5])
        s1_vec.append(d[6])
        s2_vec.append(d[7])
    return x_padded, y_padded, xy_length,t_vec,pt_vec,l1_vec,l2_vec,s1_vec,s2_vec


# def LoadDataSet(path_match_formatted_pkl, batch_size):
#     fformattedpkl = open(path_match_formatted_pkl, 'rb')
#     T,L,X,Y = pickle.load(fformattedpkl)
#     fformattedpkl.close()

#     X = np.array(X)
#     Y = np.array(Y)

#     logging.info('X.shape:%s'%str(X.shape))
#     logging.info('Y.shape:%s'%str(Y.shape))
    
#     X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#     train_data = MyData(X_train, Y_train)
#     train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#     test_data = MyData(X_test, Y_test)
#     test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#     logging.info('full data size:%d'%len(X))
#     logging.info('train size:%d'%len(X_train))
#     logging.info('test size:%d'%len(X_test))

#     return train_data_loader, test_data_loader



def LoadDataSet(path_match_formatted_pkl, batch_size):
    fformattedpkl = open(path_match_formatted_pkl, 'rb')
    T,PT,L,L1,L2,S1,S2,X,Y = pickle.load(fformattedpkl)
    fformattedpkl.close()

    X = np.array(X)
    Y = np.array(Y)

    logging.info('X.shape:%s'%str(X.shape))
    logging.info('Y.shape:%s'%str(Y.shape))
    
    data = MyData(X, Y, T, PT, L1, L2, S1, S2)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    logging.info('data size:%d'%len(X))

    return data_loader

################### for predict ################


# 读取单肽测试数据，数据格式和match_info.txt一致，只不过没有最后一列
def ReadpFind3PredictFile(pFind3_test_path):
    pass


# 读取CSM信息txt格式，包含匹配/预测强度信息
def ReadpLink2MatchInfoFile(pLink2_path):
    fin=open(pLink2_path)
    lines=fin.readlines()
    fin.close()

    int_idxs=[1,2,10,13]
    float_idxs=[4,5,6]
    dict_idxs=[9,12]
    list_idxs=[14,15]

    CSMs=[]
    mpTitleLines={}
    header=lines[0].strip()

    for i in range(1,len(lines)):
        contents=lines[i].strip().split('\t')
        mpTitleLines[contents[0]]=lines[i].strip() # 保留原始行信息

        for idx in int_idxs:
            contents[idx]=int(contents[idx])
        for idx in float_idxs:
            contents[idx]=float(contents[idx])
        for idx in dict_idxs:
            contents[idx]=eval(contents[idx])
        for idx in list_idxs:
            if idx < len(contents):
                contents[idx]=eval(contents[idx])

        CSMs.append(contents)
    return CSMs,mpTitleLines,header

# 读取交联测试数据,数据格式和match_info.txt一致，只不过没有最后两列
def ReadpLink2PredictFile(pLink2_path):
    CSMs,mpTitleLines,header=ReadpLink2MatchInfoFile(pLink2_path)

    new_CSMs=[]
    new_mpTitleLines={}
    new_header=header.split('\t')[:14]
    new_header='\t'.join(new_header)

    for csm in CSMs:
        new_CSMs.append(csm[:14])
    
    for title,line in mpTitleLines.items():
        new_mpTitleLines[title]='\t'.join(line.split('\t')[:14])
    
    return new_CSMs,new_mpTitleLines,new_header

# pLink2_path=r'/data/zlchen/pDeepXL/data/PXD019926/DSS/8PM/pLink2_data/pLink2_match_info_small_predict.txt'
# CSMs0,mpTitleLines0,header0=ReadpLink2PredictFile(pLink2_path)
# CSMs1,mpTitleLines1,header1=ReadpLink2PredictFile2(pLink2_path)
# print(CSMs0==CSMs1)
# print(mpTitleLines0==mpTitleLines1)
# print(header0==header1)
# pLink2_path=r'/data/zlchen/pDeepXL/data/PXD019926/DSS/8PM/pLink2_data/pLink2_match_info_small.txt'
# CSMs,mpTitleLines,header=ReadpLink2MatchInfoFile(pLink2_path)
# print(CSMs)
# print(mpTitleLines)
# print(header)





class MyPredictData(torch.utils.data.Dataset):
    def __init__(self, data_seq, data_title, data_type, data_l1, data_l2, data_s1, data_s2):
        self.data_seq = data_seq
        self.data_title = data_title
        self.data_type = data_type
        self.data_l1 = data_l1
        self.data_l2 = data_l2
        self.data_s1 = data_s1
        self.data_s2 = data_s2

    def __len__(self):
        return len(self.data_seq)
    def __getitem__(self, idx):
        return self.data_seq[idx], self.data_title[idx], self.data_type[idx], self.data_l1[idx], self.data_l2[idx], self.data_s1[idx], self.data_s2[idx]

def predict_collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True) # pack_padded_sequence 要求要按照序列的长度倒序排列
    x = [torch.Tensor(i[0]) for i in data]
    x_padded = rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)

    xy_length,t_vec,pt_vec,l1_vec,l2_vec,s1_vec,s2_vec=[],[],[],[],[],[],[]
    for d in data:
        xy_length.append(len(d[0]))
        t_vec.append(d[1])
        pt_vec.append(d[2])
        l1_vec.append(d[3])
        l2_vec.append(d[4])
        s1_vec.append(d[5])
        s2_vec.append(d[6])
    return x_padded, xy_length,t_vec,pt_vec,l1_vec,l2_vec,s1_vec,s2_vec


def ConvertPredictDataToDataLoader(batch_size, X, T, PT, L1, L2, S1, S2):
    predict_data = MyPredictData(X, T, PT, L1, L2, S1, S2)
    predict_data_loader = DataLoader(predict_data, batch_size=batch_size, shuffle=False, collate_fn=predict_collate_fn) # 测试数据不shuffle
    print('prediction size:', len(predict_data))
    return predict_data_loader





################## for fine tune #############
# 解析训练过程生成的日志文件

def ParseLineFromLog(line):
    ans=[]
    line=line.split('[INFO]')[1]
    contents=line.split(',')
    for part in contents:
        sep='=' if '=' in part else ':'
        k,v=part.split(sep)
        ans.append(float(v))
    return ans


def SmoothLog(log_info):
    min_idxs=[1,5]
    max_idxs=[3,4,6,7]
    smoothed=[]
    for i, contents in enumerate(log_info):
        if i==0:
            smoothed.append(contents)
        else:
            for idx in min_idxs:
                contents[idx]=min(contents[idx],smoothed[-1][idx])
            for idx in max_idxs:
                contents[idx]=max(contents[idx],smoothed[-1][idx])
            smoothed.append(contents)
    return smoothed

def ParseTrainLog(path_log,smooth=False):
    fin=open(path_log)
    lines=fin.readlines()
    fin.close()

    train_info=[]
    val_info=[]

    for line in lines:
        if 'Epoch:' in line:
            epoch=line.split('|')[0].split('Epoch:')[1].strip()
            epoch=int(epoch)
        elif 'Train Loss:' in line:
            cur_info=[epoch]+ParseLineFromLog(line)
            train_info.append(cur_info)
        elif 'Val. Loss:' in line:
            cur_info=[epoch]+ParseLineFromLog(line)
            val_info.append(cur_info)
    
    if smooth:
        return SmoothLog(train_info), SmoothLog(val_info)
    else:
        return train_info,val_info