# 批量画交联的肽谱匹配图

import os
import pDeepXL.utils as utils
import pDeepXL.mass_util as mass_util
import matplotlib
matplotlib.use('AGG')#或者PDF, SVG或PS
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np


# 不可碎裂
def plot_non_clv_single(title,pep_pair,prec_charge,pred_matrix,path_fig):
    LinkerName,seq1,mods1,linksite1,seq2,mods2,linksite2=pep_pair
    alpha_preds,beta_preds=pred_matrix
    alpha_preds,beta_preds=alpha_preds[:4],beta_preds[:4]
    
    l1,l2=len(seq1),len(seq2)

    b1mass1,b2mass1,y1mass1,y2mass1,b1mass2,b2mass2,y1mass2,y2mass2=mass_util.calfragmass4xl(seq1,mods1,linksite1,seq2,mods2,linksite2,LinkerName)

    alpha_ions=[b1mass1,b2mass1,y1mass1,y2mass1]
    beta_ions=[b1mass2,b2mass2,y1mass2,y2mass2]

    ion_charges=(1,2,1,2)
    ion_names=('b+','b++','y+','y++')
    ion_colors=('green','blue','red','orange')

    plt.figure(figsize=(15,8))

    min_mz,max_mz=10000,0

    # plot pred spec
    # alpha
    for ion_mzs, ion_charge, ion_name, ion_color,pred_intens in zip(alpha_ions, ion_charges, ion_names, ion_colors, alpha_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            min_mz=min(min_mz,theo_mz)
            max_mz=max(max_mz,theo_mz)
            relative_intensity=pred_inten
            ion_pos= pos+1 if ion_name[0]=='b' else l1-pos
            ion_txt='α%s%d%s'%(ion_name[0],ion_pos,ion_name[1:])
            plt.plot([theo_mz,theo_mz], [0, relative_intensity], color=ion_color, lw=2)
            plt.text(theo_mz, relative_intensity + 0.03, ion_txt, rotation = 90, color=ion_color, horizontalalignment="center",verticalalignment='bottom')

    # beta
    for ion_mzs, ion_charge, ion_name, ion_color,pred_intens in zip(beta_ions, ion_charges, ion_names, ion_colors, beta_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            min_mz=min(min_mz,theo_mz)
            max_mz=max(max_mz,theo_mz)
            relative_intensity=pred_inten
            ion_pos= pos+1 if ion_name[0]=='b' else l2-pos
            ion_txt='β%s%d%s'%(ion_name[0],ion_pos,ion_name[1:])
            plt.plot([theo_mz,theo_mz], [0, relative_intensity], color=ion_color, lw=2)
            plt.text(theo_mz, relative_intensity + 0.03, ion_txt, rotation = 90, color=ion_color, horizontalalignment="center",verticalalignment='bottom')

    mz_margin=20
    minx,maxx=min_mz-mz_margin,max_mz+mz_margin

    plt.plot([0,maxx],[0,0],color='black',lw=0.5)

    plt.title('predicted spectrum of %s'%title)
    # plt.legend()

    # plt.xlim(0,2800)
    plt.xlim(minx,maxx)
    plt.ylim(0, 1.2)

    # 修改纵坐标
    ax = plt.gca()
    labels = ['%d%%'%(i*10) for i in range(0,14,2)]
    ax.set_yticklabels(labels)

    plt.xlabel('m/z')
    plt.ylabel('Relative Abundance')
    # plt.savefig(path_fig+'.svg', format='svg', dpi=1200)
    plt.savefig(path_fig)


# 可碎裂
def plot_clv_single(title,pep_pair,prec_charge,pred_matrix,path_fig):
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
    #ion_names=('linear_b1','linear_b2','linear_y1','linear_y2','clv_long_b1','clv_long_b2','clv_long_y1','clv_long_y2','clv_short_b1','clv_short_b2','clv_short_y1','clv_short_y2')
    ion_names=('b+','b++','y+','y++','Lb+','Lb++','Ly+','Ly++','Sb+','Sb++','Sy+','Sy++')
    ion_colors=('green','blue','red','orange','green','blue','red','orange','green','blue','red','orange')

    plt.figure(figsize=(15,8))

    min_mz,max_mz=10000,0

    # plot pred spec
    # alpha
    for ion_mzs, ion_charge, ion_name, ion_color,pred_intens in zip(alpha_ions, ion_charges, ion_names, ion_colors, alpha_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            min_mz=min(min_mz,theo_mz)
            max_mz=max(max_mz,theo_mz)
            relative_intensity=pred_inten
            ion_pos= pos+1 if 'b' in ion_name else l1-pos
            ion_txt='α%s%d%s'%(ion_name.split('+')[0],ion_pos,'+'*ion_charge)
            # print('type=%s,theomz=%f,intensity=%f'%(ion_txt,theo_mz,relative_intensity))
            plt.plot([theo_mz,theo_mz], [0, relative_intensity], color=ion_color, lw=2)
            plt.text(theo_mz, relative_intensity + 0.03, ion_txt, rotation = 90, color=ion_color, horizontalalignment="center",verticalalignment='bottom')

    # beta
    for ion_mzs, ion_charge, ion_name, ion_color,pred_intens in zip(beta_ions, ion_charges, ion_names, ion_colors, beta_preds):
        if prec_charge == 2 and ion_charge == 2:
            continue
        for pos,(theo_mz,pred_inten) in enumerate(zip(ion_mzs,pred_intens)):
            if pred_inten==0.0 or pred_inten==0 or theo_mz<=0.0:
                continue
            min_mz=min(min_mz,theo_mz)
            max_mz=max(max_mz,theo_mz)
            relative_intensity=pred_inten
            ion_pos= pos+1 if 'b' in ion_name else l2-pos
            ion_txt='β%s%d%s'%(ion_name.split('+')[0],ion_pos,'+'*ion_charge)
            # print('type=%s,theomz=%f,intensity=%f'%(ion_txt,theo_mz,relative_intensity))
            plt.plot([theo_mz,theo_mz], [0, relative_intensity], color=ion_color, lw=2)
            plt.text(theo_mz, relative_intensity + 0.03, ion_txt, rotation = 90, color=ion_color, horizontalalignment="center",verticalalignment='bottom')

    mz_margin=20
    minx,maxx=min_mz-mz_margin,max_mz+mz_margin
    plt.plot([0,maxx],[0,0],color='black',lw=0.5)

    plt.title('predicted spectrum of %s'%title)
    # plt.legend()

    # plt.xlim(50,1200)
    plt.xlim(minx,maxx)
    plt.ylim(0, 1.2)

    # 修改纵坐标
    ax = plt.gca()
    labels = ['%d%%'%(i*10) for i in range(0,14,2)]
    ax.set_yticklabels(labels)

    plt.xlabel('m/z')
    plt.ylabel('Relative Abundance')
    # plt.savefig(path_fig+'.svg', format='svg', dpi=1200)
    plt.savefig(path_fig)



def plot_batch(path_pLink2_match_info, path_img_folder):
    if not os.path.exists(path_img_folder):
        os.mkdir(path_img_folder)
    print('start plotting...')
    CSMs,mpTitleLines,header=utils.ReadpLink2MatchInfoFile(path_pLink2_match_info)
    for i,csm in enumerate(CSMs):
        title,scan,charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2,seq1_pred_b1b2y1y2,seq2_pred_b1b2y1y2=csm
        print('%d/%d plotting %s...'%(i,len(CSMs),title))
        pep_pair=[crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2]
        pred_matrix=seq1_pred_b1b2y1y2,seq2_pred_b1b2y1y2
        path_fig=r'%s/%s.png'%(path_img_folder,title)
        if crosslinker=='DSSO' or crosslinker=='DSBU':
            plot_clv_single(title,pep_pair,charge,pred_matrix,path_fig)
        elif crosslinker=='DSS' or crosslinker=='Leiker':
            plot_non_clv_single(title,pep_pair,charge,pred_matrix,path_fig)
        else:
            print('do not support %s cross-linker'%crosslinker)
    print('plot done.')



def plot_single(title,prec_charge,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2,pred_matrix,path_fig):
    print('start plotting...')
    pep_pair=crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2
    if crosslinker=='DSSO' or crosslinker=='DSBU':
        plot_clv_single(title,pep_pair,prec_charge,pred_matrix,path_fig)
    elif crosslinker=='DSS' or crosslinker=='Leiker':
        plot_non_clv_single(title,pep_pair,prec_charge,pred_matrix,path_fig)
    else:
        print('do not support %s cross-linker'%crosslinker)
    print('plot done.')