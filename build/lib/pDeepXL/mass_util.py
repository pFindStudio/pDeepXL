import pDeepXL.utils as utils

import pkg_resources # https://stackoverflow.com/a/16892992

paa=pkg_resources.resource_filename('pDeepXL', 'configs/aa.ini')
pele=pkg_resources.resource_filename('pDeepXL', 'configs/element.ini')

mpLinkerXLMass={'DSS':138.068,'Leiker':316.142,'SS':-2.016,'DSSO':158.004,'DSBU':196.085}
mpClvLinkerLongShortMass={'DSSO':[85.983,54.011],'DSBU':[111.032,85.053]}

mpModMassN14={'Carbamidomethyl[C]':57.021464,'Oxidation[M]':15.994915,'Deamidated[N]':0.984016,
           'Acetyl[ProteinN-term]':42.010565,'Sulfo-NHS-LC-LC-Biotin[AnyN-term]':452.245726,
           'LG-anhydrolactam[AnyN-term]':314.188195,'Biotin_Thermo-21330[AnyN-term]':473.219571,
           'NHS-LC-Biotin[AnyN-term]':339.161662,'LG-anhyropyrrole[AnyN-term]':298.193280,
           'Cytopiloyne+water[AnyN-term]':380.147118,'LG-Hlactam-K[ProteinN-term]':348.193674,
           'bisANS-sulfonates[T]':437.201774,'CLIP_TRAQ_3[AnyN-term]':271.148736,
           'Gln->pyro-Glu[AnyN-termQ]':-17.026547,'Nethylmaleimide[C]':125.047679} # 添加的修饰名称及质量

mpaa=utils.readaaini(paa)
mpele=utils.readeleini(pele)

n14aamass={}

for aa, formula in mpaa.items():
    n14mass=0.0
    for e,n in formula.items():
        n14mass+=(n*mpele[e])
    n14aamass[aa]=n14mass

H2OMass=mpele['H']*2+mpele['O']
PMass=1.0072766

# 计算含水的单肽完整质量
def calpepmass(seq,mods={}):
#    H2OMass=0.0
    n14mass=H2OMass
    for aa in seq:
        n14mass+=n14aamass[aa]
    
    for msite,mname in mods.items():
        n14mass+=mpModMassN14[mname]
    
    return n14mass


# seq='EKYIDQEELNK'
# mods=[]
# n14mass=calpepmass(seq,mods)
# print(n14mass)

# 计算含水的交联完整质量
def calxlpepmass(seq,LinkerName,mods={}):
#    H2OMass=0.0
    linear=''.join([i if i.isalpha() else '' for i in seq])
    n14mass=calpepmass(linear,mods)
    LinkerXLMass=mpLinkerXLMass[LinkerName]
    return n14mass+H2OMass+LinkerXLMass


# seq='DSGKELHINLIPNKQ(3)-EKYIDQEELNK(2)'
# mods=[]
# n14mass=calxlpepmass(seq,mods)
# print(n14mass)


# 计算单肽的碎片离子质量
def calfragmass4regular(seq,mods={}):
    n=len(seq)
    mass=[0.0]*n
    for i,aa in enumerate(seq):
        mass[i]=n14aamass[aa]
    for msite,mname in mods.items():
        mass[msite]+=(mpModMassN14[mname])
        
    b1mass=[0.0]*n
    b1mass[0]=mass[0]+PMass
    for i in range(1,n):
        b1mass[i]=b1mass[i-1]+mass[i]
    y1mass=[0.0]*n
    y1mass[n-1]=mass[n-1]+H2OMass+PMass
    for i in range(n-2,-1,-1):
        y1mass[i]=y1mass[i+1]+mass[i]
    

    b2mass=[0.0]*n
    b3mass=[0.0]*n
    y2mass=[0.0]*n
    y3mass=[0.0]*n
    for i in range(n):
        b2mass[i]=(b1mass[i]+PMass)/2
        b3mass[i]=(b1mass[i]+2*PMass)/3

        y2mass[i]=(y1mass[i]+PMass)/2
        y3mass[i]=(y1mass[i]+2*PMass)/3
        
    b1mass[n-1]=b2mass[n-1]=b3mass[n-1]=y1mass[0]=y2mass[0]=y3mass[0]=-1
    
    # return b1mass,b2mass,b3mass,y1mass,y2mass,y3mass
    return b1mass,b2mass,y1mass,y2mass


# seq='DSGKELHINLIPNKQ'
# mods=[]
# ions=calfragmass4regular(seq,mods)
# for i in ions:
#     print(i)


# 给定1+的mh，计算任意价态的m/z
def cal_mz(mh,charge):
    if mh<0:
        return -1
    else:
        return (mh+PMass*(charge-1))/charge

# 计算可碎裂交联肽段的单条肽段的碎片离子质量
# 交联剂断裂之后，相当于计算单肽的碎片离子质量
def calfragmass4clv(seq,LinkerName,linksite,mods={}):
    long_mass, short_mass=mpClvLinkerLongShortMass[LinkerName]
    n=len(seq)
    mass=[0.0]*n
    for i,aa in enumerate(seq):
        mass[i]=n14aamass[aa]
    for msite,mname in mods.items():
        mass[msite]+=(mpModMassN14[mname])
    
    #----b ion----
    linear_b1mass=[0.0]*n
    linear_b1mass[0]=mass[0]+PMass
    for i in range(1,n):
        linear_b1mass[i]=linear_b1mass[i-1]+mass[i]

    clv_long_b1mass=[0.0]*n
    clv_short_b1mass=[0.0]*n
    for i in range(n):
        if i>=linksite:
            clv_long_b1mass[i]=linear_b1mass[i]+long_mass
            clv_short_b1mass[i]=linear_b1mass[i]+short_mass
            linear_b1mass[i]=-1
        else:
            clv_long_b1mass[i]=-1
            clv_short_b1mass[i]=-1

    #---y ion---
    linear_y1mass=[0.0]*n
    linear_y1mass[n-1]=mass[n-1]+H2OMass+PMass
    for i in range(n-2,-1,-1):
        linear_y1mass[i]=linear_y1mass[i+1]+mass[i]
    

    clv_long_y1mass=[0.0]*n
    clv_short_y1mass=[0.0]*n
    for i in range(n):
        if i <= linksite:
            clv_long_y1mass[i]=linear_y1mass[i]+long_mass
            clv_short_y1mass[i]=linear_y1mass[i]+short_mass
            linear_y1mass[i]=-1
        else:
            clv_long_y1mass[i]=-1
            clv_short_y1mass[i]=-1
    # 2+
    linear_b2mass=[0.0]*n
    clv_long_b2mass=[0.0]*n
    clv_short_b2mass=[0.0]*n

    linear_y2mass=[0.0]*n
    clv_long_y2mass=[0.0]*n
    clv_short_y2mass=[0.0]*n

    for i in range(n):
        linear_b2mass[i]=cal_mz(linear_b1mass[i],2)
        linear_y2mass[i]=cal_mz(linear_y1mass[i],2)

        clv_long_b2mass[i]=cal_mz(clv_long_b1mass[i],2)
        clv_long_y2mass[i]=cal_mz(clv_long_y1mass[i],2)

        clv_short_b2mass[i]=cal_mz(clv_short_b1mass[i],2)
        clv_short_y2mass[i]=cal_mz(clv_short_y1mass[i],2)

    linear_b1mass[n-1]=linear_b2mass[n-1]=linear_y1mass[0]=linear_y2mass[0]=-1
    clv_long_b1mass[n-1]=clv_long_b2mass[n-1]=clv_long_y1mass[0]=clv_long_y2mass[0]=-1
    clv_short_b1mass[n-1]=clv_short_b2mass[n-1]=clv_short_y1mass[0]=clv_short_y2mass[0]=-1
    
    return linear_b1mass,linear_b2mass,linear_y1mass,linear_y2mass,clv_long_b1mass,clv_long_b2mass,clv_long_y1mass,clv_long_y2mass,clv_short_b1mass,clv_short_b2mass,clv_short_y1mass,clv_short_y2mass


# seq='GLSDGEWQQVLNVWGK'
# ions=calfragmass4clv(seq,'DSSO',15)
# for i in ions:
#     print(i)



def calonepep4xl(seq,mods,linksite,anothermass,LinkerName):
    LinkerXLMass=mpLinkerXLMass[LinkerName]
    n=len(seq)
    mass=[0.0]*n
    for i,aa in enumerate(seq):
        mass[i]=n14aamass[aa]
        
    try:
        for msite,mname in mods.items():
            mass[msite]+=(mpModMassN14[mname])
        mass[linksite]+=(LinkerXLMass+anothermass)
    except IndexError: #可能交联位点错误或者修饰位点错误
        print('linksite error, seq=%s,linksite=%d,mods=%s'%(seq,linksite,str(mods)))
    
    b1mass=[0.0]*n
    b1mass[0]=mass[0]+PMass
    for i in range(1,n):
        b1mass[i]=b1mass[i-1]+mass[i]
    y1mass=[0.0]*n
    y1mass[n-1]=mass[n-1]+H2OMass+PMass
    for i in range(n-2,-1,-1):
        y1mass[i]=y1mass[i+1]+mass[i]
        
    b2mass=[0.0]*n
    b3mass=[0.0]*n
    y2mass=[0.0]*n
    y3mass=[0.0]*n
    for i in range(n):
        b2mass[i]=(b1mass[i]+PMass)/2
        b3mass[i]=(b1mass[i]+2*PMass)/3
        
        y2mass[i]=(y1mass[i]+PMass)/2
        y3mass[i]=(y1mass[i]+2*PMass)/3
        
        
    b1mass[n-1]=b2mass[n-1]=b3mass[n-1]=y1mass[0]=y2mass[0]=y3mass[0]=-1
    
    return b1mass,b2mass,b3mass,y1mass,y2mass,y3mass


def calfragmass4xl(seq1,mods1,linksite1,seq2,mods2,linksite2,LinkerName):
    n14mass1=calpepmass(seq1)
    for msite,mname in mods1.items():
        n14mass1+=mpModMassN14[mname]
        
    n14mass2=calpepmass(seq2)
    for msite,mname in mods2.items():
        n14mass2+=mpModMassN14[mname]
    
    b1mass1,b2mass1,b3mass1,y1mass1,y2mass1,y3mass1=calonepep4xl(seq1,mods1,linksite1,n14mass2,LinkerName)
    b1mass2,b2mass2,b3mass2,y1mass2,y2mass2,y3mass2=calonepep4xl(seq2,mods2,linksite2,n14mass1,LinkerName)
    
    # return b1mass1,b2mass1,b3mass1,y1mass1,y2mass1,y3mass1,b1mass2,b2mass2,b3mass2,y1mass2,y2mass2,y3mass2
    return b1mass1,b2mass1,y1mass1,y2mass1,b1mass2,b2mass2,y1mass2,y2mass2



# seq1='GKSDNVPSEEVVK'
# mods1={}
# linksite1=1
# seq2='VKYVTEGMR'
# mods2={7:'Oxidation[M]'}
# linksite2=1
# LinkerName='DSS'

# ions=calfragmass4xl(seq1,mods1,linksite1,seq2,mods2,linksite2,LinkerName)
# for i in ions:
#     print(i)
