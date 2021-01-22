

class PeptideFeaturer:
    # 系统参数
    MIN_PREC_CHARGE, MAX_PREC_CHARGE = 0, 0
    MIN_PEPTIDE_LEN, MAX_PEPTIDE_LEN = 0, 0
    MAX_NCE = 0
    VALID_MODS, VALID_INSTS, VALID_CROSS_LINKERS = [], [], []
    VALID_AA = 'ACDEFGHIKLMNPQRSTVWY-' # 最后一个字符-表示交联剂，用来分割alpha和beta序列

    def __init__(self, config):
        self.MIN_PREC_CHARGE=int(config['DEFAULT']['min_prec_charge'])
        self.MAX_PREC_CHARGE=int(config['DEFAULT']['max_prec_charge'])

        self.MIN_PEPTIDE_LEN=int(config['DEFAULT']['min_peptide_len'])
        self.MAX_PEPTIDE_LEN=int(config['DEFAULT']['max_peptide_len'])

        self.MAX_NCE=float(config['DEFAULT']['max_nce'])

        self.VALID_MODS=config['DEFAULT']['valid_mods'].split(',')
        self.VALID_INSTS=config['DEFAULT']['instruments'].split(',')
        self.VALID_CROSS_LINKERS=config['DEFAULT']['cross_linkers'].split(',')

    def IsSeqValid(self, seq):
        for aa in seq:
            if aa not in self.VALID_AA:
                return False
        return True

    def AA2OneHot(self, aa):
        ans=[0]*len(self.VALID_AA)
        ans[self.VALID_AA.index(aa)]=1
        return ans

    def Mod2OneHot(self, mname=''):
        ans=[0]*len(self.VALID_MODS)
        if mname =='':
            return ans
        ans[self.VALID_MODS.index(mname)]=1
        return ans

    def Charge2OneHot(self, c=0):
        ans=[0]*self.MAX_PREC_CHARGE
        if c==0:
            return ans
        ans[c-1]=1
        return ans

    def Instrument2OneHot(self, inst=''):
        ans=[0]*len(self.VALID_INSTS)
        if inst=='':
            return ans
        ans[self.VALID_INSTS.index(inst)]=1
        return ans

    def CrossLinker2OneHot(self, cross_linker=''):
        ans=[0]*len(self.VALID_CROSS_LINKERS)
        if cross_linker=='':
            return ans
        ans[self.VALID_CROSS_LINKERS.index(cross_linker)]=1
        return ans

    def PaddingZero(self, input_vec):
        n=len(input_vec)
        if n<self.MAX_PEPTIDE_LEN-1:
            for i in range(self.MAX_PEPTIDE_LEN-1-n):
                input_vec.append([0]*len(input_vec[0]))
        return input_vec

    def Add2ListElementWise(self,list1,list2):
        return [sum(x) for x in zip(list1, list2)]

    def LinkerSymbol2Vec(self):
        cur_aa_vec = self.AA2OneHot('-')
        cur_mod_vec = self.Mod2OneHot()
        left_vec = right_vec = left_sum_vec = right_sum_vec = cur_aa_vec + cur_mod_vec
        is_left_n_term, is_right_c_term = 0, 0
        is_left_link_site,is_right_link_site = 0, 0

        charge_vec = self.Charge2OneHot()
        inst_vec = self.Instrument2OneHot()
        cross_linker_vec = self.CrossLinker2OneHot()
        NCE_low,NCE_medium,NCE_high=0,0,0

        clv_vec = left_vec + right_vec + left_sum_vec + right_sum_vec + [is_left_n_term, is_right_c_term] + \
            [is_left_link_site, is_right_link_site] + charge_vec + inst_vec + [NCE_low,NCE_medium,NCE_high] + cross_linker_vec
            
        return clv_vec

    # 如果cross_linker=='',link_site==-1表示非交联
    def Sequence2Vec(self, seq, mod, charge, inst, NCE_low,NCE_medium,NCE_high, cross_linker='', link_site=-1, padding=False):
        NCE_low,NCE_medium,NCE_high=NCE_low/self.MAX_NCE,NCE_medium/self.MAX_NCE,NCE_high/self.MAX_NCE
        lseq = len(seq)
        aa_mod_vec = [] # 每个aa位置的aa+mod的vec
        for i in range(lseq):
            cur_aa_vec = self.AA2OneHot(seq[i])
            cur_mod_vec = self.Mod2OneHot()
            if i in mod:
                cur_mod_vec = self.Mod2OneHot(mod[i])

            cur_vec = cur_aa_vec + cur_mod_vec
            aa_mod_vec.append(cur_vec)

        # 左边aa的累加和，右边aa的累加和
        left_sum_aa_mod_vec, right_sum_aa_mod_vec = [0]*lseq, [0]*lseq
        for i in range(lseq):
            left_idx, right_idx = i, lseq-i-1
            cur_left_vec = aa_mod_vec[left_idx]
            cur_right_vec = aa_mod_vec[right_idx]

            if i == 0:
                left_sum_aa_mod_vec[left_idx]=cur_left_vec
                right_sum_aa_mod_vec[right_idx]=cur_right_vec
            else:
                left_sum_aa_mod_vec[left_idx]=self.Add2ListElementWise(cur_left_vec, left_sum_aa_mod_vec[left_idx-1])
                right_sum_aa_mod_vec[right_idx]=self.Add2ListElementWise(cur_right_vec, right_sum_aa_mod_vec[right_idx+1])

        # 最终向量
        seq_vec=[]
        for i in range(0, lseq-1):
            left_vec,right_vec=aa_mod_vec[i],aa_mod_vec[i+1]
            left_sum_vec,right_sum_vec=left_sum_aa_mod_vec[i],right_sum_aa_mod_vec[i+1]

            is_left_n_term, is_right_c_term = 0, 0
            if i == 0:
                is_left_n_term = 1
            elif i == lseq-2:
                is_right_c_term = 1

            is_left_link_site,is_right_link_site = 0, 0
            if i == link_site:
                is_left_link_site = 1
            elif i+1 == link_site:
                is_right_link_site = 1
            
            charge_vec = self.Charge2OneHot(charge)
            inst_vec = self.Instrument2OneHot(inst)
            cross_linker_vec = self.CrossLinker2OneHot(cross_linker)

            clv_vec = left_vec + right_vec + left_sum_vec + right_sum_vec + [is_left_n_term, is_right_c_term] + \
                [is_left_link_site, is_right_link_site] + charge_vec + inst_vec + [NCE_low,NCE_medium,NCE_high] + cross_linker_vec
            seq_vec.append(clv_vec)

        if padding:
            seq_vec = self.PaddingZero(seq_vec)

        return seq_vec, left_sum_aa_mod_vec[lseq-1] # 同时返回整个肽段的aa_mod累加和，检验过与right_sum_aa_mod_vec[0]相同


    def AddAnotherSeq(self, target_vec, link_site, another_seq_aa_mod_sum_vec):
        vec_len = len(target_vec)
        left_sum_st, left_sum_end = 52,78
        right_sum_st, right_sum_end = 78,104
        for i in range(link_site, vec_len):
            target_vec[i][left_sum_st:left_sum_end] = self.Add2ListElementWise(target_vec[i][left_sum_st:left_sum_end], another_seq_aa_mod_sum_vec)
        for i in range(0,link_site):
            target_vec[i][right_sum_st:right_sum_end] = self.Add2ListElementWise(target_vec[i][right_sum_st:right_sum_end], another_seq_aa_mod_sum_vec)
        return target_vec


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('/data/zlchen/pDeepXL/code/pDeepXL/config.ini')
    pf = PeptideFeaturer(config)
    # seq_vec = pf.Sequence2Vec('MIAKSEQEIGK',{0:'Oxidation[M]',4:'Carbamidomethyl[C]',5:'Carbamidomethyl[C]',7:'Oxidation[M]'},5,'Lumos',0,30,0,'',-1,False)
    seq_vec, left_sum, right_sum = pf.Sequence2Vec('MIAKSEQEIGK',{0:'Oxidation[M]',5:'Carbamidomethyl[C]',7:'Oxidation[M]'},3,'Lumos',0,30,0,'DSS',3,False)
    for clv_vec in seq_vec:
        print(clv_vec)
    print(left_sum==right_sum)
    print(left_sum)
