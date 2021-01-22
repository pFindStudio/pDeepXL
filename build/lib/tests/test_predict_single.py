import pDeepXL.predict
import pDeepXL.plot



# input example of a non-cleavable cross-linked peptide pair
# ecoli_enri0228_E_bin5_7ul.11740.11740.4.0.dta
prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2=\
4,'QE',0.0,27.0,0.0,'Leiker','EISCVDSAELGKASR',{3: 'Carbamidomethyl[C]'},11,'KIIIGK',{},0

path_non_clv_fig=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predicted_non_clv_spectrum.png'
title='example of non-cleavable cross-linked spectrum'

non_clv_predictions=pDeepXL.predict.predict_single(prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2)
pDeepXL.plot.plot_single(title,prec_charge,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2,non_clv_predictions[2],path_non_clv_fig)



# input example of a cleavable cross-linked peptide pair
# HEK293_FAIMS_60_70_80_Fr2.32448.32448.3.0.dta
prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2=\
3,'Lumos',21.0,27.0,33.0,'DSSO','VLLDVKLK',{},5,'EVASAKPK',{},5

path_clv_fig=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predicted_clv_spectrum.png'
title='example of cleavable cross-linked spectrum'

clv_predictions=pDeepXL.predict.predict_single(prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2)
pDeepXL.plot.plot_single(title,prec_charge,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2,clv_predictions[2],path_clv_fig)

