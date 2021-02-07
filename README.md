# pDeepXL: MS/MS spectrum prediction for cross-linked peptide pairs by deep learning

Table of Contents
=================
* [Introduction](#introduction)
* [Installation](#installation)
* [Script mode](#script-mode)
    * [Single prediction](#single-prediction)
        * [pDeepXL.predict.predict_single](#pdeepxlpredictpredict_single)
        * [pDeepXL.plot.plot_single](#pdeepxlplotplot_single)
        * [Demonstration](#demonstration)
    * [Batch prediction](#batch-prediction)
        * [pDeepXL.predict.predict_batch](#pdeepxlpredictpredict_batch)
        * [pDeepXL.predict.save_result_batch](#pdeepxlpredictsave_result_batch)
        * [pDeepXL.plot.plot_batch](#pdeepxlplotplot_batch)
        * [pDeepXL.predict.generate_spectra_library](#pdeepxlpredictgenerate_spectra_library)
        * [Demonstration](#demonstration-1)
* [Command line mode](#command-line-mode)
    * [Batch prediction](#batch-prediction-1)
        * [pDeepXL_predict_save_batch](#pdeepxl_predict_save_batch)
        * [pDeepXL_predict_save_plot_batch](#pdeepxl_predict_save_plot_batch)
        * [Demonstration](#demonstration-2)
* [Citation](#citation)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

## Introduction

In cross-linking mass spectrometry, identification of cross-linked peptide pairs heavily relies on similarity measurements between experimental spectra and theoretical ones. The lack of accurate ion intensities in theoretical spectra impairs the performances of search engines for cross-linked peptide pairs, especially at proteome scales. Here, we introduce pDeepXL, a deep neural network to predict MS/MS spectra of cross-linked peptide pairs. We used the transfer learning technique to train pDeepXL, facilitating the training with limited benchmark data of cross-linked peptide pairs. Test results on over ten datasets showed that pDeepXL accurately predicted spectra of both non-cleavable DSS/BS3/Leiker cross-linked peptide pairs (>80% of predicted spectra have Pearson correlation coefficients (PCCs) higher than 0.9), and cleavable DSSO/DSBU cross-linked peptide pairs (>75% of predicted spectra have PCCs higher than 0.9). Furthermore, we showed that accurate prediction was achieved for unseen datasets using an online fine-tunning technique. Finally, integrating pDeepXL into a database search engine increased the number of identified cross-linked spectra by 18% on average.

## Installation

Please install pDeepXL from PyPI. During installation, all required dependencies will be installed automatically. 

```shell
pip install pDeepXL
```

Please also download example datasets from [here](https://github.com/pFindStudio/pDeepXL/raw/master/pDeepXL/examples/examples.zip), which will be used in the following tutorial. There are two example datasets in the downloaded zip file, one is for non-cleavable cross-linkers DSS/Leiker (`examples/non_cleavable`), and the other is for cleavable cross-linkers DSSO/DSBU (`examples/cleavable`). For each dataset, there are 2 folders: the `data` folder contains 1 file with 15 cross-linked peptide pairs, and the `predict_results` folder contains predicted MS/MS spectra, spectra library, and the corresponding images.

## Script mode

For developers, pDeepXL can be easily integrated into a new python project. Once installation, import pDeepXL using two lines:

```python
import pDeepXL.predict
import pDeepXL.plot
```

### Single prediction

#### pDeepXL.predict.predict_single

Use the function `pDeepXL.predict.predict_single` to predict a spectrum for a single cross-linked peptide pair.

```python
predictions=pDeepXL.predict.predict_single(prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2)
```

The arguments contain information about the input cross-linked peptide pair:

* **prec_charge** (int): the precursor charge of the cross-linked peptide pair. Only charges in [2+, 5+] are supported.
* **instrument** (str): the mass spectrometer name. Only instruments in ['QEPlus','QE','QEHF','QEHFX','Fusion','Lumos'] are supported.
* **NCE_low, NCE_medium, NCE_high** (floats): the low, medium, and high normalized collision energies (NCE). Only NCEs in [0.0, 100.0] are supported. If single NCE was used, please set it as NCE_medium, and set the NCE_low and NCE_high as zeros. If stepped-NCE was used, please set three NCEs accordingly.
* **crosslinker** (str): the cross-linker name. Only cross-linkers in ['DSS','Leiker','DSSO','DSBU'] are supported.
* **seq1** (str): the first sequence.
* **mods1** (dict): the modifications on the first sequence, where the key is the position (zero-based numbering) of a modification, and the value is the corresponding modification name. For example, `{3: 'Carbamidomethyl[C]'}` means Carbamidomethyl modified the 4th Cys. Only modifications in ['Carbamidomethyl[C]','Oxidation[M]'] are support.
* **linksite1** (int): the cross-linked site of the first sequence (also zero-based numbering).
* **seq2** (str): same description to **seq1**.
* **mods2** (dict): same description to **mods1**.
* **linksite2** (int): same description to **linksite1**.


Return value is a tuple containing 3 elements, where the last one is the predicted intensity matrix, which can be used to plot the predicted spectrum subsequently.

#### pDeepXL.plot.plot_single

Use the function `pDeepXL.plot.plot_single` to plot a single predicted spectrum.

```
pDeepXL.plot.plot_single(title,prec_charge,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2,predictions[2],path_fig)
```

The arguments contain information about the input cross-linked peptide pair:

* **title** (str): the title of the predicted spectrum.
* **prec_charge,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2**: same descriptions to those for `pDeepXL.predict.predict_single`.
* **predictions[2]** (tuple): the last element of the returned value of `pDeepXL.predict.predict_single`, and the tuple contains predicted intensity matrices for the first and the second sequences.
* **path_fig** (str): the path of the figure to be generated.

#### Demonstration

For example, run the following python script to predict and plot the demo non-cleavable cross-linked peptide pair (**please use your local path**):

```python
# input example of a non-cleavable cross-linked peptide pair
# ecoli_enri0228_E_bin5_7ul.11740.11740.4.0.dta
prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2=\
4,'QE',0.0,27.0,0.0,'Leiker','EISCVDSAELGKASR',{3: 'Carbamidomethyl[C]'},11,'KIIIGK',{},0
# please use your local path
path_non_clv_fig=r'/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predicted_non_clv_spectrum.png'
title='example of non-cleavable cross-linked spectrum'

non_clv_predictions=pDeepXL.predict.predict_single(prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2)
pDeepXL.plot.plot_single(title,prec_charge,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2,non_clv_predictions[2],path_non_clv_fig)
```

Run the following python script to predict and plot the demo cleavable cross-linked peptide pair (**please use your local path**):

```python
# input example of a cleavable cross-linked peptide pair
# HEK293_FAIMS_60_70_80_Fr2.32448.32448.3.0.dta
prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2=\
3,'Lumos',21.0,27.0,33.0,'DSSO','VLLDVKLK',{},5,'EVASAKPK',{},5
# please use your local path
path_clv_fig=r'/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predicted_clv_spectrum.png'
title='example of cleavable cross-linked spectrum'

clv_predictions=pDeepXL.predict.predict_single(prec_charge,instrument,NCE_low,NCE_medium,NCE_high,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2)
pDeepXL.plot.plot_single(title,prec_charge,crosslinker,seq1,mods1,linksite1,seq2,mods2,linksite2,clv_predictions[2],path_clv_fig)
```

### Batch prediction

If you want to predict spectra for many cross-linked peptide pairs, batch prediction is a better and more efficient way to do this. Before batch prediction, please prepare a data file containing all cross-linked peptide pairs you want to predict. In the data file, one line for one cross-linked peptide pair, and the column header is: `title	scan	charge	instrument	NCE_low	NCE_medium	NCE_high	crosslinker	seq1	mods1	linksite1	seq2	mods2	linksite2`, which is separated by the tab `\t`. These parameters have been described in the [Single prediction](#Single-prediction) section. Below is a demo table, and you can find the example non-cleavable data file from [here](https://github.com/pFindStudio/pDeepXL/blob/master/pDeepXL/examples/non_cleavable/data/non_clv_dataset.txt), and the example cleavable data file from [here](https://github.com/pFindStudio/pDeepXL/blob/master/pDeepXL/examples/cleavable/data/clv_dataset.txt).


|title|scan|charge|instrument|NCE_low|NCE_medium|NCE_high|crosslinker|seq1|mods1|linksite1|seq2|mods2|linksite2|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|C_Lee_011216_ymitos_WT_Gly_BS3_XL_12_R2.57721.57721.4.0.dta|57721|4|Fusion|0.0|30.0|0.0|DSS|FKYAPGTIVLYAER|{}|1|INELTLLVQKR|{}|9|
|C_Lee_090916_ymitos_BS3_XL_B13_C1_13_Rep1.14188.14188.3.0.dta|14188|3|Lumos|0.0|30.0|0.0|DSS|KLEDAEGQENAASSE|{}|0|DINLLKNGK|{}|5|
|ecoli_enri0302_E_bin8_7ul_re.5306.5306.3.0.dta|5306|3|QE|0.0|27.0|0.0|Leiker|LKEIIHQQMGGLR|{8: 'Oxidation[M]'}|1|KPNACK|{4: 'Carbamidomethyl[C]'}|0|
|ecoli_enri0228_E_bin5_7ul.11740.11740.4.0.dta|11740|4|QE|0.0|27.0|0.0|Leiker|EISCVDSAELGKASR|{3: 'Carbamidomethyl[C]'}|11|KIIIGK|{}|0|

#### pDeepXL.predict.predict_batch

Use the function `pDeepXL.predict.predict_batch` for batch prediction.

```python
predictions=pDeepXL.predict.predict_batch(path_data_file, is_non_cleavable)
```

The arguments contain information about the input data:

* **path_data_file** (str): the path of the data file, whose format likes the table above, and please make sure the `title` is unique for each line.
* **is_non_cleavable** (bool): whether the data is cross-linked by non-cleavable or cleavable cross-linkers. `True` for non-cleavable and `False` for cleavable. Please note that one data file could not contain both non-cleavable and cleavable cross-linked peptide pairs.

Return value is a tuple containing all predicted spectra. 

#### pDeepXL.predict.save_result_batch

Before spectra plot, please save prediction results to a file, which will be used to plot spectra subsequently. Use the function `pDeepXL.predict.save_result_batch` to save the batch prediction results.

```python
pDeepXL.predict.save_result_batch(path_result_file, predictions)
```

The arguments contain information about the result file path and the prediction results:

* **path_result_file** (str): the path of the result file to be generated.
* **predictions** (tuple): the batch prediction results returned by the function `pDeepXL.predict.predict_batch`.

#### pDeepXL.plot.plot_batch

Then, use the function `pDeepXL.plot.plot_batch` to batch plot all predicted spectra.

```python
pDeepXL.plot.plot_batch(path_result_file, path_img_folder)
```

The arguments contain information about the result file path and the image folder path:

* **path_result_file** (str): the path of the result file generated by the function `pDeepXL.predict.save_result_batch`.
* **path_img_folder** (str): the path of the image folder about to contain all predicted spectra.

#### pDeepXL.predict.generate_spectra_library

In batch prediction mode, you can also save the prediction results to a spectra library file. Supported spectra library format includes `*.blib` and `*.msp`.

Use the function `pDeepXL.predict.generate_spectra_library` to generate a spectra library file from prediction results.

```python
pDeepXL.predict.generate_spectra_library(path_spectra_library_file, library_format, predictions)
```

The arguments contain information about the spectra library file and the prediction results:

* **path_spectra_library_file** (str): the path of the spectra library file to be generated.
* **library_format** (str): the spectra library format. Only formats in ['blib','msp'] are supported. Please see [here](https://skyline.ms/wiki/home/software/BiblioSpec/page.view?name=BiblioSpec%20input%20and%20output%20file%20formats) for description of `*.blib` format, and see [here](http://www.matrixscience.com/msparser/help/group__spectral_library_classes.html) for description of `*.msp` format.
* **predictions** (tuple): the batch prediction results returned by the function `pDeepXL.predict.predict_batch`.

Please note that the current version of `*.blib` and `*.msp` formats have no definition for a cross-linked peptide pair. We use `seq1(linksite1)-crosslinker-seq2(linksite2)` to represent a cross-linked peptide pair, where the `linksite` starts from 0 (zero-based numbering). For example, `DFWSNFKEEVK(6)-DSSO-HFGKIINK(3)` means that the 7th site of peptide `DFWSNFKEEVK` crosslinks to the 4th site of peptide `HFGKIINK` by `DSSO`.

#### Demonstration

For example, run the following python script to batch predict and plot the demo non-cleavable cross-linked dataset, and then save the prediction results to a `blib` spectra library file (**please use your local path**):

```python
# --- non cleavable cross-linked example ----
# please use your local path
path_non_clv_data_file=r'/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/data/non_clv_dataset.txt'
path_non_clv_result_file=r'/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/non_clv_predicted_res.txt'
path_non_clv_img_folder=r'/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/imgs'

non_clv_predictions=pDeepXL.predict.predict_batch(path_non_clv_data_file, True)
pDeepXL.predict.save_result_batch(path_non_clv_result_file, non_clv_predictions)
pDeepXL.plot.plot_batch(path_non_clv_result_file, path_non_clv_img_folder)

non_clv_library_format='blib'
path_non_clv_spectra_library_file=r'/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/non_clv_spectra_library.blib'
pDeepXL.predict.generate_spectra_library(path_non_clv_spectra_library_file, non_clv_library_format, non_clv_predictions)
```

Run the following python script to batch predict and plot the demo cleavable cross-linked dataset, and then save the prediction results to a `msp` spectra library file (**please use your local path**):

```python
# --- cleavable cross-linked example ----
# please use your local path
path_clv_data_file=r'/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/data/clv_dataset.txt'
path_clv_result_file=r'/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/clv_predicted_res.txt'
path_clv_img_folder=r'/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/imgs'

clv_predictions=pDeepXL.predict.predict_batch(path_clv_data_file, False)
pDeepXL.predict.save_result_batch(path_clv_result_file, clv_predictions)
pDeepXL.plot.plot_batch(path_clv_result_file, path_clv_img_folder)

clv_library_format='msp'
path_clv_spectra_library_file=r'/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/clv_spectra_library.msp'
pDeepXL.predict.generate_spectra_library(path_clv_spectra_library_file, clv_library_format, clv_predictions)
```

## Command line mode

For ordinary users who know little about programming, pDeepXL also provides the interactive command line mode. Only batch prediction is available for the command line mode.

### Batch prediction

After installation of pDeepXL from PyPI, pDeepXL provides two command line entry points. You can run pDeepXL directly in the command line window without python programming.

#### pDeepXL_predict_save_batch

Use the command `pDeepXL_predict_save_batch` to batch predict and save the prediction results to file or spectra library.

```shell
pDeepXL_predict_save_batch path_data_file is_non_cleavable path_result_file save_format
```

The command accepts four arguments:

* **path_data_file** (str): the path of the data file.
* **is_non_cleavable** (int): whether the data is cross-linked by non-cleavable or cleavable cross-linkers. 1 for non-cleavable and 0 for cleavable.
* **path_result_file** (str): the path of the prediction result file to be generated.
* **save_format** (str): the format of the prediction result file. If you want to generate spectra library file, set it as `blib` or `msp`, otherwise, just set it as `txt`.


#### pDeepXL_predict_save_plot_batch

Use the command `pDeepXL_predict_save_plot_batch` if you also want to batch plot all predicted spectra.

```shell
pDeepXL_predict_save_plot_batch path_data_file is_non_cleavable path_result_file save_format path_img_folder
```

The command accepts five arguments, including four arguments same to the command `pDeepXL_predict_save_batch`:

* **path_img_folder** (str): the path of the image folder about to contain all predicted spectra.


#### Demonstration

For example, run the following command to batch predict the demo non-cleavable cross-linked dataset, and then save the prediction results to a `msp` spectra library file (**please use your local path**):

```shell
# --- non cleavable cross-linked example ----
# please use your local path
pDeepXL_predict_save_batch /pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/data/non_clv_dataset.txt 1 /pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/non_clv_predicted_res.txt msp
```

Run the following command to batch predict and plot the demo cleavable cross-linked dataset, and DO NOT save the prediction results to a spectra library file (**please use your local path**):

```python
# --- cleavable cross-linked example ----
# please use your local path
pDeepXL_predict_save_plot_batch /pFindStudio/pDeepXL/pDeepXL/examples/cleavable/data/clv_dataset.txt 0 /pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/clv_predicted_res.txt txt /pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/imgs
```

 Here are two examples of predicted spectra, one for DSS and the other for DSSO.

![HEK293_DSS_FAIMS_405060_Fr1.36531.36531.4.0.dta.png](https://github.com/pFindStudio/pDeepXL/raw/master/pDeepXL/examples/non_cleavable/predict_results/imgs/HEK293_DSS_FAIMS_405060_Fr1.36531.36531.4.0.dta.png)

![HEK293_FAIMS_60_70_80_Fr2.32448.32448.3.0.dta.png](https://github.com/pFindStudio/pDeepXL/raw/master/pDeepXL/examples/cleavable/predict_results/imgs/HEK293_FAIMS_60_70_80_Fr2.32448.32448.3.0.dta.png)


## Citation

```
pDeepXL: MS/MS spectrum prediction for cross-linked peptide pairs by deep learning. under review.
```