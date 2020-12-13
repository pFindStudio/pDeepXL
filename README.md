# pDeepXL: MS/MS spectrum prediction for cross-linked peptide pairs by deep learning

## Introduction

In cross-linking mass spectrometry, identification of cross-linked peptide pairs heavily relies on similarity measurements between experimental spectra and theoretical ones. The lack of accurate ion intensities in theoretical spectra impairs the performances of search engines for cross-linked peptide pairs, especially at proteome scales. Here, we introduce pDeepXL, a deep neural network to predict MS/MS spectra of cross-linked peptide pairs. We used the transfer learning technique to train pDeepXL, facilitating the training with limited benchmark data of cross-linked peptide pairs. Test results on over ten datasets showed that pDeepXL accurately predicted spectra of both non-cleavable DSS/BS3/Leiker cross-linked peptide pairs (>80% of predicted spectra have Pearson correlation coefficients (PCCs) higher than 0.9), and cleavable DSSO/DSBU cross-linked peptide pairs (>75% of predicted spectra have PCCs higher than 0.9). Furthermore, we showed that accurate prediction was achieved for unseen datasets using an online fine-tunning technique. Finally, integrating pDeepXL into a database search engine increased the number of identified cross-linked spectra by 18% on average.

## Download

Please download pDeepXL from [http://pfind.ict.ac.cn/software/pDeepXL/pDeepXL.zip](http://pfind.ict.ac.cn/software/pDeepXL/pDeepXL.zip), which contains the source code and test datasets.


## Dependencies

* Python >=3.6.9
* PyTorch >= 1.0.1

Anaconda enviroment is recommended.

## Predict

There are two test datasets in the downloaded pDeepXL.zip, one is for non-cleavable cross-linkers DSS/Leiker (`pDeepXL/datasets/non_cleavable`), and the other is for cleavable cross-linkers DSSO/DSBU (`pDeepXL/datasets/cleavable`). For each dataset, there are 3 folders: the `data` folder contains 1 file with 15 cross-linked peptide pairs, the `model` folder contains 1 file with the trained model, and the `predict_results` folder contains predicted MS/MS spectra and the corresponding images.

You can predict MS/MS spectra for cross-linked peptide pairs with following steps.

0. Activate your PyTorch environment and goto the `pDeepXL/model` folder
1. Run `linear_predict.py` with following parameters to predict:

    ```
    python linear_predict.py path_data_file path_predict_result_file path_model is_non_cleavable
    ```

    where:
    * `path_data_file` is the path of test data file. Please make sure one line for one cross-linked peptide pair, each line contains 14 columns, and the `title` should be unique for each line. For details about the data format, please see the demo data file
    * `path_predict_result_file` is the path of predicted result file
    * `path_model` is the path of model file
    * `is_non_cleavable` is whether the test data is for non-cleavable cross-linkers. `1` is for non-cleavable cross-linkers and `0` is for cleavable cross-linkers

    For example, run the following command for the demo non-cleavable cross-linked dataset (**please use your local paths**):

    ```
    python linear_predict.py ../datasets/non_cleavable/data/non_clv_dataset.txt ../datasets/non_cleavable/predict_results/non_clv_predicted_res.txt ../datasets/non_cleavable/model/non_clv_model.pt 1
    ```

    or run the following command for the demo cleavable cross-linked dataset (**please use your local paths**):

    ```
    python linear_predict.py ../datasets/cleavable/data/clv_dataset.txt ../datasets/cleavable/predict_results/clv_predicted_res.txt ../datasets/cleavable/model/clv_model.pt 0
    ```

2. After that, a predicted result file will be generated in the `path_predict_result_file` you specified. In the result file, the last two columns (`seq1_pred_b1b2y1y2`, `seq2_pred_b1b2y1y2`) are the predicted intensity matrix for alpha peptide and beta peptide, respectively. For non-cleavable cross-linkers, only the first four rows of the intensity matrix are valid, and they are `b+`, `b++`, `y+`, and `y++` successively. For cleavable cross-linkers, all eight rows are valid, and they are `b+`, `b++`, `y+`, `y++`, `bS+`, `bS++`, `yS+`, and `yS++`successively. For more details, please see Supplementary Figure 2 of the manuscript.

3. Goto the `pDeepXL/visualize` folder, and run `plot_csm.py` with following parameters to generate images of predicted MS/MS spectra according to the predicted results:

    ```
    python plot_csm.py path_predict_result_file path_img_folder
    ```

    where:
    * `path_predict_result_file` is the path of predicted result file generated in step 1
    * `path_img_folder` is the path of the folder where you want to place images for predicted MS/MS spectra. If the folder does not exist, we will make the folder for you

    For example, run the following command for the demo non-cleavable cross-linked dataset (**please use your local paths**):

    ```
    python plot_csm.py ../datasets/non_cleavable/predict_results/non_clv_predicted_res.txt ../datasets/non_cleavable/predict_results/imgs
    ```

    or run the following command for the demo cleavable cross-linked dataset (**please use your local paths**):

    ```
    python plot_csm.py ../datasets/cleavable/predict_results/clv_predicted_res.txt ../datasets/cleavable/predict_results/imgs
    ```

4. 
    After that, images of predicted spectra for all cross-linked peptide pairs will be generated in the `path_img_folder` you specified. The name of each image is the `title` of each line in `path_data_file`. Here are two examples, one for DSS and the other for DSSO.

    ![HEK293_DSS_FAIMS_405060_Fr1.36531.36531.4.0.dta.png](datasets/non_cleavable/predict_results/imgs/HEK293_DSS_FAIMS_405060_Fr1.36531.36531.4.0.dta.png)

    ![HEK293_FAIMS_60_70_80_Fr2.32448.32448.3.0.dta.png](datasets/cleavable/predict_results/imgs/HEK293_FAIMS_60_70_80_Fr2.32448.32448.3.0.dta.png)


## Citation

```
pDeepXL: MS/MS spectrum prediction for cross-linked peptide pairs by deep learning, under review.
```