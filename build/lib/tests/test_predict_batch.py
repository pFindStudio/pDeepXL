import pDeepXL.predict
import pDeepXL.plot


# --- non cleavable cross-linked example ----
path_non_clv_data_file=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/data/non_clv_dataset.txt'
path_non_clv_result_file=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/non_clv_predicted_res.txt'
path_non_clv_spectra_library_file=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/non_clv_spectra_library.blib'
non_clv_library_format='blib'
path_non_clv_img_folder=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/imgs'

non_clv_predictions=pDeepXL.predict.predict_batch(path_non_clv_data_file, True)
pDeepXL.predict.save_result_batch(path_non_clv_result_file, non_clv_predictions)
# pDeepXL.predict.generate_spectra_library(path_non_clv_spectra_library_file, non_clv_library_format, non_clv_predictions)
pDeepXL.plot.plot_batch(path_non_clv_result_file, path_non_clv_img_folder)



# --- cleavable cross-linked example ----
path_clv_data_file=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/data/clv_dataset.txt'
path_clv_result_file=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/clv_predicted_res.txt'
path_clv_spectra_library_file=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/clv_spectra_library.msp'
clv_library_format='msp'
path_clv_img_folder=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/imgs'

clv_predictions=pDeepXL.predict.predict_batch(path_clv_data_file, False)
pDeepXL.predict.save_result_batch(path_clv_result_file, clv_predictions)
# pDeepXL.predict.generate_spectra_library(path_clv_spectra_library_file, clv_library_format, clv_predictions)
pDeepXL.plot.plot_batch(path_clv_result_file, path_clv_img_folder)

