import pDeepXL.predict
import pDeepXL.plot


path_non_clv_data=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/data/non_clv_dataset.txt'
path_non_clv_prediction_results=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/non_clv_predicted_res.txt'
path_non_clv_img_folder=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/non_cleavable/predict_results/imgs'

non_clv_predictions=pDeepXL.predict.predict_batch(path_non_clv_data, True)
pDeepXL.predict.save_result_batch(path_non_clv_prediction_results, non_clv_predictions)
pDeepXL.plot.plot_batch(path_non_clv_prediction_results, path_non_clv_img_folder)



path_clv_data=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/data/clv_dataset.txt'
path_clv_prediction_results=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/clv_predicted_res.txt'
path_clv_img_folder=r'/data/zlchen/pDeepXL/code/test_pip/pFindStudio/pDeepXL/pDeepXL/examples/cleavable/predict_results/imgs'

clv_predictions=pDeepXL.predict.predict_batch(path_clv_data, False)
pDeepXL.predict.save_result_batch(path_clv_prediction_results, clv_predictions)
pDeepXL.plot.plot_batch(path_clv_prediction_results, path_clv_img_folder)


