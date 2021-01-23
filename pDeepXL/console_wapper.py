def hello_world():
    print("Hello world")

def find_max():
    import sys
    arg1, arg2 = sys.argv[1], sys.argv[2]
    ans=max(int(arg1),int(arg2))
    print(ans)
    return ans



import pDeepXL.predict
import pDeepXL.plot
import sys

def predict_save_batch():
    if len(sys.argv) == 5:
        print('have:', len(sys.argv), ' command args')
        print('arg list:', str(sys.argv))
        path_data_file,is_non_cleavable,path_result_file,save_format=sys.argv[1:]
        print('path_data_file=%s'%path_data_file)
        print('is_non_cleavable=%s'%is_non_cleavable)
        print('path_result_file=%s'%path_result_file)
        print('save_format=%s'%save_format)
        tmp=int(is_non_cleavable.strip())
        is_non_cleavable = True if tmp==1 else False
        print('is_non_cleavable=%d'%is_non_cleavable)
        
        valid_formats=set(['txt','blib','msp'])
        save_format=save_format.strip().lower()
        if save_format not in valid_formats:
            print('sorry, %s format is not supported.'%save_format)
            return
    else:
        print('no command arguments, or #args !=4.') # 用户只需要输入4个参数，对非计算机人员友好
        print('please run the command with path_data_file is_non_cleavable path_result_file save_format.')
        print('please visit https://github.com/pFindStudio/pDeepXL for more details.')
        return
    

    predictions=pDeepXL.predict.predict_batch(path_data_file, is_non_cleavable)
    if save_format=='txt':
        pDeepXL.predict.save_result_batch(path_result_file, predictions)
    else:
        pDeepXL.predict.save_result_batch(path_result_file+'.txt', predictions)
        # save to other format


    
def predict_save_plot_batch():
    if len(sys.argv) == 6:
        print('have:', len(sys.argv), ' command args')
        print('arg list:', str(sys.argv))
        path_data_file,is_non_cleavable,path_result_file,save_format,path_img_folder=sys.argv[1:]
        print('path_data_file=%s'%path_data_file)
        print('is_non_cleavable=%s'%is_non_cleavable)
        print('path_result_file=%s'%path_result_file)
        print('save_format=%s'%save_format)
        tmp=int(is_non_cleavable.strip())
        is_non_cleavable = True if tmp==1 else False
        print('is_non_cleavable=%d'%is_non_cleavable)
        
        valid_formats=set(['txt','blib','msp'])
        save_format=save_format.strip().lower()
        if save_format not in valid_formats:
            print('sorry, %s format is not supported.'%save_format)
            return
    else:
        print('no command arguments, or #args !=5.') # 用户只需要输入5个参数，对非计算机人员友好
        print('please run the command with path_data_file is_non_cleavable path_result_file save_format path_img_folder.')
        print('please visit https://github.com/pFindStudio/pDeepXL for more details.')
        return
    

    predictions=pDeepXL.predict.predict_batch(path_data_file, is_non_cleavable)
    if save_format=='txt':
        path_txt=path_result_file
        pDeepXL.predict.save_result_batch(path_result_file, predictions)
    else:
        path_txt=path_result_file+'.txt'
        pDeepXL.predict.save_result_batch(path_txt, predictions)
        # save to other format

    pDeepXL.plot.plot_batch(path_txt, path_img_folder)
