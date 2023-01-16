import os
from glob import glob
import pandas as pd
import numpy as np
from utils import id2label, label2id

def ensemble(dir_path):
    file_list = glob(dir_path + '/*.csv')
    
    if not file_list:
        print('no file!')
        return 0
    
    dfs = [pd.read_csv(file_name) for file_name in file_list]
    
    ensemble_df = dfs[0].copy()
    ensemble_df = ensemble_df.drop(['predict', 'logits', 'probs'], axis=1)
    
    sum_probs = []
    labels = []
    
    for i in range(len(dfs[0])):
        probs = np.array([0 for _ in range(len(id2label))], dtype=np.float64)
        
        for j in range(len(dfs)):
            probs += np.array(list(dfs[j]['probs'][i][1:-1].split(',')), dtype=np.float64)
            
        label = id2label[np.argmax(probs)]
        
        sum_probs.append(list(probs))
        labels.append(label)
    
    ensemble_df['predict'] = labels
    ensemble_df['probs'] = sum_probs
    
    ensemble_df.to_csv(dir_path + '/ensemble.csv')
    

if __name__ == "__main__":
    # 앙상블 할 파일을 한 폴더에 넣고 폴더 경로 인자로 전달
    ensemble('./outputs/prediction')