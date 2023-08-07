import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

# "PM2_5(μg/m3)": ['mean', 'min', 'max', 'std']
# "TEMPERATURE(℃)":  ['mean', 'min', 'max', 'std'],
# "HUMIDITY(%)":  ['mean', 'min', 'max', 'std'],
# "VOC(ppb)":  ['mean', 'min', 'max', 'std'],

# 在外部創一個陣列抓要通用的X欄位
independent_columns = ["TEMPERATURE(℃)", "HUMIDITY(%)", "VOC(ppb)"]
# 在外部創一個陣列抓要通用的Y欄位
dependent_columns = ["PM2_5(μg/m3)"]

class AqDataset(Dataset):
    def __init__(self, data_path):     
        # 如果裡面的變數想跟class的其他地方共用才要綁self
        data = pd.read_csv(data_path)
        # 抓取特定欄位資料 從dataFrame(有欄位)轉換成numpy array(純數字陣列)
        np_x = data[independent_columns].to_numpy(dtype=np.float32)
        # 再轉成pytorch 的tensor 因為pytorch無法直接吃numpy格式
        self.X = torch.tensor(np_x)      
        
        np_y = data[dependent_columns].to_numpy(dtype=np.float32)
        self.Y = torch.tensor(np_y)
        print(self.Y.shape[0])   #可以檢查整個資料的長度跟結構一列有幾欄(向量) 如果要特別某個欄位
        # print()


    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :]


    def __len__(self):       
        return self.Y.shape[0]