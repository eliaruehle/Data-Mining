import numpy as np
import pandas as pd
from datasets import Loader
import math
import re
from numpy import nan


def data_interpolation():
    data:Loader = Loader("kp_test") 

    is_list = False
    for strategy in data.get_strategy_names():
        for dataset in data.get_dataset_names():
            for metric in data.get_metric_names():
                frame: pd.DataFrame = data.get_single_dataframe(
                    strategy, dataset, metric
                )

                if frame.shape[1] < 11:
                    continue
                
                is_list = False
                pattern = r'\[(?!\s*\])[\d.,\s]+\]'

                for ind, r in frame.iterrows():
                    if any(re.match(pattern, str(value)) for value in r):
                        is_list = True
                
                if is_list:
                    print("with arrays")
                    frame = interpolation_with_arrays(frame)
                    frame = interpolation_without_arrays(frame)
                    frame.to_csv(f"with_"+ metric +".csv", index=False, sep = ',')
                
                else:
                    if re.search(r'_lag', metric):
                        print("lag metrics")
                        frame = interpolation_lag_metrics(frame)
                        frame.to_csv(f"without_lag_"+ metric +".csv", index=False, sep = ',')
                    
                    else:
                        print("without arrays")  
                        frame = interpolation_without_arrays(frame)        
                        frame.to_csv(f"without_"+ metric +".csv", index=False, sep = ',')

            break
        break

    
def interpolation_with_arrays(frame: pd.DataFrame):
    for index, row in frame.iterrows():
        if any(type(el)==str and el == '[]' for el in row):
            first_empty = int((row == '[]').idxmax())           
        elif any(type(el) == float and math.isnan(el) for el in row):
            first_empty = int(row.isnull().idxmax())
        else: 
            first_empty = 51

        frame = change_arrays_to_values(frame, first_empty)
    return frame


def change_arrays_to_values(frame: pd.DataFrame, first_empty: int):
    for index, row in frame.iterrows():
        for position, value in row.iloc[:first_empty-1].items():
            if type(value) == str and value != '[]':
                data_list = value.strip('[]').split(",")                     
                d = [float(x) for x in data_list]                  
                mean_value = sum(d) / len(d)                   
                frame.loc[index, position] = mean_value
            elif type(value) == float and not math.isnan(value):
                mean_value = np.mean(value)
                frame.loc[index, position] = mean_value   
            else : continue  
    return frame


def interpolation_without_arrays(frame: pd.DataFrame):
    for index, row in frame.iterrows():
        if any(type(el) == float and math.isnan(el) for el in row):
            first_empty = int(row.isnull().idxmax())
        elif any(row == '[]'):
            first_empty = int((row == '[]').idxmax())
        else: continue

        last_not_empty = row[first_empty - 1]
        frame.iloc[index, first_empty:50] = last_not_empty

    return frame


def interpolation_lag_metrics(frame: pd.DataFrame):
    for index, row in frame.iterrows():
        if any(type(el) == float and math.isnan(el) for el in row):
            first_empty = int(row.isnull().idxmax())
        elif any(row == '[]'):
            first_empty = int((row == '[]').idxmax())
        else: continue

        frame.iloc[index, first_empty:50] = 0
    return frame


if __name__ == "__main__":
    data_interpolation()