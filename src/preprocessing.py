import numpy as np
import pandas as pd
from datasets import Loader
import math
import re
from numpy import nan


def data_interpolation():
    data:Loader = Loader("kp_test") 

    helper = False
    for strategy in data.get_strategy_names():
        for dataset in data.get_dataset_names():
            for metric in data.get_metric_names():
                frame: pd.DataFrame = data.get_single_dataframe(
                    strategy, dataset, metric
                )
                if metric == 'y_pred_test' or metric == 'y_pred_train':
                    continue

                helper = False
                pattern = r'\[(?!\s*\])[\d.,\s]+\]'

                for ind, r in frame.iterrows():
                    if any(re.match(pattern, str(value)) for value in r):
                        helper = True
                
                if helper:
                    #print("with arrays")
                    print(metric, strategy, dataset)

                    frame = interpolation_with_arrays(frame)
                    frame = change_arrays_to_values(frame)
                    frame.to_csv(f"with_"+ metric +".csv", index=False, sep = ',')

                else: 
                    #print("without arrays")
                    frame = interpolation_without_arrays(frame)
                    frame.to_csv(f"without_"+ metric +".csv", index=False, sep = ',')

            break
        break
    print(frame.shape)

    

def interpolation_with_arrays(frame: pd.DataFrame):
    for index, row in frame.iterrows():
        if any(type(el)==str and el == '[]' for el in row):
            first_empty = int((row == '[]').idxmax())
        elif any(type(el) == float and math.isnan(el) for el in row):
            first_empty = np.argwhere(np.isnan(row))
            if len(first_empty) > 0:
                first_empty = first_empty[0][0]
                print(type(first_empty))
        else: continue
        filler = np.stack(row.apply(pd.eval).iloc[first_empty - 5: first_empty - 1].values).mean(axis = 0) #last four valid arrays                  
        numbers_string = np.array2string(filler, separator=',')
        frame.iloc[index, first_empty - 1:50] = pd.Series([numbers_string for _ in row.iloc[first_empty - 1:50]])
        
    return frame


def interpolation_without_arrays(frame: pd.DataFrame):
    for index, row in frame.iterrows():
        if any(type(el) == float and math.isnan(el) for el in row):
            first_empty = np.argwhere(np.isnan(row))
            if len(first_empty) > 0:
                first_empty = first_empty[0][0]
            filler = np.mean(row[first_empty-5:first_empty-1])
            frame.iloc[index, first_empty - 1:50] = filler
        elif any(row == '[]'):
            first_empty = int((row == '[]').idxmax())
            filler = row.apply(pd.eval).iloc[first_empty- 5: first_empty - 1].values.mean(axis = 0)
            frame.iloc[index, first_empty - 1:50] = filler
            print(first_empty)
        else: continue
    return frame


def change_arrays_to_values(frame: pd.DataFrame):
    for index, row in frame.iterrows():
        for position, value in row.iloc[:50].items():  
            print(index, position)                   
            print(type(value), value)
            data_list = value.strip('[]').split(",")                     
            d = [float(x) for x in data_list]                  
            mean_value = sum(d) / len(d)                   
            frame.loc[index, position] = mean_value
    return frame


if __name__ == "__main__":
    data_interpolation()