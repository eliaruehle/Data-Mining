import numpy as np
import pandas as pd

# create a large dataframe with some NaN values
df = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5, 6],
        "B": [7, 8, 9, 10, 11, np.nan],
        "C": [np.nan, 13, 14, 15, 16, 17],
        "D": [18, 19, 20, 21, np.nan, 23],
    }
)

# remove rows that are entirely NaN
df = df.dropna(how="all")

# print the resulting dataframe
print(df)

# Replace the old Iris.csv with a fixed version
data = pd.read_csv("./kp_test/datasets/Iris.csv")
print(data)

data.drop("0", inplace=True, axis=1)
data.to_csv("Iris.csv", index=False)
