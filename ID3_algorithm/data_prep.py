import pandas as pd
import numpy as np


df = pd.read_csv('data/cardio_train.csv', sep=';', index_col='id')

def discretize(data: pd.core.frame.DataFrame):

    # Discretize the age column into 4 bins (in years): 0-35 = 1, 35-45 = 2, 45-65 = 3
    data['age'] = pd.cut(data['age'], bins=[0, 35*365.25, 45*365.25, 65*365.25], labels=[1, 2, 3])

    # Discretize the systolic blood pressure column into 4 bins: 0-120 = 1, 120-140 = 2, 140-190 = 3
    data['ap_hi'] = pd.cut(data['ap_hi'], bins=[0, 120, 140, 190], labels=[1, 2, 3])

    # Discretize the diastolic blood pressure column into 5 bins: 0-70 = 2, 70-80 = 1, 80-90 = 2, 90-120 = 3
    data['ap_lo'] = pd.cut(data['ap_lo'], bins=[0, 70, 80, 90, 120], labels=[2, 1, 2, 3], ordered=False)

    # Calculate the BMI
    data['height'] = data['weight'] / (data['height']/100)**2
    data.rename(columns={'height': 'bmi'}, inplace=True)
    data.drop(['weight'], axis=1, inplace=True)

    # Discretize the BMI column into 4 bins: 0-24.9 = 1, 24.9-29.9 = 2, 29.9-39.9 = 3
    data['bmi'] = pd.cut(data['bmi'], bins=[0, 24.9, 29.9, 39.9], labels=[1, 2, 3])

    return data

# df = df.iloc[:5, :]
# print(df)
# df = discretize(df)
# print(df)