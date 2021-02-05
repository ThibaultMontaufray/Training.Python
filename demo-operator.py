import numpy as np
import pandas as pd

print("Pandas version : " + pd.__version__)

print("- intro ----------------------------")

df_one = pd.DataFrame({'k1':['A', 'A', 'B', 'B', 'C', 'C'],
  'col1':[100, 200, 300, 300, 400, 500],
  'col2':['NY', 'CA', 'WA', 'WA', 'AK', 'NV']})

print(df_one)
print(df_one['k1'].unique())
print(df_one['col2'].value_counts())
print(df_one.drop_duplicates())

def grap_first_letter(state):
    return state[0]

df_one['first letter'] = df_one['col2'].apply(grap_first_letter)
print(df_one)

def complex_state(state):
    if state[0] == 'W':
        return 'Washington'
    else:
        return 'Error'

df_one['complex'] = df_one['col2'].apply(complex_state)
print(df_one)

my_map = {'A':1, 'B':2, 'C':3 }
df_one['num'] = df_one['k1'].map(my_map)
print(df_one)

print(df_one.columns)
df_one.columns = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
print(df_one)

print(df_one.sort_values('C6', ascending=False))

features = pd.DataFrame({'A':[100, 200, 300, 400, 500], 'B':[12, 13, 14, 15, 16]})
prediction = pd.DataFrame({'pred':[0, 1, 1, 0, 1]})
print(pd.concat([features, prediction],axis=1))

print(df_one)
print(pd.get_dummies(df_one['C1']))

print("end of program")
