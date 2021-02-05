import numpy as np
import pandas as pd

print("Pandas version : " + pd.__version__)

print("- intro ----------------------------")
df = pd.read_csv('Universities.csv')
print(df.head())
print(df.groupby(['Year', 'Sector']).sum().sort_index(ascending=False))

print("end of program")
