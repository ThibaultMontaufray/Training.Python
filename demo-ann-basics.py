import numpy as np
import pandas as pd

print("Pandas version : " + pd.__version__)

print("- intro ----------------------------")
df = pd.read_csv('data/fake_reg.csv')
print(df.head())
sns.pairplot(df)

print("end of program")
