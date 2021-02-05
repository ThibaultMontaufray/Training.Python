import numpy as np
import pandas as pd

print("Pandas version : " + pd.__version__)

print("- intro ----------------------------")
labels = ['a', 'b', 'c']
mylist = [1, 2, 3]
arr = np.array([10, 20, 30])
d = {'a':10, 'b':20, 'c':30}

print(pd.Series(data=mylist))
print(pd.Series(arr,labels))
salesQ1 = pd.Series(data=[250,450,200,150],index=['USA', 'China', 'India', 'Brazil'])
salesQ2 = pd.Series(data=[260,500,210,100],index=['USA', 'China', 'India', 'Japan'])
print(salesQ1['China'])
print(salesQ1 + salesQ2)

print("- table ----------------------------")
columns = ['W','X','Y','Z']
index = ['A', 'B', 'C', 'D', 'E']
np.random.seed(42)
data = np.random.randint(-100, 100, (5,4))
df = pd.DataFrame(data,index,columns)
print(df)
print(df['W'])
df['new'] = df['W'] + df['Y']
print(df)
df = df.drop('new',axis=1)
print(df)
print(df.loc[['A', 'E']])
print(df[df > 0])

print("- table ++ -------------------------")
new_ind = ['CA', 'NY', 'WY', 'OR', 'CO']
df['States'] = new_ind
print(df)
print(df.set_index('States'))
print(df.describe())
print(df.info())

print("- null val -------------------------")
df = pd.DataFrame({'A':[1, 2, np.nan, 4], 'B':[5, np.nan, np.nan, 8], 'C':[10, 20, 30, 40]})
print(df)
print(df.dropna(axis=1, thresh=3))
print(df.fillna(value="0"))

print("end of program")
