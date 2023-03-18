import numpy as np
#1
a = np.random.random((4,2))
print(a)
print("SHAPE : ",a.shape)
print("Array dimensions are ", a.ndim)

#2
b = np.arange(100, 200, 10)
b = b.reshape(5,2)
print (b)

#3
c = np.array([[1.8, 4.6, 5.8],[ 8.6, 7.0, 5.7],[ 5.9, 1.6, 7.2]])
print(c)

#4
a = np.array([[3 ,6, 9, 12], [15 ,18, 21, 24], 
[27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]])
print(a)

d = a[::2, 1::2]
print(d)

#5
e = np.arange(10, 34, 1)
e = e.reshape(8,3)
print (e)
g = np.split(e, 4) 
print(g)

#6
oned = np.arange(1,6)
print(oned)
oned + np.arange(5, 0, -1)

#7
import pandas as pd
data=pd.read_excel('C:/Users/rahul/Downloads/titanic.csv.xlsx')
print(data)
rows, columns=data.shape
print(data.shape[0])
print(data.shape[1])
print(data['survived'].sum())
print(data[data['sex']=='male'].shape[0])
print(data[data['sex']=='female'].shape[0])
print(data[(data['age']>40)&(data['survived']==1)].shape[0])
print(data.isnull().values.any())
data.fillna(pd.NA, inplace=True)
print(data.isnull().values.any())