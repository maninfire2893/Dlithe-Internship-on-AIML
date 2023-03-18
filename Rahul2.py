import pandas as pd
import numpy as np

#1
data_csv=pd.read_csv(r'D:\Dataset\titanic.csv')
print("our dataset...",data_csv)
#2
data=pd.DataFrame(data_csv)

#3
data.dtypes
print(data)

#4
data.columns
attr_list=list(data.columns)
print(attr_list)

#5
def get_attr_index(attr_list):
    age_index=data.columns.get_loc("age") if "age" in data.columns else None
    survival_index=data.columns.get_loc("survived") if "survived" in data.columns else None
    return age_index, survival_index
get_attr_index(attr_list)

#6
data.size

#7
data.info()

#8
data.describe()
 #from the  stats of the dataframe we can analyse that the 'count' gives the number of non-null values for each column
 #'min and 'max' gives us the minimum and maximum value from each column and 25%, 50%, 75% will give percentiles of the values in the dataframe
 
#9
for i in range(10):
     print(data.iloc[i]['age'])
print(data.isna().any())
print(data['age'].isna().sum())
data['age']=np.where(np.isnan(data['age']), np.nanmean(data['age']), data['age'])
data['age']=data['age'].fillna(data['age'].mean())

pass_above_60=data[data['age']>60]
print(pass_above_60)
pass_below_15=data[data['age']<=15]
print(pass_below_15)
age_15=(data['age']<=15).sum()
print(age_15)
pass_male_60=data[(data['age']>60)&(data['sex']=='male')]
print(pass_male_60)