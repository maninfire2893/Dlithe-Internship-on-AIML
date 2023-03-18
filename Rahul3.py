import numpy as np
import pandas as pd

#1
data= pd.read_csv('D:\Dataset\Bengaluru_House_Data - Bengaluru_House_Data.csv')
print("Our data.....",data)

df=pd.DataFrame(data)
print(df)

print("Size : ",data.size)

print("Shape : ",data.shape)

print("Info :",data.info)

#2
missing_values=data.isna().sum()
data['location'].fillna(data['location'].mode()[0], inplace=True)
#The Location for the houses are important part of the dataset and can't be left blank
data['size'].fillna(data['size'].mode()[0],inplace=True)
data['society'].fillna(data['society'].mode()[0],inplace=True)
data['bath'].fillna(data['bath'].mode()[0],inplace=True)
data['balcony'].fillna(data['balcony'].mode()[0],inplace=True)
missing_values=data.isna().sum()
print(missing_values)

#3
import matplotlib.pyplot as plt
num_features = data.select_dtypes(include=['int64', 'float64']).columns
for feature in num_features:
    fig, ax = plt.subplots()
    ax.hist(data[feature], bins=30)
    ax.set_title(feature)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    plt.show()
    
corr_matrix = data.corr()
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, cmap='coolwarm')
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)
ax.set_title('Correlation matrix')
fig.colorbar(im)
plt.show()

#4
plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.scatter(data['total_sqft'], data['price'])
plt.xlabel('Total SqFt')
plt.ylabel('Price')
plt.title('Scatterplot of Total SqFt vs Price')

data=data[(data['price']<2500)]

#5
data.describe()
data.to_csv('D:\Dataset\housing_prices.csv', index=False)
df=pd.read_csv('D:\Dataset\housing_prices.csv') 


