import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# READING THE DATA 
data=pd.read_csv('C:/Users/rahul/OneDrive/Desktop/breast-cancer.csv')

# ADDING COLUMNS
data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

# CONVERT THE STRING VALUES TO FLOAT 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['age'] = le.fit_transform(data['age'])
data['menopause'] = le.fit_transform(data['menopause'])
data['tumor-size'] = le.fit_transform(data['tumor-size'])
data['node-caps'] = le.fit_transform(data['node-caps'])
data['breast'] = le.fit_transform(data['breast'])
data['breast-quad'] = le.fit_transform(data['breast-quad'])
data['irradiat'] = le.fit_transform(data['irradiat'])
data['inv-nodes'] = le.fit_transform(data['inv-nodes'])

# PRE-PROCESSING THE DATA 
data=data.dropna()
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# IMPLEMENTATION OF VARIOUS MODELS
logistic_regression=LogisticRegression(random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
naive_bayes=GaussianNB() 
 # TRAINING THE MODELS
logistic_regression.fit(X_train, y_train)
knn.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# TESTING THE MODELS
lr_pred = logistic_regression.predict(X_test)
knn_pred = knn.predict(X_test)
nb_pred = naive_bayes.predict(X_test)

# CREATING A CONFUSION MATRIX 
cm_logreg = confusion_matrix(y_test, lr_pred)
cm_knn = confusion_matrix(y_test, knn_pred)
cm_nb = confusion_matrix(y_test, nb_pred)

# TABLE TO SUMMARIZE THE THREE MODELS USED
# Create a dictionary of classifier names and accuracy scores
lr_accuracy = accuracy_score(y_test, lr_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
nb_accuracy = accuracy_score(y_test, nb_pred)
results = {'Logistic Regression': lr_accuracy,
           'KNN': knn_accuracy,
           'Naive Bayes': nb_accuracy}

# Convert the dictionary to a pandas DataFrame
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])

# Save the DataFrame as a CSV file
df_results.to_csv('classifier_comparison.csv')

# Print the DataFrame
print(df_results)


#Q2
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

url = 'https://monkeylearn.com/sentiment-analysis/'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

text = ''
for paragraph in soup.find_all('p'):
    text += paragraph.get_text()
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(text)

print("Text:\n ", text)

print("Sentiment:",end = ' ')
if sentiment['compound'] > 0:
    print('Positive')
elif sentiment['compound'] < 0:
    print('Negative')
else:
    print('Neutral')
    
#Q3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
# Reading data
data = pd.read_csv('C:/Users/rahul/Downloads/CC_GENERAL.csv')
print(data.head())
print(data.describe())
# Removing unwanted columns
data = data.drop(['CUST_ID'], axis = 1)
data = data.fillna(method='ffill')
#Scaling data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
#Printing output inform of plot
le = LabelEncoder()
data_scaled[:, 4] = le.fit_transform(data_scaled[:, 4])
data_scaled[:, 5] = le.fit_transform(data_scaled[:, 5])
#Clustering of data and visualising the data
kmeans = KMeans(n_clusters=5, max_iter=50)
kmeans.fit(data_scaled)

data['cluster'] = kmeans.labels_
data['cluster'].value_counts()

sns.scatterplot(x="PURCHASES", y="CASH_ADVANCE", hue="cluster", data=data)
plt.show() 