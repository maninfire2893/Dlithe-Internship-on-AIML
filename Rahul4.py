import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

url = "http://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv"
df = pd.read_csv(url)

X = df[['GNP.deflator', 'GNP', 'Unemployed', 'Armed.Forces', 'Population', 'Year']]
y = df['Employed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)