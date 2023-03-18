import statsmodels.api as sm
import pandas as pd

df=pd.read_csv('C:/Users/rahul/Downloads/longley.csv', index_col=0)
df.head()
df.columns

y = df.Employed
X = df.GNP
X = sm.add_constant(X)
model = sm.OLS(y, X)
model = model.fit()
model.summary()
model.fittedvalues
res=model.resid
figure=sm.qqplot(res, fit=True, line='r')


data=pd.read_csv('C:/Users/rahul/Downloads/student-mat.csv - student-mat.csv.csv')
data.head()
data.columns

x=data[['G1','G2','G3']]
y=data[['absences']]

x=sm.add_constant(x)
model1=sm.OLS(y, x)
model1=model1.fit()
model1.summary()
model1.fittedvalues
res1=model1.resid
fig=sm.qqplot(res1, fit=True, line='r')