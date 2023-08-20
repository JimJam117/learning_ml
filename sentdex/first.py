import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = quandl.get('WIKI/GOOGL', api_key= "Ht3wGaVKYzgw6iAbhzRW")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forcast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forcast_col].shift(-forcast_out)

#print(df.head)

# df.hist(bins=50, figsize=(20,15))
# df.plot(kind="scatter", x="Adj. Close", y="Adj. Volume", alpha=0.1)
# plt.show()

# X is features (everything except label)
X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

X = preprocessing.scale(X)

# remove all x where we dont have forcast
X = X[:-forcast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = svm.SVR()
clf.fit(X_train, y_train)
accuaracy = clf.score(X_test, y_test)

print(accuaracy)