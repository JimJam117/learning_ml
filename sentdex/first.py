import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL', api_key= "Ht3wGaVKYzgw6iAbhzRW")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forcast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.001*len(df)))

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
X_lately = X[-forcast_out:]
X = X[:-forcast_out]


df.dropna(inplace=True)
y = np.array(df['label'])

print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

clf = LinearRegression(n_jobs=10)
clf.fit(X_train, y_train)
accuaracy = clf.score(X_test, y_test)

#print(accuaracy)

forcast_set = clf.predict(X_lately)

print(forcast_set, accuaracy, forcast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day_in_secs = 86400
next_unix = last_unix + one_day_in_secs

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day_in_secs
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()