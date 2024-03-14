import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("Forest_fire.csv")
df = np.array(df)

X = df[1:, 1:-1]
y = df[1:, -1]
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)
logi = LogisticRegression()
logi.fit(X_train, y_train)

ip = [int(x) for x in "45 32 60".split(' ')]
final = [np.array(ip)]

res = logi.predict_proba(final)

pickle.dump(logi, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))