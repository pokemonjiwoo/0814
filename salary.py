import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/1.salary.csv')
print(data)

array = data.values
x = array[:,0]
y = array[:,1]
plt.clf()
fig, ax = plt.subplots()
plt.scatter(x,y, label='random', color='gold', marker='*',
            s=30, alpha=0.5)
plt.show()
x=x.reshape(-1,1)

(x_train, x_text, y_train, y_test) = train_test_split(x,y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_text)
print(y_pred)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue',
            marker='o')
plt.plot(range(len(y_pred)), y_pred, color='r', marker='x')
plt.show()

mae=mean_absolute_error(y_pred, y_test)
print(mae)



