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
from sklearn.tree import DecisionTreeClassifier
from pandas.plotting import scatter_matrix

header=['sepal-length', 'sepcal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('./data/2.iris.csv', names=header)
array = data.values
x = array[:,0:4]
y = array[:,4]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_x = scaler.fit_transform(x)
(x_train, x_text,
 y_train, y_test) = train_test_split(rescaled_x,y,test_size=0.2)

model = DecisionTreeClassifier()
fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, rescaled_x, y, cv=fold, scoring='accuracy')
print(acc)



