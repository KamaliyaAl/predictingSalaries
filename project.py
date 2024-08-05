import os

import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')
# finding the variables with the biggest correlation
# corr = data.drop('salary', axis = 1).corr()
# labels = corr.values
# listOfBigCor = []
# for i in range(labels.shape[0]):
#     for j in range(labels.shape[1]):
#         if i != j and labels[j, i] > 0.2:
#             if corr.columns[i] not in listOfBigCor:
#                 listOfBigCor.append(corr.columns[i])
#             if corr.columns[j] not in listOfBigCor:
#                 listOfBigCor.append(corr.columns[j])
# count_variables = len(listOfBigCor)
# new = itertools.combinations(listOfBigCor, 2)
# for el in new: listOfBigCor.append(list(el))
X = pd.DataFrame(data).drop('salary', axis=1)[['rating', 'age']]
y = pd.Series(data['salary'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred1 = model.predict(X_test)
y_pred2 = y_pred1
y_pred1[y_pred1<0] = 0
y_pred2[y_pred2<0] = y_train.median()
mape1 = mape(y_test, y_pred1)
mape2 = mape(y_test, y_pred2)
# printing the smallest mape
print('{:.5f}'.format(min(mape1, mape2)))
# finding the best mape
# for i in range(len(listOfBigCor)):
#     if i<count_variables:
#         X_train_curr = X_train.drop([listOfBigCor[i]], axis=1)
#         X_test_curr = X_test.drop([listOfBigCor[i]], axis=1)
#     else:
#         X_train_curr = X_train.drop([listOfBigCor[i][0], listOfBigCor[i][1]], axis=1)
#         X_test_curr = X_test.drop([listOfBigCor[i][0], listOfBigCor[i][1]], axis=1)
#     model.fit(X_train_curr, y_train)
#     y_pred = model.predict(X_test_curr)
#     mapes.append(mape(y_test, y_pred))


