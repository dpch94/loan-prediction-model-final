
#import mkl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split  # To split the dataset into train and test set

from sklearn import preprocessing
from sklearn.metrics import f1_score

loan_data1  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )

loan_data1.rename({"Unnamed: 0":"u"}, axis="columns", inplace=True)
loan_data1.drop(["u"], axis=1, inplace=True)

X1 = loan_data1.drop(['Loan_Status','Loan_ID'],axis = 1)
y1 = loan_data1['Loan_Status']

#Splitting data into train and test
X_train11, X_test11, y_train11, y_test11 = train_test_split(X1, y1, test_size=0.2, random_state=1)

X_train11.head()

X_train11.isnull().sum()

num_cols11 = X_train11.select_dtypes(include=np.number).columns
cat_cols11 = X_train11.select_dtypes(include = 'object').columns
X_train11[num_cols11] = X_train11[num_cols11].fillna(X_train11[num_cols11].mean()) 
X_train11[cat_cols11] = X_train11[cat_cols11].fillna(X_train11[cat_cols11].mode().iloc[0])

X_train11.isnull().sum() / len(X_train11) * 100

X_train11 = pd.get_dummies(X_train11, columns=cat_cols11)
print(X_train11.shape)
print(X_train11.head())

#Missing values treatment
num_cols21 = X_test11.select_dtypes(include=np.number).columns
cat_cols21 = X_test11.select_dtypes(include = 'object').columns
X_test11[num_cols21] = X_test11[num_cols21].fillna(X_test11[num_cols21].mean()) 
X_test11[cat_cols21] = X_test11[cat_cols21].fillna(X_test11[cat_cols21].mode().iloc[0])

X_test11 = pd.get_dummies(X_test11, columns=cat_cols21)
print(X_test11.shape)
print(X_test11.head())

missing_levels_cols22= list(set(X_train11.columns) - set(X_test11.columns))
print(len(missing_levels_cols22))
for c in missing_levels_cols22:
    X_test11[c]=0

# Select only those columns which are there in training data
X_test11=X_test11[X_train11.columns]

final_ts2 = pd.DataFrame(data=X_test11)
final_ts2.columns= X_test11.columns
print(final_ts2.head())
print(final_ts2.shape)

final_ts2

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train11, y_train11)

def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

from sklearn.model_selection import GridSearchCV
gbc = GradientBoostingClassifier()
parameters = {
    'n_estimators': [80, 90, 100, 125, 150],
    'max_depth': [2,3,4,5,8,16,None],
    'learning_rate': [0.03, 0.1, 0.3, 0.5]
}
cv2 = GridSearchCV(gbc, parameters, cv=5)
cv2.fit(X_train11, y_train11)

print_results(cv2)

cv2.best_score_

print(f1_score(y_train11,cv2.predict(X_train11)))

print(f1_score(y_test11,cv2.predict(X_test11)))

X_train11.shape





"""Predictions on test dataset"""



test_datasetfs = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')

test_datasetfs.describe()

test_datasetfs.info()

test_datasetfs.head()

num_cols33 = test_datasetfs.select_dtypes(include=np.number).columns
cat_cols33 = test_datasetfs.select_dtypes(include = 'object').columns
test_datasetfs[num_cols33] = test_datasetfs[num_cols33].fillna(test_datasetfs[num_cols33].mean()) 
test_datasetfs[cat_cols33] = test_datasetfs[cat_cols33].fillna(test_datasetfs[cat_cols33].mode().iloc[0])

test_datasetfs.info()

test_datasetfs = pd.get_dummies(test_datasetfs, columns=cat_cols33)

missing_levels_cols41= list(set(X_train11.columns) - set(test_datasetfs.columns))
for c in missing_levels_cols41:
    test_datasetfs[c]=0

# Select only those columns which are there in training data
test_datasetfs=test_datasetfs[X_train11.columns]

from sklearn import preprocessing

final_tst41 = pd.DataFrame(data=test_datasetfs)
final_tst41.columns= test_datasetfs.columns
print(final_tst41.head())
print(final_tst41.shape)

final_tst41

final_tst41.info()

print(final_tst41.columns)

predictions1 = cv2.predict(final_tst41)

print(predictions1)

#save the model 
filename = 'loan_predfinal_model.pkl'
pickle.dump(cv2, open(filename, 'wb'))

