import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn import preprocessing

loan_data1  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )

loan_data1.rename({"Unnamed: 0":"u"}, axis="columns", inplace=True)
loan_data1.drop(["u"], axis=1, inplace=True)

print(loan_data1.head())

loan_data1.info()

X1 = loan_data1.drop(['Loan_Status','Loan_ID'],axis = 1)
y1 = loan_data1['Loan_Status']

X_train11, X_test11, y_train11, y_test11 = train_test_split(X1, y1)

X_train11.isnull().sum()

num_cols11 = X_train11.select_dtypes(include=np.number).columns
cat_cols11 = X_train11.select_dtypes(include = 'object').columns
X_train11[num_cols11] = X_train11[num_cols11].fillna(X_train11[num_cols11].mean()) 
X_train11[cat_cols11] = X_train11[cat_cols11].fillna(X_train11[cat_cols11].mode().iloc[0])

X_train11.isnull().sum() / len(X_train11) * 100

y_train11 = y_train11.astype('category')
X_train11['Gender'] = X_train11['Gender'].astype('category')
X_train11['Married'] = X_train11['Married'].astype('category')
X_train11['Dependents'] = X_train11['Dependents'].astype('category')
X_train11['Education'] = X_train11['Education'].astype('category')
X_train11['Self_Employed'] = X_train11['Self_Employed'].astype('category')
X_train11['Property_Area'] = X_train11['Property_Area'].astype('category')

X_train11.dtypes

X_test11['Gender'] = X_test11['Gender'].astype('category').cat.set_categories(X_train11['Gender'].cat.categories)

X_test11['Married'] = X_test11['Married'].astype('category').cat.set_categories(X_train11['Married'].cat.categories)

X_test11['Dependents'] = X_test11['Dependents'].astype('category').cat.set_categories(X_train11['Dependents'].cat.categories)

X_test11['Education'] = X_test11['Education'].astype('category').cat.set_categories(X_train11['Education'].cat.categories)

X_test11['Self_Employed'] = X_test11['Self_Employed'].astype('category').cat.set_categories(X_train11['Self_Employed'].cat.categories)

X_test11['Property_Area'] = X_test11['Property_Area'].astype('category').cat.set_categories(X_train11['Property_Area'].cat.categories)

test_dataset = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')

test_dataset.describe()

test_dataset.head()

test_dataset.info()

test_dataset = test_dataset.drop('Loan_ID',axis = 1)
num_cols33 = test_dataset.select_dtypes(include=np.number).columns
cat_cols33 = test_dataset.select_dtypes(include = 'object').columns
test_dataset[num_cols33] = test_dataset[num_cols33].fillna(test_dataset[num_cols33].mean())
test_dataset[cat_cols33] = test_dataset[cat_cols33].fillna(test_dataset[cat_cols33].mode().iloc[0])
test_dataset.info()

test_dataset['Gender'] = test_dataset['Gender'].astype('category').cat.set_categories(X_train11['Gender'].cat.categories)
test_dataset['Married'] = test_dataset['Married'].astype('category').cat.set_categories(X_train11['Married'].cat.categories)
test_dataset['Dependents'] = test_dataset['Dependents'].astype('category').cat.set_categories(X_train11['Dependents'].cat.categories)
test_dataset['Education'] = test_dataset['Education'].astype('category').cat.set_categories(X_train11['Education'].cat.categories)
test_dataset['Self_Employed'] = test_dataset['Self_Employed'].astype('category').cat.set_categories(X_train11['Self_Employed'].cat.categories)
test_dataset['Property_Area'] = test_dataset['Property_Area'].astype('category').cat.set_categories(X_train11['Property_Area'].cat.categories)

missing_levels_cols41= list(set(X_train11.columns) - set(test_dataset.columns))
for c in missing_levels_cols41:
    test_dataset[c]=0


# Select only those columns which are there in training data
test_dataset=test_dataset[X_train11.columns]
print(test_dataset)

xgclf = xgb.XGBClassifier(
    tree_method="hist", enable_categorical=True, max_cat_to_onehot=5)
parameters = {
    'n_estimators': [80, 90, 100, 125, 150],
    'max_depth': [2, 3, 4, 5, 6, 8, 16, None],
    'learning_rate': [0.01, 0.03, 0.1, 0.3, 0.5]
}
xgclf = GridSearchCV(xgclf, parameters, cv=5)
xgclf.fit(X_train11, y_train11)

print(xgclf)

xgclf.best_score_

print(f1_score(y_train11,xgclf.predict(X_train11)))

print(xgclf.predict(X_test11))

print(f1_score(y_test11,xgclf.predict(X_test11)))

X_train11.shape

final_tst = pd.DataFrame(data=test_dataset)
final_tst.columns= test_dataset.columns
print(final_tst.head())
print(final_tst.shape)

final_tst

final_tst.info()

print(final_tst.columns)

predictions1 = xgclf.predict(final_tst)

print(predictions1)

#save the model 
filename = 'xgboost_loanpred_model.pkl'
pickle.dump(xgclf, open(filename, 'wb'))