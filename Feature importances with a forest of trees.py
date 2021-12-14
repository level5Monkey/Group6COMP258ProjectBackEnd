# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:25:32 2021

@author: Jiaxing Li
"""

import os
import numpy as np
import pandas as pd
from typing import Callable
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
from sklearn import preprocessing
from keras import callbacks
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import time


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

path = r"E:/workspace/neural network/group project/HYPE Dataset for Group Project.xlsx"
data = pd.read_excel(path, sheet_name='HYPE-Retention')

data.isnull().sum()

data.head(3)
data.shape
data.dtypes
data.columns
data.info()
data.keys()
data.isnull().sum()
data['INTAKE COLLEGE EXPERIENCE'].unique()

for column in data.columns:
    print(column)
    print(data[column].unique())

# drop columns which have too many missing value
data = data.drop('HS AVERAGE GRADE', axis = 1)
data = data.drop('HS AVERAGE MARKS', axis = 1)
data = data.drop('FIRST GENERATION IND', axis = 1)
data = data.drop('APPL EDUC INST TYPE NAME', axis = 1)

# create location column with the first letter of postal code and remove other location columns
data['LOCATION'] = data['MAILING POSTAL CODE'].astype(str).str[:1]
data['LOCATION'] = data['LOCATION'].replace('n','N')
data = data.drop('MAILING CITY NAME', axis = 1)
data = data.drop('MAILING POSTAL CODE GROUP 3', axis = 1)
data = data.drop('MAILING POSTAL CODE', axis = 1)
data = data.drop('MAILING PROVINCE NAME', axis = 1)
data = data.drop('MAILING COUNTRY NAME', axis = 1)


# drop columns which have too many unique values
data = data.drop('PROGRAM LONG NAME', axis = 1)
data = data.drop('FUTURE TERM ENROL', axis = 1)

# create length of program with expected grad term minus admin term 
data['PROGRAM LENGTH'] = (data['EXPECTED GRAD TERM CODE'].astype(str).str[:4].astype(int) * 12 + data['EXPECTED GRAD TERM CODE'].astype(str).str[4:6].astype(int)) - (data['INTAKE TERM CODE'].astype(str).str[:4].astype(int) * 12 + data['INTAKE TERM CODE'].astype(str).str[4:6].astype(int))
data = data.drop('INTAKE TERM CODE', axis = 1)
data = data.drop('ADMIT TERM CODE', axis = 1)
data = data.drop('EXPECTED GRAD TERM CODE', axis = 1)

# columns to drop because they have little relation with the target
data = data.drop('ID 2', axis = 1)
data = data.drop('RECORD COUNT', axis = 1)
data = data.drop('PRIMARY PROGRAM CODE', axis = 1)

# columns to drop because they have the same value or too many unique value
data = data.drop('STUDENT TYPE NAME', axis = 1)
data = data.drop('STUDENT TYPE GROUP NAME', axis = 1)
data = data.drop('CURRENT STAY STATUS', axis = 1)

# adjust ENGLISH TEST SCORE
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(170,'Good')
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(171,'Good')
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(160,'Average')
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(161,'Average')
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(130,'Below Average')
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(131,'Below Average')
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(140,'Below Average')
data['ENGLISH TEST SCORE'] = data['ENGLISH TEST SCORE'].replace(141,'Below Average')
data['APPL FIRST LANGUAGE DESC'] = data['APPL FIRST LANGUAGE DESC'].replace('Unknown','Other')

# PREV EDU CRED LEVEL NAME is duplicate with APPLICANT CATEGORY NAME
data = data.drop('PREV EDU CRED LEVEL NAME', axis = 1)
data['APPLICANT CATEGORY NAME'] = data['APPLICANT CATEGORY NAME'].replace('High School, Domestic','High School')
data['APPLICANT CATEGORY NAME'] = data['APPLICANT CATEGORY NAME'].replace('Mature: Domestic  With Post Secondary','Post Secondary')
data['APPLICANT CATEGORY NAME'] = data['APPLICANT CATEGORY NAME'].replace('Mature: Domestic 19 or older No Academic History','No Academic History')
data['APPLICANT CATEGORY NAME'] = data['APPLICANT CATEGORY NAME'].replace('BScN, High School Domestic','High School')
data['APPLICANT CATEGORY NAME'] = data['APPLICANT CATEGORY NAME'].replace('International Student, with Post Secondary','Post Secondary')


# create new columns 'rest semesters' = TOTAL PROGRAM SEMESTERS - PROGRAM SEMESTERS and drop the later two columns
data['REST SEMESTERS'] = data['TOTAL PROGRAM SEMESTERS'] - data['PROGRAM SEMESTERS']
data = data.drop('TOTAL PROGRAM SEMESTERS', axis = 1)
data = data.drop('PROGRAM SEMESTERS', axis = 1)

# fill missing value with most frequent value
data = data.apply(lambda x:x.fillna(x.mode().iloc[0]))

# remove 'SUCCESS LEVEL' = 'In Progress' data 
data = data.drop(data[data['SUCCESS LEVEL'] == 'In Progress'].index)
data = data.replace('Successful', 1)
data = data.replace('Unsuccessful', 0)

y = pd.DataFrame(data=data['SUCCESS LEVEL'], columns=['SUCCESS LEVEL'])
data = data.drop('SUCCESS LEVEL', axis = 1)

data['FIRST YEAR PERSISTENCE COUNT'] = data['FIRST YEAR PERSISTENCE COUNT'].astype(int)
data['PROGRAM LENGTH'] = data['PROGRAM LENGTH'].astype(int)
data['REST SEMESTERS'] = data['REST SEMESTERS'].astype(int)

# one hot encode categorical features
CATEGORICAL_FEATURE_KEYS = ['INTAKE COLLEGE EXPERIENCE', 'SCHOOL CODE', 'STUDENT LEVEL NAME',
       'TIME STATUS NAME', 'RESIDENCY STATUS NAME', 'FUNDING SOURCE NAME',
       'GENDER', 'DISABILITY IND', 'ACADEMIC PERFORMANCE', 'FIRST YEAR PERSISTENCE COUNT',
       'ENGLISH TEST SCORE','AGE GROUP LONG NAME', 'APPL FIRST LANGUAGE DESC',
       'APPLICANT CATEGORY NAME', 'APPLICANT TARGET SEGMENT NAME', 'LOCATION'
]

data_cat = data[CATEGORICAL_FEATURE_KEYS]
data_num = data.drop(CATEGORICAL_FEATURE_KEYS, axis=1)

enc_features = OneHotEncoder()
data_cat = enc_features.fit_transform(data_cat).toarray()

with open("encoder.pkl", "wb") as f: 
    pickle.dump(enc_features, f)

# scale numerical features
"""
normalizer = preprocessing.Normalizer().fit(data_num)

cols_to_norm = data_num.columns
data_num[cols_to_norm] = data_num[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

data_num = normalizer.transform(data_num)
data_num = pd.DataFrame(data = data_num, columns = normalizer.feature_names_in_)
"""
scaler = MinMaxScaler().fit(data_num)

data_num = scaler.transform(data_num)
data_num = pd.DataFrame(data = data_num, columns = scaler.feature_names_in_)

with open("scaler.pkl", "wb") as f: 
    pickle.dump(scaler, f)

#X = pd.DataFrame(data = data_cat, columns = enc_features.get_feature_names_out())
X = pd.DataFrame(data = data_cat, columns = enc_features.get_feature_names_out()).join(data_num)

"""
# convert y
enc_targets = OneHotEncoder()
y = enc_targets.fit_transform(y).toarray()
print(enc_targets.get_feature_names_out())
y = pd.DataFrame(data = y, columns = enc_targets.get_feature_names_out())
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


plt.rcParams.update({'figure.figsize': (120.0, 80.0)})
plt.rcParams.update({'font.size': 70})

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

sorted_idx = rf.feature_importances_.argsort()

plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")


from sklearn.inspection import permutation_importance

forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

sorted_idx = forest.feature_importances_.argsort()

start_time = time.time()
result = permutation_importance(
    forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=X.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()