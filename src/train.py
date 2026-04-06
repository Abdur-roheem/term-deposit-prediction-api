import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import xgboost as xgb

#Parameters
output_file = 'bank-model.bin'


#Data Preparation

# Load data
df = pd.read_csv("../data/bank-full.csv", sep=";")

# Preprocess data
categorical = ['job', 'marital', 'education', 'default',  'housing',
               'loan', 'contact', 'month', 'poutcome']
numerical = ['age', 'balance', 'day', 'duration', 
             'campaign', 'pdays','previous']


df.y = (df.y == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=.2, random_state=24)
df_train, df_val = train_test_split(df_full_train, test_size=.25, random_state=24)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.y.values
y_val = df_val.y.values
y_test = df_test.y.values

del df_train['y']
del df_val['y']
del df_test['y']


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.feature_names_)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.feature_names_)

# Train model
xgb_params = { 
'eta': 0.3, 
'max_depth': 6, 
'min_child_weight': 1, 
'objective': 'binary:logistic',
'eval_metric' : 'auc',
'nthread': 8, 
'seed': 1, 
'verbosity' : 1
} 

print("Fitting model")
model = xgb.train(xgb_params, dtrain, num_boost_round=15)

# Evaluate model
y_pred = model.predict(dval)
score = roc_auc_score(y_val, y_pred)
print(f"Auc score is {score}")

# save the model

with open(output_file, 'wb') as f_out:      
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
