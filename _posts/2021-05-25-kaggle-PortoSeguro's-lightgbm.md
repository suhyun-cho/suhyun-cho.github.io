---
layout: single
title : "Kaggle - PortoSeguro's safe driver prediction_v2 (Simple-classification-challenge-with-lightgbm)"
author_profile: true
read_time: false
comments: true
categories:
- Kaggle
---

## A Simple Classification Challenge With LightGBM 

https://medium.com/@invest_gs/a-simple-classification-challenge-with-lightgbm-kaggle-competition-e12467cfec96

#### a) target=1 데이터를 Over-Sampling 했을때 (Over-fitting될 수 있음)

* Private Score : 0.27944
* Public Score : 0.27624

#### b) target=0 데이터를 Under-Sampling 했을때 (정보손실 될 수 있음)

* Private Score : 0.27450
* Public Score : 0.27004


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
```

### 1. Load data sets


```python
train_df = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')
val_df = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')
submission_df=pd.read_csv('../input/porto-seguro-safe-driver-prediction/sample_submission.csv')
```


```python
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in

# import numpy as np
# import pandas as pd
# import lightgbm
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelBinarizer


# #
# # Prepare the data
# #

# train = pd.read_csv('../input/train.csv')

# # get the labels
# y = train.target.values
# train.drop(['id', 'target'], inplace=True, axis=1)

# x = train.values

# #
# # Create training and validation sets
# #
# x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# #
# # Create the LightGBM data containers
# #
# categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]
# train_data = lightgbm.Dataset(x, label=y, categorical_feature=categorical_features)
# test_data = lightgbm.Dataset(x_test, label=y_test)


# #
# # Train the model
# #

# parameters = {
#     'application': 'binary',
#     'objective': 'binary',
#     'metric': 'auc',
#     'is_unbalance': 'true',
#     'boosting': 'gbdt',
#     'num_leaves': 31,
#     'feature_fraction': 0.5,
#     'bagging_fraction': 0.5,
#     'bagging_freq': 20,
#     'learning_rate': 0.05,
#     'verbose': 0
# }

# model = lightgbm.train(parameters,
#                        train_data,
#                        valid_sets=test_data,
#                        num_boost_round=5000,
#                        early_stopping_rounds=100)
# #
# # Create a submission
# #

# submission = pd.read_csv('../input/test.csv')
# ids = submission['id'].values
# submission.drop('id', inplace=True, axis=1)


# x = submission.values
# y = model.predict(x)

# output = pd.DataFrame({'id': ids, 'target': y})
# output.to_csv("submission.csv", index=False)
```


```python
# train, test데이터를 merge해서 한꺼번에 살펴본다
combined_df=pd.concat([train_df, val_df])
print(combined_df.describe())
```

### 2. Missing values



```python
# target컬럼외에 null값이 있는 컬럼은 없음.
print(combined_df.columns[combined_df.isnull().any()])

# 결측치 비율 확인 -> 없음
print(combined_df[[]].isnull().sum()/len(combined_df)*100)
```

### 3. Data types
* bin, cat으로 끝나는 컬럼은 categorical 타입.


```python
# data type확인 : int타입이 48개, float타입이 11개
from collections import Counter
print(Counter([combined_df[col].dtype for col in combined_df.columns.values.tolist()]).items())

# ID 컬럼을 index로
combined_df.set_index('id',inplace=True)

# categorical, binary값들은 더미변수 만들어주기 (for LGBM train remove it)
combined_df=pd.get_dummies(combined_df, columns=[col for col in combined_df if col.endswith('bin') or col.endswith('cat')])

# 더미변수 만든 후 data type체크
print(Counter([combined_df[col].dtype for col in combined_df.columns.values.tolist()]).items())
```

### 4. Split data
* train, validation set으로 데이터 분할


```python
# combined_df를 다시 train_df, val_df로 나눈다
train_df=combined_df.loc[combined_df['target'].isin([1,0])]
val_df=combined_df[combined_df.index.isin(submission_df['id'])]
val_df=val_df.drop(['target'],axis=1)

# X_train_df, y_train_df를 만든다
X_train_df=train_df.drop('target',axis=1)
y_train_df=train_df['target']
```

### 5. Scaling the data
*  RobustScaler, StandardScaler, MinMaxScaler 를 모두 적용해보고 score가 좋은것을 채택할것이다.
    * 결과적으로 StandardScaler가 데이터에 가장 적합했다.


```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# 각각의 scaler정의
#scaler = RobustScaler()
#scaler = MinMaxScaler()
scaler=StandardScaler()

# Scale the X_train set
X_train_scaled=scaler.fit_transform(X_train_df.values)
X_train_df=pd.DataFrame(X_train_scaled, index=X_train_df.index, columns=X_train_df.columns)

# Scale the X_test set
val_scaled=scaler.transform(val_df.values)
val_df=pd.DataFrame(val_scaled, index=val_df.index, columns=val_df.columns)
```

### 6. Train test split
* target값이 있는 train 데이터를 train,test로 나눠서 검증까지해보려고함
    * 이 노트북에서 val_df는 target값이 없는 실제 우리가 예측할 x값들이 저장되어있고
    * X_train은 모델 학습할 용도, X_test는 모델 검증용도임


```python
# 20%를 검증용 데이터셋으로 구성해서 train, test분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=20)
```

### 7. Class Imbalance
* 클래스 불균형이 심한 데이터이므로 under sampling이나 over sampling을 해야하는데
* 정보 손실을 막기위해 target=1인 값을 over sampling할것이다

> 검증용 데이터(X_test, y_test)말고 훈련용 데이터(X_train, y_train)만 가지고 target=1인데이터 over sampling한것


```python
# Upsample minority class

# Concatenate our training data back together
X=pd.concat([X_train, y_train],axis=1)

# Separate minority and majority classes
not_target=X[X.target==0]
yes_target=X[X.target==1]
print(not_target.shape)
print(yes_target.shape)
```

#### a) target=1인 데이터를 Over Sampling
* target=0 개수 : 458,853
* target=1 개수 : 17316 -> 458,853


```python
# target=1인 데이터를 target=0인 데이터만큼 oversampling한것 (=yes_target_up)
from sklearn.utils import resample
yes_target_up = resample(yes_target,
                                replace = True, # sample without replacement
                                n_samples = len(not_target), # match majority n
                                random_state = 27) # reproducible results

# Combine minority and downsampled majority
upsampled=pd.concat([yes_target_up, not_target])

# Checking counts (target=0과 target=1의 데이터개수가 동일해짐)
print(upsampled.target.value_counts())

# Create training set again
X_train=upsampled.drop('target',axis=1)
y_train=upsampled.target

print(len(X_train))
```

#### b) target=0 인 데이터를 Under Sampling
target=0인 데이터의 30%만 resample <br>

* target=0 개수 : 458,853 -> 137,655
* target=1 개수 : 17316 


```python
# from sklearn.utils import resample
# target_ratio=0.3  # target=0 데이터의 30% => 약 137,655 개
# not_target_down = resample(not_target,
#                                 replace = True, # sample without replacement
#                                 n_samples = int(not_target.shape[0]*target_ratio), 
#                                 random_state = 27) # reproducible results

# # Combine minority and downsampled majority
# upsampled=pd.concat([yes_target, not_target_down])

# # Checking counts (target=0과 target=1의 데이터개수가 동일해짐)
# print(upsampled.target.value_counts())

# # Create training set again
# X_train=upsampled.drop('target',axis=1)
# y_train=upsampled.target

# print(len(X_train))
```


```python
# 여전히 검증용set에서는 target=1과 0의 비중은 다른 상태임!
## 검증용set은 oversampling안하고 훈련용set만 oversampling해서 훈련할것
pd.concat([X_test,y_test],axis=1).target.value_counts()
```

### 8. Model and submission
* LightGBM을 모델링을 할건데, 5000round 훈련시킬것임
* overfitting을 방지하기위해 100round에서 stoppint


```python
# LIGHT GBM
import lightgbm as lgbm

# Indicate the categorical features for the LGBM classifier
## 컬럼에 'cat'가 들어있는 컬럼의 위치 추출
categorical_features=[c for c, col in enumerate(X_train.columns) if 'cat' in col]

# Get the train and test data for the training sequence
train_data=lgbm.Dataset(X_train,label=y_train, categorical_feature=categorical_features)
test_data=lgbm.Dataset(X_test,label=y_test)

# Set the parameters for training
parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    #'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

# Train the classifier
classifier=lgbm.train(parameters, 
                      train_data, 
                      valid_sets=test_data,
                     num_boost_round=5000,
                     early_stopping_rounds=100)

# Make predictions
predictions=classifier.predict(val_df.values)

# Create submission file
my_pred_lgbm=pd.DataFrame({'id':val_df.index,'target':predictions})

# Create CSV file
my_pred_lgbm.to_csv('pred_lgbm_undersampling.csv',index=False)
```


```python

```
