---
layout: single
title : "Kaggle필사 - HousePrices 회귀모델을 이용한 예측"
author_profile: true
read_time: false
comments: true
categories:
- Kaggle
---

# House Prices
## 3. A Study on Regression Applied to the Ames Dataset
[캐글]: https://www.kaggle.com/c/house-prices-advanced-regression-techniques <br>
[참고 커널모음] :https://subinium.github.io/kaggle-tutorial/house-prices <br>

#### Python Tutorials
아래 튜토리얼 하나씩 필사할 예정

> * 1. Comprehensive Data Exploration with Python (필사 시작일 : 2020-09-01)
https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

> * 2. House Prices EDA (필사 시작일 : 2020-09-04) (마무리한 날짜: 2020-10-25)
https://www.kaggle.com/dgawlik/house-prices-eda

> * 3. A Study on Regression Applied to the Ames Dataset (필사 시작일 :2020-11-15)
https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

> * 4. Regularized Linear Models
https://www.kaggle.com/apapiu/regularized-linear-models




--------------------------------------------------------------------------------
# [정리]

### Data-Preprocessing
* 결측치 처리방식
    * 데이터의 형태별로 0으로 채우거나, 실제 없는 값일경우 No, None등으로 채움<br>
    <br>
* 카테고리변수 전처리
    * MoSold 변수는 month를 나타내는데, 숫자 1,2,3,, 으로 되어있어서 이를 Jan, Feb,,, 이렇게 문자로 바꿔줌<br>
    <br>
* 순서를 나타내는 카테고리 전처리
    * GarageQual 퀄리티가 나쁨, 중간, 좋음 이런식의 순서가 있는 변수일 경우 나쁨->1, 중간->2,좋음->3,,이런식으로 넘버링 해준다.<br>
    <br>

* 새로운 컬럼 만들기 
    >  * 존재하는 feature들을 단순화
        * ex) 퀄리티가 1부터 10까지 다양하게 있으면, 이걸 범주화해서 1~3은 1로, 4~6은 2로, 7~10은 3으로.
    * 존재하는 Feature들 조합
        * ex) Qual와 Cond의 곱으로 %Grade라는 컬럼을 새로 만듬.
        * ex) 주차장 면적과 주차장Quality를 곱해서 주차장 score 변수를 만듬.
        * ex) 1층넓이 변수와 2층넓이변수가 있으면 Total넓이라는 변수로 두 변수를 더해서 만듬.
    * 상위10개 Feature로 다항식
        * 상위 10개 feature를 가지고 2제곱, 3제곱, 제곱근항을 각각 세개씩 만듬.

* 수치형 변수와 범주형 변수를 따로 저장.
    * select_dtypes() 함수
        * 범주형 데이터타입인 컬럼만 따로 빼올수있음. 반대로, 범주형이아닌 변수도 따로 뺄 수 있음.

* 수치형 변수에있는 결측치를 median값으로 채운다
* 비대칭(skew) 수치형변수 처리
    * 아웃라이어의 영향을 줄이기위해서 비대칭(skew)한 수치형 변수들을 로그변환한다.
* one-hot encoding으로 범주형에 대해서 더미변수를 만든다.


--------------------------------------------------------------------------------
### Modeling
> * Scaling
    * train_test_split으로 X_train,X_test,y_train,y_test로 데이터 나눈다음 X_train, X_test에만 스케일링 적용.
    * y는 이미 사전에 로그변환 취한 상태. <br>
<br>
* Cross-validation
    * fold=10으로해서 교차검증을 하고, RMSE로 평가<br>


**(1) 회귀모델 (without Regularization)**
* RMSE on Test set : 0.395779797728

**(2) Ridge(L2 penalty)**
* Ridge RMSE on Test set : 0.116427213778
    * 계수(coefficeint)로 변수의 중요도 판단가능할듯..(계수가 큰 상위10개, 작은것 10개)

**(3) Lasso(L1 penalty)**
* Lasso RMSE on Test set : 0.115832132218

**(4) ElasticNet regularization(L1 and L2 penalty)**
* ElasticNet RMSE on Test set : 0.115832132218



-------------------------------------------------
이 커널에서는 Regularization알고리즘을 포함한 선형회귀에 대해 다룬다.  <br>
RF,  xgboost, ensembling etc. 등을 사용하지 않고 선형회귀만으로 0.121 이라는 스코어획득함.


```python
from IPython.core.display import display, HTML
display(HTML("<style> .container{width:90% !important;}</style>"))
```


<style> .container{width:90% !important;}</style>



```python
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%matplotlib inline
#njobs = 4
```


```python
# Get data
train = pd.read_csv("./data/train.csv")
print("train : " + str(train.shape))
```

    train : (1460, 81)



```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.000</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.000</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.000</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.000</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.000</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
# 중복확인.
idsUnique=len(set(train.Id))  # 1460
idsTotal=train.shape[0]
idsDupli=idsTotal-idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

# Id 컬럼 제거
train.drop("Id",axis=1, inplace=True)
```

    There are 0 duplicate IDs for 1460 total entries


------------------------------------------------------------
### Preprocessing


```python
# outlier체크
plt.scatter(train.GrLivArea, train.SalePrice, c="blue", marker="s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

train=train[train.GrLivArea<4000]
```


![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_10_0.png)


* 오른쪽 하단에 면적이 큰데, 매우 저렴하게 팔리는 2개의 outlier가 보인다. 데이터셋의 author은 4000square feet이상의 집들은 제거할것을 권했다. 


```python
train.SalePrice=np.log1p(train.SalePrice)
y=train.SalePrice
```

* 로그를 취하는것에 의미는, 비싼 집과 저렴한 집을 예측하는것에 대한 에러가 결과에 동일하게 영향을 미치게 한다는 것을 의미한다.

**(결측치 처리)**


```python
# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"
train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)
```


```python
# Some numerical features are actually really categories
## 카테고리를 의미하는것은 문자로 바꿔줌.
train = train.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
```


```python
# Encode some categorical features as ordered numbers when there is information in the order
## 순서를 나타내는 카테고리는 0,1,2,3,,이런식으로 넘버링 해준다.
train = train.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )
```

**3가지 방법으로 새로운 feature를 만들것이다**
1. Simplifications of existing features
    * 존재하는 feature들을 단순화
        * ex) 퀄리티가 1부터 10까지 다양하게 있으면, 이걸 범주화해서 1~3은 1로, 4~6은 2로, 7~10은 3으로.
2. Combinations of existing features
    * 존재하는 Feature들 조합
        * ex) Qual와  Cond의 곱으로  %Grade라는 컬럼을 새로 만듬.
        * ex) 주차장 면적과 주차장Quality를 곱해서 주차장 score 변수를 만듬.
3. Polynomials on the top 10 existing features
    * 상위10개 Feature로 다항식 


```python
# Create new features
# 1* Simplifications of existing features
train["SimplOverallQual"] = train.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
train["SimplOverallCond"] = train.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })
train["SimplPoolQC"] = train.PoolQC.replace({1 : 1, 2 : 1, # average
                                             3 : 2, 4 : 2 # good
                                            })
train["SimplGarageCond"] = train.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
train["SimplGarageQual"] = train.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
train["SimplFireplaceQu"] = train.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
train["SimplFireplaceQu"] = train.FireplaceQu.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
train["SimplFunctional"] = train.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })
train["SimplKitchenQual"] = train.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })
train["SimplHeatingQC"] = train.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
train["SimplBsmtFinType1"] = train.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
train["SimplBsmtFinType2"] = train.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })
train["SimplBsmtCond"] = train.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
train["SimplBsmtQual"] = train.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
train["SimplExterCond"] = train.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
train["SimplExterQual"] = train.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })

# 2* Combinations of existing features
# Overall quality of the house
train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]
# Overall quality of the garage
train["GarageGrade"] = train["GarageQual"] * train["GarageCond"]
# Overall quality of the exterior
train["ExterGrade"] = train["ExterQual"] * train["ExterCond"]
# Overall kitchen score
train["KitchenScore"] = train["KitchenAbvGr"] * train["KitchenQual"]
# Overall fireplace score
train["FireplaceScore"] = train["Fireplaces"] * train["FireplaceQu"]
# Overall garage score
train["GarageScore"] = train["GarageArea"] * train["GarageQual"]
# Overall pool score
train["PoolScore"] = train["PoolArea"] * train["PoolQC"]
# Simplified overall quality of the house
train["SimplOverallGrade"] = train["SimplOverallQual"] * train["SimplOverallCond"]
# Simplified overall quality of the exterior
train["SimplExterGrade"] = train["SimplExterQual"] * train["SimplExterCond"]
# Simplified overall pool score
train["SimplPoolScore"] = train["PoolArea"] * train["SimplPoolQC"]
# Simplified overall garage score
train["SimplGarageScore"] = train["GarageArea"] * train["SimplGarageQual"]
# Simplified overall fireplace score
train["SimplFireplaceScore"] = train["Fireplaces"] * train["SimplFireplaceQu"]
# Simplified overall kitchen score
train["SimplKitchenScore"] = train["KitchenAbvGr"] * train["SimplKitchenQual"]
# Total number of bathrooms
train["TotalBath"] = train["BsmtFullBath"] + (0.5 * train["BsmtHalfBath"]) + \
train["FullBath"] + (0.5 * train["HalfBath"])
# Total SF for house (incl. basement)
train["AllSF"] = train["GrLivArea"] + train["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
train["AllFlrsSF"] = train["1stFlrSF"] + train["2ndFlrSF"]
# Total SF for porch
train["AllPorchSF"] = train["OpenPorchSF"] + train["EnclosedPorch"] + \
train["3SsnPorch"] + train["ScreenPorch"]
# Has masonry veneer or not
## 벽돌 유/무로 구분(있으면=1,없으면=0)
train["HasMasVnr"] = train.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})
# House completed before sale or not
train["BoughtOffPlan"] = train.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
```

**SalePrice와의 상관관계확인**


```python
# Find most important features relative to target
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
```

    Find most important features relative to target
    SalePrice         1.000
    OverallQual       0.819
    AllSF             0.817
    AllFlrsSF         0.729
    GrLivArea         0.719
                      ...  
    LandSlope        -0.040
    SimplExterCond   -0.042
    KitchenAbvGr     -0.148
    EnclosedPorch    -0.149
    LotShape         -0.286
    Name: SalePrice, Length: 88, dtype: float64


**상위 10개 feature를 가지고 2제곱, 3제곱, 제곱근항을 각각 세개씩 만듬.**


```python
# 상위 10개 feature를 가지고 2제곱, 3제곱, 제곱근항을 각각 세개씩 만듬.
# Create new features
# 3* Polynomials on the top 10 existing features
train["OverallQual-s2"] = train["OverallQual"] ** 2
train["OverallQual-s3"] = train["OverallQual"] ** 3
train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])
train["AllSF-2"] = train["AllSF"] ** 2
train["AllSF-3"] = train["AllSF"] ** 3
train["AllSF-Sq"] = np.sqrt(train["AllSF"])
train["AllFlrsSF-2"] = train["AllFlrsSF"] ** 2
train["AllFlrsSF-3"] = train["AllFlrsSF"] ** 3
train["AllFlrsSF-Sq"] = np.sqrt(train["AllFlrsSF"])
train["GrLivArea-2"] = train["GrLivArea"] ** 2
train["GrLivArea-3"] = train["GrLivArea"] ** 3
train["GrLivArea-Sq"] = np.sqrt(train["GrLivArea"])
train["SimplOverallQual-s2"] = train["SimplOverallQual"] ** 2
train["SimplOverallQual-s3"] = train["SimplOverallQual"] ** 3
train["SimplOverallQual-Sq"] = np.sqrt(train["SimplOverallQual"])
train["ExterQual-2"] = train["ExterQual"] ** 2
train["ExterQual-3"] = train["ExterQual"] ** 3
train["ExterQual-Sq"] = np.sqrt(train["ExterQual"])
train["GarageCars-2"] = train["GarageCars"] ** 2
train["GarageCars-3"] = train["GarageCars"] ** 3
train["GarageCars-Sq"] = np.sqrt(train["GarageCars"])
train["TotalBath-2"] = train["TotalBath"] ** 2
train["TotalBath-3"] = train["TotalBath"] ** 3
train["TotalBath-Sq"] = np.sqrt(train["TotalBath"])
train["KitchenQual-2"] = train["KitchenQual"] ** 2
train["KitchenQual-3"] = train["KitchenQual"] ** 3
train["KitchenQual-Sq"] = np.sqrt(train["KitchenQual"])
train["GarageScore-2"] = train["GarageScore"] ** 2
train["GarageScore-3"] = train["GarageScore"] ** 3
train["GarageScore-Sq"] = np.sqrt(train["GarageScore"])
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>GarageCars-Sq</th>
      <th>TotalBath-2</th>
      <th>TotalBath-3</th>
      <th>TotalBath-Sq</th>
      <th>KitchenQual-2</th>
      <th>KitchenQual-3</th>
      <th>KitchenQual-Sq</th>
      <th>GarageScore-2</th>
      <th>GarageScore-3</th>
      <th>GarageScore-Sq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SC60</td>
      <td>RL</td>
      <td>65.000</td>
      <td>8450</td>
      <td>2</td>
      <td>None</td>
      <td>4</td>
      <td>Lvl</td>
      <td>4</td>
      <td>Inside</td>
      <td>...</td>
      <td>1.414</td>
      <td>12.250</td>
      <td>42.875</td>
      <td>1.871</td>
      <td>16</td>
      <td>64</td>
      <td>2.000</td>
      <td>2702736</td>
      <td>4443297984</td>
      <td>40.546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SC20</td>
      <td>RL</td>
      <td>80.000</td>
      <td>9600</td>
      <td>2</td>
      <td>None</td>
      <td>4</td>
      <td>Lvl</td>
      <td>4</td>
      <td>FR2</td>
      <td>...</td>
      <td>1.414</td>
      <td>6.250</td>
      <td>15.625</td>
      <td>1.581</td>
      <td>9</td>
      <td>27</td>
      <td>1.732</td>
      <td>1904400</td>
      <td>2628072000</td>
      <td>37.148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SC60</td>
      <td>RL</td>
      <td>68.000</td>
      <td>11250</td>
      <td>2</td>
      <td>None</td>
      <td>3</td>
      <td>Lvl</td>
      <td>4</td>
      <td>Inside</td>
      <td>...</td>
      <td>1.414</td>
      <td>12.250</td>
      <td>42.875</td>
      <td>1.871</td>
      <td>16</td>
      <td>64</td>
      <td>2.000</td>
      <td>3326976</td>
      <td>6068404224</td>
      <td>42.708</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SC70</td>
      <td>RL</td>
      <td>60.000</td>
      <td>9550</td>
      <td>2</td>
      <td>None</td>
      <td>3</td>
      <td>Lvl</td>
      <td>4</td>
      <td>Corner</td>
      <td>...</td>
      <td>1.732</td>
      <td>4.000</td>
      <td>8.000</td>
      <td>1.414</td>
      <td>16</td>
      <td>64</td>
      <td>2.000</td>
      <td>3709476</td>
      <td>7144450776</td>
      <td>43.886</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SC60</td>
      <td>RL</td>
      <td>84.000</td>
      <td>14260</td>
      <td>2</td>
      <td>None</td>
      <td>3</td>
      <td>Lvl</td>
      <td>4</td>
      <td>FR2</td>
      <td>...</td>
      <td>1.732</td>
      <td>12.250</td>
      <td>42.875</td>
      <td>1.871</td>
      <td>16</td>
      <td>64</td>
      <td>2.000</td>
      <td>6290064</td>
      <td>15775480512</td>
      <td>50.080</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 144 columns</p>
</div>




```python
# 수치형 변수와 범주형 변수를 따로 저장.
categorical_features=train.select_dtypes(include=["object"]).columns
numerical_features=train.select_dtypes(exclude=['object']).columns
# 수치형 변수에서 SalePrice는 드랍
numerical_features=numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))

train_num=train[numerical_features]
train_cat=train[categorical_features]
```

    Numerical features : 117
    Categorical features : 26


**수치형 변수에있는 결측치를 median값으로 채운다**


```python
# 수치형 변수에있는 결측치를 median값으로 채운다
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))   # train데이터에서 수치형데이터의 NA개수
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
```

    NAs for numerical features in train : 81
    Remaining NAs for numerical features in train : 0



```python
skewness=train_num.apply(lambda x:skew(x))
skewness
```




    LotFrontage       -0.006
    LotArea           12.575
    Street           -15.481
    LotShape          -1.290
    Utilities        -38.118
                       ...  
    KitchenQual-3      1.229
    KitchenQual-Sq     0.140
    GarageScore-2      2.403
    GarageScore-3      5.268
    GarageScore-Sq    -1.494
    Length: 117, dtype: float64



* 아웃라이어의 영향을 줄이기위해서 비대칭(skew)한 수치형 변수들을 로그변환한다.


```python
# 아웃라이어의 영향을 줄이기위해서 비대칭한 수치형 변수들을 로그변환한다.
# 비대칭 절대값이 0.5보다크면 least moderately skewed로 간주됨.

skewness=train_num.apply(lambda x:skew(x))
skewness=skewness[abs(skewness)>0.5]
# 86개의 수치형 변수를 로그변환하겠다. (수치형변수 117개중에 86개가 skewness가 0.5보다 큰것)
print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features=skewness.index
train_num[skewed_features]=np.log1p(train_num[skewed_features])
```

    86 skewed numerical features to log transform


* one-hot encoding으로 범주형에 대해서 더미변수를 만든다.
### 더미변수

* 더미변수 만드는 이유
    * 범주형 변수로는 사용할 수 없고 연속형 변수로만 가능한 분석기법을 사용할 수 있게 해준다.
    * 예를 들어 선형 회귀분석, 로지스틱 회귀분석 등 회귀분석 계열은 원래 설명변수가 연속형 변수여야지 사용할 수 있는 분석 기법이다. <br>
하지만 만약 설명변수 중에 범주형 변수가 섞여 있다면, 그 변수를 더미변수로 변환 즉, 연속형 변수스럽게 만들어서 회귀분석을 사용할 수 있다.

* 더미변수의 의미
    * 더미변수는 회귀식에서 해당 변수의 효과를 0 또는 상수값으로 만들어 준다. 
    * 예를들어, y= ax1+bx2+c라는 회귀식이 있을때
        * 원래 회귀식에서 x2가 1이면 b만 남아서 y절편은 b+c가 된다.
        * 원래 회귀식에서 x2가 0이면 b도 0이되어서 y절편은 c가 된다.
            * 이처럼 더미변수는 회귀 기울기를 바꾸지는 않고 절편만을 바꾸어 평행하게 움직이게 하는 역할을 한다.



```python
# 범주형변수중 null인 값은 1개.
train_cat[train_cat.isnull().values]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>Alley</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>...</th>
      <th>Heating</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MoSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1379</th>
      <td>SC80</td>
      <td>RL</td>
      <td>None</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>Timber</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>SLvl</td>
      <td>...</td>
      <td>GasA</td>
      <td>Y</td>
      <td>NaN</td>
      <td>BuiltIn</td>
      <td>Fin</td>
      <td>No</td>
      <td>No</td>
      <td>May</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 26 columns</p>
</div>




```python
train_cat.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>Alley</th>
      <th>LandContour</th>
      <th>LotConfig</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>...</th>
      <th>Heating</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>GarageType</th>
      <th>GarageFinish</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MoSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SC60</td>
      <td>RL</td>
      <td>None</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>...</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>No</td>
      <td>No</td>
      <td>Feb</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SC20</td>
      <td>RL</td>
      <td>None</td>
      <td>Lvl</td>
      <td>FR2</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>...</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>No</td>
      <td>No</td>
      <td>May</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SC60</td>
      <td>RL</td>
      <td>None</td>
      <td>Lvl</td>
      <td>Inside</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>...</td>
      <td>GasA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>Attchd</td>
      <td>RFn</td>
      <td>No</td>
      <td>No</td>
      <td>Sep</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 26 columns</p>
</div>




```python
# 더미변수로만듬.
pd.get_dummies(train_cat).head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass_SC120</th>
      <th>MSSubClass_SC160</th>
      <th>MSSubClass_SC180</th>
      <th>MSSubClass_SC190</th>
      <th>MSSubClass_SC20</th>
      <th>MSSubClass_SC30</th>
      <th>MSSubClass_SC40</th>
      <th>MSSubClass_SC45</th>
      <th>MSSubClass_SC50</th>
      <th>MSSubClass_SC60</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 202 columns</p>
</div>




```python
# Create dummy features for categorical values via one-hot encoding
print("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))
train_cat = pd.get_dummies(train_cat)
print("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))
```

    NAs for categorical features in train : 1
    Remaining NAs for categorical features in train : 0


------------------------------------------------------------------------------------
### Modeling


```python
train=pd.concat([train_num, train_cat], axis=1)
print("New number of features : " + str(train.shape[1]))

X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.3, random_state=0)   # 여기서 y는 SalePrice를 로그취한것.
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))
```

    New number of features : 319
    X_train : (1019, 319)
    X_test : (437, 319)
    y_train : (1019,)
    y_test : (437,)


* 스케일링 (StandardScaler)


```python
stdSc=StandardScaler()
X_train.loc[:,numerical_features]=stdSc.fit_transform(X_train.loc[:,numerical_features])
X_test.loc[:,numerical_features]=stdSc.transform(X_test.loc[:,numerical_features])
```

    /Users/suhyun/anaconda3/envs/suhyun2/lib/python3.6/site-packages/pandas/core/indexing.py:964: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s
    /Users/suhyun/anaconda3/envs/suhyun2/lib/python3.6/site-packages/pandas/core/indexing.py:964: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.obj[item] = s



```python
# 스케일링 후
X_train.loc[:,numerical_features].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>LotShape</th>
      <th>Utilities</th>
      <th>LandSlope</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>...</th>
      <th>GarageCars-Sq</th>
      <th>TotalBath-2</th>
      <th>TotalBath-3</th>
      <th>TotalBath-Sq</th>
      <th>KitchenQual-2</th>
      <th>KitchenQual-3</th>
      <th>KitchenQual-Sq</th>
      <th>GarageScore-2</th>
      <th>GarageScore-3</th>
      <th>GarageScore-Sq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>328</th>
      <td>-1.720</td>
      <td>0.533</td>
      <td>0.063</td>
      <td>-0.930</td>
      <td>0.031</td>
      <td>0.227</td>
      <td>-0.058</td>
      <td>0.475</td>
      <td>-1.775</td>
      <td>0.436</td>
      <td>...</td>
      <td>0.377</td>
      <td>-0.161</td>
      <td>-0.158</td>
      <td>-0.170</td>
      <td>0.807</td>
      <td>0.809</td>
      <td>0.793</td>
      <td>0.221</td>
      <td>0.221</td>
      <td>0.221</td>
    </tr>
    <tr>
      <th>1026</th>
      <td>0.467</td>
      <td>0.031</td>
      <td>0.063</td>
      <td>0.671</td>
      <td>0.031</td>
      <td>0.227</td>
      <td>-0.794</td>
      <td>-0.421</td>
      <td>-0.338</td>
      <td>-1.221</td>
      <td>...</td>
      <td>0.377</td>
      <td>-0.161</td>
      <td>-0.158</td>
      <td>-0.170</td>
      <td>-0.754</td>
      <td>-0.750</td>
      <td>-0.764</td>
      <td>0.247</td>
      <td>0.247</td>
      <td>0.246</td>
    </tr>
    <tr>
      <th>843</th>
      <td>0.677</td>
      <td>-0.277</td>
      <td>0.063</td>
      <td>0.671</td>
      <td>0.031</td>
      <td>0.227</td>
      <td>-0.794</td>
      <td>-1.480</td>
      <td>-0.306</td>
      <td>-1.173</td>
      <td>...</td>
      <td>-3.632</td>
      <td>-0.161</td>
      <td>-0.158</td>
      <td>-0.170</td>
      <td>-0.754</td>
      <td>-0.750</td>
      <td>-0.764</td>
      <td>-3.955</td>
      <td>-3.955</td>
      <td>-3.963</td>
    </tr>
    <tr>
      <th>994</th>
      <td>1.156</td>
      <td>0.629</td>
      <td>0.063</td>
      <td>0.671</td>
      <td>0.031</td>
      <td>0.227</td>
      <td>2.886</td>
      <td>-0.421</td>
      <td>1.129</td>
      <td>1.070</td>
      <td>...</td>
      <td>0.939</td>
      <td>1.029</td>
      <td>1.032</td>
      <td>1.020</td>
      <td>2.058</td>
      <td>2.035</td>
      <td>2.164</td>
      <td>0.557</td>
      <td>0.557</td>
      <td>0.547</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>0.856</td>
      <td>0.953</td>
      <td>0.063</td>
      <td>-0.930</td>
      <td>0.031</td>
      <td>0.227</td>
      <td>-0.058</td>
      <td>-0.421</td>
      <td>1.161</td>
      <td>1.070</td>
      <td>...</td>
      <td>0.939</td>
      <td>0.477</td>
      <td>0.485</td>
      <td>0.455</td>
      <td>0.807</td>
      <td>0.809</td>
      <td>0.793</td>
      <td>0.462</td>
      <td>0.462</td>
      <td>0.455</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 117 columns</p>
</div>



* RMSE정의


```python
make_scorer
```




    <function sklearn.metrics._scorer.make_scorer(score_func, greater_is_better=True, needs_proba=False, needs_threshold=False, **kwargs)>




```python
lr=LinearRegression()
lr.fit(X_train,y_train)
# cross-validation 10번 한것의 평균값으로
scorer=make_scorer(mean_squared_error,greater_is_better=False)
np.sqrt(-cross_val_score(lr, X_test, y_test, scoring=scorer, cv=10)).mean()
```




    0.41796306672777817



* cross_val_score()
    * estimator : 학습할 모델
    * x: 학습시킬 훈련 데이터
    * y : 학습시킬 훈련 데이터의 Label
    * scoring : 각모델에서 사용할 평가 방법
        * Regression모델에서는 MSE를 얻기위해 주로  'neg_mean_squared_error'값 사용.
    * cv : Fold의 수


```python
scorer=make_scorer(mean_squared_error,greater_is_better=False)
# cross-validation
def rmse_cv_train(model):
    # 평가 결과는 neg_mean_squared_error값으로 리턴되는 코드, 이를 양수로 바꾸고 rmse를 계산.
    rmse=np.sqrt(-cross_val_score(model, X_train, y_train, scoring=scorer, cv=10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)
```

### [1] regularization없이 선형회귀
**1 * Linear Regression without regularization**
* RMSE on Test set : 4.917572756174604

> 오차
* 만약 모집단에서 회귀식을 얻었다면, 그 회귀식을 통해 얻은 예측값과 실제 관측값의 차이 <br>

> 잔차
* 표본집단에서 회귀식을 얻었다면, 그 회귀식을 통해 얻은 예측값과 실제 관측값의 차이


```python
lr=LinearRegression()
lr.fit(X_train,y_train)

# Look at predictions on training and validation set
print("RMSE on Training set :", rmse_cv_train(lr).mean())
print("RMSE on Test set :", rmse_cv_test(lr).mean())

y_train_pred=lr.predict(X_train)
y_test_pred=lr.predict(X_test)

# 잔차=> 예측값-실제값
## x축 : 예측값, y축: 예측값-실제값
plt.scatter(y_train_pred, y_train_pred-y_train, c="blue", marker="s", label="Training data")
plt.scatter(y_test_pred, y_test_pred-y_test, c="lightgreen", marker="s", label="Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
plt.show()

# 예측값 그리기
# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
```

    RMSE on Training set : 0.3889446260408515
    RMSE on Test set : 0.41796306672777817



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_48_1.png)



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_48_2.png)


> Error는 센터 라인 주변으로 랜덤하게 분포되어있다. 그 의미는 이 모델이 대부분의 설명가능한 정보를 포함하고있다는것을 의미한다.
*  (+) training set에 RMSE값은 좀 이상함.

### [2] Ridge(릿지) L2 penalty
**2 * Linear Regression with Ridge regularization (L2 penalty)** 

* Regularization은 다중공선성을 다루고, 데이터의 노이즈를 필터링하고, overfitting을 방지하는데 유용한 방법이다. regularization의 컨셉은 극단적인 파라미터의 weights에 패널티를 주기위해 bias라는 추가적인 정보를 도입하는것이다.
* L2 penalized model은 squared sum of the weights 를 cost function에 추가하는 것이다.

> 계수를 0에 가깝게 만드는 모델이므로, 아래 bar-plot으로 계수(coefficeint)가 큰것 상위10개, 계수가 작은것 하위 10개를 나타냄.


```python
# 2* Ridge
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train, y_train)
# alpha 값은 규제의 강도를 의미
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)

# Plot residuals
plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
```

    Best alpha : 30.0
    Try again for more precision with alphas centered around 30.0
    Best alpha : 24.0
    Ridge RMSE on Training set : 0.11540572328450796
    Ridge RMSE on Test set : 0.11642721377799559



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_51_1.png)



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_51_2.png)



```python
# 계수가 0인애들
coefs = pd.Series(ridge.coef_, index = X_train.columns)
coefs[coefs==0]
```




    Condition2_RRAe    0.000
    RoofMatl_Membran   0.000
    MiscFeature_TenC   0.000
    dtype: float64




```python
coefs.sort_values()
```




    MSZoning_C (all)        -0.055
    Neighborhood_Edwards    -0.038
    Neighborhood_IDOTRR     -0.033
    SaleCondition_Abnorml   -0.030
    Condition1_Artery       -0.030
                             ...  
    LotArea                  0.037
    Condition1_Norm          0.040
    YearBuilt                0.041
    OverallGrade             0.041
    Neighborhood_Crawfor     0.066
    Length: 319, dtype: float64




```python
# Plot important coefficients
coefs = pd.Series(ridge.coef_, index = X_train.columns)
print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
plt.show()
```

    Ridge picked 316 features and eliminated the other 3 features



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_54_1.png)


### [3] Lasso(라쏘)
**3 * Linear Regression with Lasso regularization (L1 penalty)**
* 여기서는 릿지보다 라쏘가 더 효과적일것이라 기대.
* 라쏘는 덜 중요한 변수의 가중치를 0으로 만들어버려 변수선택의 효과가 있다)


```python
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha=lasso.alpha_
print("Best alpha :", alpha)   # Best alpha : 0.0006

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
print("Lasso RMSE on Test set:", rmse_cv_test(lasso).mean())
y_train_las=lasso.predict(X_train)
y_test_las=lasso.predict(X_test)

# Plot residuals
plt.scatter(y_train_las, y_train_las-y_train, c='blue', marker='s', label='Training data')
plt.scatter(y_test_las, y_test_las - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_las, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_las, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Lasso regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
```

    Best alpha : 0.0006
    Try again for more precision with alphas centered around 0.0006
    Best alpha : 0.0006
    Lasso RMSE on Training set : 0.11411150837458059
    Lasso RMSE on Test set: 0.11583213221750707



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_56_1.png)



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_56_2.png)



```python
# Plot important coefficients
coefs=pd.Series(lasso.coef_, index=X_train.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs=pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
imp_coefs.plot(kind='barh')
plt.title("Coefficients in the Lasso Model")
plt.show()
```

    Lasso picked 111 features and eliminated the other 208 features



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_57_1.png)


* training과 test set에서 RMSE값이 더 좋아졌다. 여기서 인상깊은 점은 Lasso는 오직 111개의 feature만 사용했다는 점이다. (Ridge는 316개사용)
* 또다른 특이점은 양수와 음수인것 모두 Neighborhood 카테고리에 큰 가중치를 줬다는 것이다.
    * 이것은 일리가있다. 집가격은 한 neighborhood에서 같은 지역에 다른 neighborohod에 의해 변동이 많기때문에.
* MSZoning_C(all) 변수는 다른것들에 비해 과장된 효과인것으로 보임.
    * 가장 상업적인 구역에 집이 있는것은 끔찍하기 때문에 나에겐 좀 이상해보임.

### [4] ElasticNet
**4 * Linear Regression with ElasticNet regularization (L1 and L2 penalty)**

* 엘라스틱 넷(Elastic Net)은 릿지회귀와 라쏘회귀를 절충한 모델이다.
    * 규제항은 릿지와 회귀의 규제항을 단순히 더해서 사용하며, 두 규제항의 혼합정도를 혼합비율 r을 사용해 조절한다.
        * r=0이면 릿지회귀와 같고, r=1이면 라쏘회귀와 같다.
        
* 언제 엘라스틱 넷을 사용하면 좋을까
    * 극단적으로 변수의 수가 훈련샘플의 수 보다도 많고, 변수 몇개가 강하게 연관(다중공선성 의심)되어 있을 경우에는 라쏘에 문제가 발생하므로 엘라스틱넷을 선호한다.


```python
# 4* ElasticNet
elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(X_train, y_train)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(X_train, y_train)
if (elasticNet.l1_ratio_ > 1):   # 릿지보다 라쏘모델이 더 좋았으므로, r=1로해서 라쏘회귀에 가깝게조정
    elasticNet.l1_ratio_ = 1    
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 
      " and alpha centered around " + str(alpha))
elasticNet = ElasticNetCV(l1_ratio = ratio,
                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 
                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
                                    alpha * 1.35, alpha * 1.4], 
                          max_iter = 50000, cv = 10)
elasticNet.fit(X_train, y_train)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1    
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())
print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())
y_train_ela = elasticNet.predict(X_train)
y_test_ela = elasticNet.predict(X_test)

# Plot residuals
plt.scatter(y_train_ela, y_train_ela - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_ela, y_test_ela - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train, y_train_ela, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test, y_test_ela, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with ElasticNet regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()
```

    /Users/suhyun/anaconda3/envs/suhyun2/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:472: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 0.05083180988642688, tolerance: 0.01426910245430051
      tol, rng, random, positive)
    /Users/suhyun/anaconda3/envs/suhyun2/lib/python3.6/site-packages/sklearn/linear_model/_coordinate_descent.py:472: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 0.04478905990840776, tolerance: 0.014439328514818323
      tol, rng, random, positive)


    Best l1_ratio : 1.0
    Best alpha : 0.0006
    Try again for more precision with l1_ratio centered around 1.0
    Best l1_ratio : 1.0
    Best alpha : 0.0006
    Now try again for more precision on alpha, with l1_ratio fixed at 1.0 and alpha centered around 0.0006
    Best l1_ratio : 1.0
    Best alpha : 0.0006
    ElasticNet RMSE on Training set : 0.11411150837458059
    ElasticNet RMSE on Test set : 0.11583213221750707



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_60_2.png)



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_60_3.png)



```python
# Plot important coefficients
coefs = pd.Series(elasticNet.coef_, index = X_train.columns)
print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the ElasticNet Model")
plt.show()
```

    ElasticNet picked 111 features and eliminated the other 208 features



![png](/images/2021-06-13-kaggle-HousePrices-Regression_files/2021-06-13-kaggle-HousePrices-Regression_61_1.png)



```python

```


```python

```
