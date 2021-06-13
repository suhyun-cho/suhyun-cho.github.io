---
layout: single
title : "Kaggle필사 - HousePrices Basic EDA"
author_profile: true
read_time: false
comments: true
categories:
- Kaggle
---

# House Prices
## 1. Comprehensive Data Exploration with Python
[캐글]: https://www.kaggle.com/c/house-prices-advanced-regression-techniques <br>
[참고 커널모음] :https://subinium.github.io/kaggle-tutorial/house-prices <br>

#### Python Tutorials
아래 튜토리얼 하나씩 필사할 예정

> * Comprehensive Data Exploration with Python (필사 시작일 : 2020-09-01)
https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

> * House Prices EDA
https://www.kaggle.com/dgawlik/house-prices-eda

> * A Study on Regression Applied to the Ames Dataset
https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

> * Regularized Linear Models
https://www.kaggle.com/apapiu/regularized-linear-models


--------------------------------------------------------------------------------------------------
## What I learned
* type here

--------------------------------------------------------------------------------------------------




```python
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
df_train = pd.read_csv('./data/train.csv')
df_submission=pd.read_csv('./data/sample_submission.csv')
df_test=pd.read_csv('./data/test.csv')
```


```python
df_submission
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
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>169277.052498</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>187758.393989</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>183583.683570</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>179317.477511</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>150730.079977</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>2915</td>
      <td>167081.220949</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2916</td>
      <td>164788.778231</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>2917</td>
      <td>219222.423400</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>2918</td>
      <td>184924.279659</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>2919</td>
      <td>187741.866657</td>
    </tr>
  </tbody>
</table>
<p>1459 rows × 2 columns</p>
</div>




```python
df_train.head()
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
      <td>65.0</td>
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
      <td>80.0</td>
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
      <td>68.0</td>
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
      <td>60.0</td>
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
      <td>84.0</td>
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
df_train.columns
```




    Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'SalePrice'],
          dtype='object')



--------------------------------------------------------------------------------------------------
### 1. So... What can we expect?
* Variable
* Type - 여기서는 numerical, categorical 두가지 타입이 있다. 
* Segment - building(e.g. 'OverallQual'), space(e.g. 'TotalBsmtSF'), location(e.g. 'Neighborhood') 세가지 분류로 변수를 나눠볼수있다.
* Expectation - 변수가 'SalePrice'에 영향을 미친다는 기대를 갖고있다. 
* Conclusion 

--------------------------------------------------------------------------------------------------
### 2. First things first: analysing 'SalePrice'


```python
df_train['SalePrice'].describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64




```python
sns.distplot(df_train['SalePrice'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b52fd68>




![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_10_1.png)



```python
#skewness and kurtosis
## 왜도, 첨도
print("Skewness: %f" % df_train['SalePrice'].skew())
print('Kurtosis: %f' % df_train['SalePrice'].kurt())
```

    Skewness: 1.882876
    Kurtosis: 6.536282


#### Relationship with numerical variables


```python
#scatter plot grlivarea/saleprice
var='GrLivArea'
data=pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b243898>




![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_13_1.png)


'SalePrice' 와 'GrLivArea' 사이에는 선형관계가 있는것으로 보여짐.


```python
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_15_0.png)


#### Relationship with categorical features


```python
#box plot overallqual/saleprice
## OverallQual: Overall material and finish quality
var='OverallQual'
data=pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=var, y="SalePrice",data=data)
fig.axis(ymin=0, ymax=800000)
```




    (-0.5, 9.5, 0.0, 800000.0)




![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_17_1.png)



```python
var='YearBuilt'
data=pd.concat([df_train['SalePrice'],df_train[var]], axis=1)
f,ax=plt.subplots(figsize=(16,8))
fig=sns.boxplot(x=var, y="SalePrice",data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90);
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_18_0.png)


최근 지은 건물이 가격이 조금 높아보이기는하나, 무조건 최근 건물이라고 비싼건 아님. 과거에 지은 건물인데도 비싼 경우가 보임.

------------------------------------------------------------------------------------------------
### 3. Keep calm and work smart


```python
#correlation matrix
corrmat=df_train.corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True);
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_21_0.png)


'TotalBsmtSF' 와 '1stFlrSF' 변수 간의 상관관계, 그리고 'GarageX' 변수가 눈에 띈다. <br>
이 변수들은 상관관계가 높아서 다중공선성이 의심됨.<br>

 'SalePrice' 와의 상관관계를 보면,  'GrLivArea', 'TotalBsmtSF', 그리고 'OverallQual' 변수가 높다.

#### 'SalePrice' correlation matrix (zoomed heatmap style)


```python
#saleprice correlation matrix
k=10 #number of variables for heatmap
cols=corrmat.nlargest(k,'SalePrice')['SalePrice'].index   # SalePrice와 상관관계가 높은 상위10개

# np.corrcoef()는 행을 기준으로 값들을 변수로 생각해서 상관계수를 구하기때문에 transform해준것.
cm=np.corrcoef(df_train[cols].values.T)  # 피어슨 상관계수 값을 계산해준다.
sns.set(font_scale=1.25)
# yticklabels, xticklabels로 라벨을 따로 달아준것.
hm=sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10},
              yticklabels=cols.values, xticklabels=cols.values);
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_24_0.png)


* 'OverallQual', 'GrLivArea' , 'TotalBsmtSF' 변수들은 'SalePrice'와 강하게 상관성을 띈다.
* 'GarageCars' 와  'GarageArea' 또한 상관관계가 높은 변수이다.
    * 그런데, GarageCars' 와  'GarageArea' 는 한쌍이라고 볼수있음. 주차장 면적이 넓을수록, 수용할수있는 차의 대수가 많아지기 때문에 
    * 이경우 종속변수와 상관관계가 더 높은 'GarageCars' 변수만 가져가기로 하자.
* 'TotalBsmtSF'와 '1stFloor' 또한 한쌍으로 보인다. 여기서 우리는 'TotalBsmtSF'변수를 가져가기로 한다.
* 'TotRmsAbvGrd' 와'GrLivArea' 또한 한쌍이다. 집 면적이 넓을수록 방의 개수도 많을것임.
* 'YearBuilt' 는 종속변수와 약간의 상관성을 갖고있는데, 시계열로 접근해야할것 같다는 느낌이 듬.

#### Scatter plots between 'SalePrice' and correlated variables (move like Jagger style)


```python
sns.set()
cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols],size=2.5);
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_27_0.png)


'TotalBsmtSF' 와 'GrLiveArea'의 관계를 보면, 점들이 선형적인 관계로 나타나는 것을 확인할 수 있다. <br>
사실 Basement areas는 above ground living area와 비슷하다고볼 수 있다. <br>

'SalePrice' 와 'YearBuilt'간의 관계를 보면 지수함수 형태로 생긴것을 확인할 수 있다. 

------------------------------------------------------------------------------
### 4. Missing data
* missing data가 얼마나 퍼져있는지?
* missing data가 랜덤한지, 패턴이 있는지?

missing data는 샘플 사이즈 감소로 이어질수있기 때문에 위 두질문은 중요하다. 


```python
#missing data
total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_data.head(20)
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
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>0.995205</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>0.963014</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>0.937671</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>0.807534</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>0.472603</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>0.177397</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.000685</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



* 만약 15%이상이 결측이면 상응하는 변수를 제거할것이다.
    * 즉, 이경우에는 결측치를 채우기위한 트릭을 사용하지 않을 것이다.
* 그러므로 결측이 많은 (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.)  변수는 제거
*  'GarageX'  변수들은 결측치 비율이 모두 동일하다. 'GarageCars' 변수가 garage관련 변수들중 가장 중요한 정보를 담고있으므로 다른 변수들은 제거
*  'BsmtX' 변수들도 같은맥락으로 제거
* MasVnrArea' 와'MasVnrType'는 중요한 변수라고 보이진 않음. 또한, 이들은 'YearBuilt'와 'OverallQual'와 상관관계가 높으므로 이 변수를 삭제하더라도 무관할것이다.
* 'Electrical' 는 결측치가 오직 1개이므로 유지. 결측치가있는 관측치는 제거.


```python
#dealing with missing data
df_train=df_train.drop((missing_data[missing_data['Total'] >1]).index,1)  # 컬럼자체를 제거
df_train=df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)   # 결측치가있는 부분의 행 제거
df_train.isnull().sum().max()
```




    0



### Out liars

Univariate analysis
* 이상치라고 판단할 threshold를 만들어야한다. 그러기 위해서 데이터를 표준화할것이다. <br>
여기서 표준화는 평균을 0으로하고 표준편차를 1로하는것을 말한다.


```python
#standardizing data
from sklearn.preprocessing import StandardScaler
# np.newaxis => indexing으로 길이가1인 새로운 축 추가
saleprice_scaled=StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])

# argsort => 작은값부터 순서대로 데이터의 index를 반환.
## 반환한 index에 saleprice_scaled를 한번씌워서 실제 값 나오도록.
low_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range(low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
```

    outer range(low) of the distribution:
    [[-1.83820775]
     [-1.83303414]
     [-1.80044422]
     [-1.78282123]
     [-1.77400974]
     [-1.62295562]
     [-1.6166617 ]
     [-1.58519209]
     [-1.58519209]
     [-1.57269236]]
    
    outer range (high) of the distribution:
    [[3.82758058]
     [4.0395221 ]
     [4.49473628]
     [4.70872962]
     [4.728631  ]
     [5.06034585]
     [5.42191907]
     [5.58987866]
     [7.10041987]
     [7.22629831]]


* low range값들은 비슷하고, 0으로부터 멀지 않다. 
* high range값들은 0으로부터 멀리 떨어져있고, 7같은 것들은 범위에서 많이 벗어남.

이제부터 저 이상치들은 고려하지 않을것이지만, 위에 두개의 7. 값들은 조심해야할 필요가 있다.

Bivariate analysis


```python
#bivariate analysis saleprice/grlivarea
var='GrLivArea'
data=pd.concat([df_train['SalePrice'], df_train[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.



![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_38_1.png)


* 면적이 큰데, 가격이 높지않은 두개의 점이 보인다. 이를 유추해보면, 농경지여서 가격이 낮을 수 있다. <br>
이 둘은 대표값을 나타내기 어려우므로 이상치로 취급하고 삭제한다. <br>
* 하지만, 위쪽에 두 점은 특이케이스처럼 보이기는 하지만 트렌드를 따라가므로 (면적이 커질수록 가격이 높아지는) 유지한다.


```python
#deleting points
df_train.sort_values(by='GrLivArea',ascending=False)[:2]
df_train=df_train.drop(df_train[df_train['Id']==1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
```


```python
#bivariate analysis saleprice/grlivarea
var='TotalBsmtSF'
data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0.800000));
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.



![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_41_1.png)


3000이상의 값들은 제거해야할것 같지만, 그냥 두기로 한다.

------------------------------------------------------------------------------------------
### 5. Getting hard core
이제 SalePrice가 multivariate techniques를 적용할수있는 통계적 가정을 만족하는지 살펴보자.
* Normality (정규성)
* Homoscedasticity (분산의 동질성, 등분산성)
* Linearity (선형성)
    * 선형성은 scatter plot을 통해 알수있다. 여기서는 scatter-plot을 그려봤을때 대부분 선형으로 보였으므로 패스함
* Absence of correlated errors (오차간의 상관성)

In the search for normality
* Histogram - Kurtosis and skewness.
* Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.


```python
#histogram and normal probability plot
sns.distplot(df_train['SalePrice'],fit=norm);
fig=plt.figure()
res=stats.probplot(df_train['SalePrice'],plot=plt)
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_45_0.png)



![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_45_1.png)


SalePrice는 정규성을 띄지 않는것으로 나타남.
* 이처럼 양의왜도를 띄는 경우, 로그변환으로 정규성을 띄게만들 수 있음.


```python
#applying log transformation
df_train['SalePrice']=np.log(df_train['SalePrice'])
```


```python
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'],fit=norm);
fig=plt.figure()
res=stats.probplot(df_train['SalePrice'], plot=plt)
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_48_0.png)



![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_48_1.png)


GrLivArea 변수에 대해서도 알아보자.


```python
#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'],fit=norm);
fig=plt.figure()
res=stats.probplot(df_train['GrLivArea'],plot=plt)
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_50_0.png)



![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_50_1.png)



```python
#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
```


```python
#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_52_0.png)



![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_52_1.png)



```python
#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_53_0.png)



![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_53_1.png)


* 몇몇 값들이 왜도를 띈다.
* 0의 값을 갖는것에 주목 (basement가 없는 집일경우)
* 0값이 있는 경우 로그변환을 취하기 어려운것이 문제.

로그변환을 취할 수 있도록 만들기 위해, basement가 있거나 없는것에 대한 효과를 갖는 새로운 변수를 만들것이다.  (이진변수) <br>
그 다음에 0이 아닌 값들에 대해 로그변환을 취할 것이다.<br>
이렇게 함으로써,  basement를 갖고있거나 갖지 않은것에 대한 효과를 무지하지 않은채로 데이터를 변환할 수 있다.


#### In the search for writing 'homoscedasticity' right at the first attempt
* 분산의 동질성


```python
# scatter plot
plt.scatter(df_train['GrLivArea'],df_train['SalePrice']);
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_56_0.png)


* 로그를 취하기 전에는 콘 모양으로 퍼져있는 scatter-plot이 그려졌는데, 로그를취하고난 이후 그런 모양이 사라진것을 확인함.


```python
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
```


![png](/images/2021-06-13-kaggle-HousePrices-Basic-EDA_files/2021-06-13-kaggle-HousePrices-Basic-EDA_58_0.png)


* 위 결과 SalePrice가 TotalBsmtSF의 분산과 동일한 수준이라는것을 보여준다.

-------------------------------------------------------------------------------------
### dummy variables


```python
#convert categorical variable into dummy
df_train=pd.get_dummies(df_train)
```


```python
df_train.iloc[:,15:]
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
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
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
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>548</td>
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
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>460</td>
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
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>608</td>
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
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>3</td>
      <td>642</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>836</td>
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
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>460</td>
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
      <th>1456</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>500</td>
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
      <th>1457</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>252</td>
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
      <th>1458</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>240</td>
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
      <th>1459</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>276</td>
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
<p>1457 rows × 206 columns</p>
</div>




```python

```
