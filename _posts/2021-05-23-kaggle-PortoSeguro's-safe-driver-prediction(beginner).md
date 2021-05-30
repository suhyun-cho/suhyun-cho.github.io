---
layout: single
title : "Kaggle - PortoSeguro's safe driver prediction_v1 (for beginner)"
author_profile: true
read_time: false
comments: true
categories:
- Kaggle
---

# Porto Seguro의 운전자 보험청구 예측 (for beginner)

* Data Preparation & Exploration
    * https://www.kaggle.com/bertcarremans/data-preparation-exploration

-----------------------------
### 대회 설명
운전자가 내년에 보험 청구를 할 것인지를 예측하는 대회.<br>
58개의 컬럼과 1개의 Target 컬럼으로 이루어져있음

### Normalized Gini Coefficient (지니계수)
* 머신러닝 분야에서 Decision Tree모델의 성능을 평가할 때 엔트로피 지수와 함께 쓰인다. 
    * 결정트리는 트리구조를 형성할때 순도가 증가하고, 불순도가 최대한 작아지는 방향으로 결정을 내려감.
* 지니 계수는 0~0.5 의 값을 가지는데 값이 작을수록 분류가 잘 되었다고 볼 수 있음
    * 불순도(섞인 정도)가 적은것이 분류가 잘된것
* 이 대회에서 지니 계수는 Actual 값의 누적 비율과 Prediciton값의 누적 비율로 산출이되는데, 이것을 표준화 시킨다.
    * 이 표준화는 Actual값이 가지고 있는 불평등 정도로 actual-prediction간의 불평등 정도를 나누어서 산출된다.


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

    /kaggle/input/porto-seguro-safe-driver-prediction/sample_submission.csv
    /kaggle/input/porto-seguro-safe-driver-prediction/train.csv
    /kaggle/input/porto-seguro-safe-driver-prediction/test.csv


-----------------------------
### Introduction
1. Visual inspection of your data
2. Defining the metadata
3. Descriptive statistics
4. Handling imbalanced classes
5. Data quality checks
6. Exploratory data visualization
7. Feature engineering
8. Feature selection
9. Feature scaling
-----------------------------


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)
```


```python
import os
print(os.listdir("../input"))
```

    ['porto-seguro-safe-driver-prediction']


### Loading data


```python
train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')
```

### Data at first sight
**데이터 특징** <br>
1. 컬럼 이름(ind, reg, car, calc)로 Grouping된다
2. 컬럼 이름에 '_bin'은 Binary Features, '_cat'은 Categorical Features를 의미
3. 컬럼 이름에 아무것도 안붙어있으면 Continuous or Ordinal Features를 의미
4. -1 은 Null값을 의미


```python
train.shape
```




    (595212, 59)




```python
# 중복되는 row가 있는지 확인하기위해 drop_duplicates()사용

train.drop_duplicates()
train.shape
```




    (595212, 59)




```python
test.shape
```




    (892816, 58)




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
      <th>id</th>
      <th>target</th>
      <th>ps_ind_01</th>
      <th>ps_ind_02_cat</th>
      <th>ps_ind_03</th>
      <th>ps_ind_04_cat</th>
      <th>ps_ind_05_cat</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_01_cat</th>
      <th>ps_car_02_cat</th>
      <th>ps_car_03_cat</th>
      <th>ps_car_04_cat</th>
      <th>ps_car_05_cat</th>
      <th>ps_car_06_cat</th>
      <th>ps_car_07_cat</th>
      <th>ps_car_08_cat</th>
      <th>ps_car_09_cat</th>
      <th>ps_car_10_cat</th>
      <th>ps_car_11_cat</th>
      <th>ps_car_11</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.2</td>
      <td>0.718070</td>
      <td>10</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>2</td>
      <td>0.400000</td>
      <td>0.883679</td>
      <td>0.370810</td>
      <td>3.605551</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>0.2</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
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
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.8</td>
      <td>0.4</td>
      <td>0.766078</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>19</td>
      <td>3</td>
      <td>0.316228</td>
      <td>0.618817</td>
      <td>0.388716</td>
      <td>2.449490</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.000000</td>
      <td>7</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>60</td>
      <td>1</td>
      <td>0.316228</td>
      <td>0.641586</td>
      <td>0.347275</td>
      <td>3.316625</td>
      <td>0.5</td>
      <td>0.7</td>
      <td>0.1</td>
      <td>2</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.580948</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>104</td>
      <td>1</td>
      <td>0.374166</td>
      <td>0.542949</td>
      <td>0.294958</td>
      <td>2.000000</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>0.1</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.7</td>
      <td>0.6</td>
      <td>0.840759</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>82</td>
      <td>3</td>
      <td>0.316070</td>
      <td>0.565832</td>
      <td>0.365103</td>
      <td>2.000000</td>
      <td>0.4</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>10</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.tail()
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
      <th>id</th>
      <th>target</th>
      <th>ps_ind_01</th>
      <th>ps_ind_02_cat</th>
      <th>ps_ind_03</th>
      <th>ps_ind_04_cat</th>
      <th>ps_ind_05_cat</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_01_cat</th>
      <th>ps_car_02_cat</th>
      <th>ps_car_03_cat</th>
      <th>ps_car_04_cat</th>
      <th>ps_car_05_cat</th>
      <th>ps_car_06_cat</th>
      <th>ps_car_07_cat</th>
      <th>ps_car_08_cat</th>
      <th>ps_car_09_cat</th>
      <th>ps_car_10_cat</th>
      <th>ps_car_11_cat</th>
      <th>ps_car_11</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>595207</th>
      <td>1488013</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
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
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.692820</td>
      <td>10</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31</td>
      <td>3</td>
      <td>0.374166</td>
      <td>0.684631</td>
      <td>0.385487</td>
      <td>2.645751</td>
      <td>0.4</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>595208</th>
      <td>1488016</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
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
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.7</td>
      <td>1.382027</td>
      <td>9</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>63</td>
      <td>2</td>
      <td>0.387298</td>
      <td>0.972145</td>
      <td>-1.000000</td>
      <td>3.605551</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>595209</th>
      <td>1488017</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.2</td>
      <td>0.659071</td>
      <td>7</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>31</td>
      <td>3</td>
      <td>0.397492</td>
      <td>0.596373</td>
      <td>0.398748</td>
      <td>1.732051</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>3</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>595210</th>
      <td>1488021</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>0.698212</td>
      <td>11</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>101</td>
      <td>3</td>
      <td>0.374166</td>
      <td>0.764434</td>
      <td>0.384968</td>
      <td>3.162278</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>9</td>
      <td>2</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>595211</th>
      <td>1488027</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>-1.000000</td>
      <td>7</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>34</td>
      <td>2</td>
      <td>0.400000</td>
      <td>0.932649</td>
      <td>0.378021</td>
      <td>3.741657</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>2</td>
      <td>3</td>
      <td>10</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# _cat로 끝나는 14개의 categorical 변수에 대해서 더미변수를 만들것이다.
# _bin으로 끝나는 binary 변수는 이미 binary이기때문에 더미화할필요없음.
## 아래 info를 보면, 데이터 타입이 int 또는 float이다. null값은 -1로 되어있기 때문에 전부 non-null로나옴. 
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 595212 entries, 0 to 595211
    Data columns (total 59 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   id              595212 non-null  int64  
     1   target          595212 non-null  int64  
     2   ps_ind_01       595212 non-null  int64  
     3   ps_ind_02_cat   595212 non-null  int64  
     4   ps_ind_03       595212 non-null  int64  
     5   ps_ind_04_cat   595212 non-null  int64  
     6   ps_ind_05_cat   595212 non-null  int64  
     7   ps_ind_06_bin   595212 non-null  int64  
     8   ps_ind_07_bin   595212 non-null  int64  
     9   ps_ind_08_bin   595212 non-null  int64  
     10  ps_ind_09_bin   595212 non-null  int64  
     11  ps_ind_10_bin   595212 non-null  int64  
     12  ps_ind_11_bin   595212 non-null  int64  
     13  ps_ind_12_bin   595212 non-null  int64  
     14  ps_ind_13_bin   595212 non-null  int64  
     15  ps_ind_14       595212 non-null  int64  
     16  ps_ind_15       595212 non-null  int64  
     17  ps_ind_16_bin   595212 non-null  int64  
     18  ps_ind_17_bin   595212 non-null  int64  
     19  ps_ind_18_bin   595212 non-null  int64  
     20  ps_reg_01       595212 non-null  float64
     21  ps_reg_02       595212 non-null  float64
     22  ps_reg_03       595212 non-null  float64
     23  ps_car_01_cat   595212 non-null  int64  
     24  ps_car_02_cat   595212 non-null  int64  
     25  ps_car_03_cat   595212 non-null  int64  
     26  ps_car_04_cat   595212 non-null  int64  
     27  ps_car_05_cat   595212 non-null  int64  
     28  ps_car_06_cat   595212 non-null  int64  
     29  ps_car_07_cat   595212 non-null  int64  
     30  ps_car_08_cat   595212 non-null  int64  
     31  ps_car_09_cat   595212 non-null  int64  
     32  ps_car_10_cat   595212 non-null  int64  
     33  ps_car_11_cat   595212 non-null  int64  
     34  ps_car_11       595212 non-null  int64  
     35  ps_car_12       595212 non-null  float64
     36  ps_car_13       595212 non-null  float64
     37  ps_car_14       595212 non-null  float64
     38  ps_car_15       595212 non-null  float64
     39  ps_calc_01      595212 non-null  float64
     40  ps_calc_02      595212 non-null  float64
     41  ps_calc_03      595212 non-null  float64
     42  ps_calc_04      595212 non-null  int64  
     43  ps_calc_05      595212 non-null  int64  
     44  ps_calc_06      595212 non-null  int64  
     45  ps_calc_07      595212 non-null  int64  
     46  ps_calc_08      595212 non-null  int64  
     47  ps_calc_09      595212 non-null  int64  
     48  ps_calc_10      595212 non-null  int64  
     49  ps_calc_11      595212 non-null  int64  
     50  ps_calc_12      595212 non-null  int64  
     51  ps_calc_13      595212 non-null  int64  
     52  ps_calc_14      595212 non-null  int64  
     53  ps_calc_15_bin  595212 non-null  int64  
     54  ps_calc_16_bin  595212 non-null  int64  
     55  ps_calc_17_bin  595212 non-null  int64  
     56  ps_calc_18_bin  595212 non-null  int64  
     57  ps_calc_19_bin  595212 non-null  int64  
     58  ps_calc_20_bin  595212 non-null  int64  
    dtypes: float64(10), int64(49)
    memory usage: 267.9 MB


### Metadata
데이터 관리를 용이하게 하기 위해서 변수에대한 meta정보를 DataFrame에저장할것이다.<br>
이는 이후 분석, 시각화, 모델링시에 유용하다
> * role : input, ID, target
* level:norminal(명목변수), interval(간격변수), ordinal(순위변수), binary
    * (참고) 명목변수, 순위변수는 범주형변수에 속함. 간격변수는 연속형변수
* keep: True or False
* dtype : int, float, str

**Python-Exercise [1]**



```python
"""
list에 for loop돌면서 딕셔너리를 append해주고, 최종적으로 DataFrame으로 만들어주는 방식
"""

data=[]
data.append({'varname':'var1','dtype':'int'})  # 1st for loop
data
```




    [{'varname': 'var1', 'dtype': 'int'}]




```python
data.append({'varname':'var2','dtype':'float'})  # 2nd for loop
data
```




    [{'varname': 'var1', 'dtype': 'int'}, {'varname': 'var2', 'dtype': 'float'}]




```python
# final DataFrame
pd.DataFrame(data)
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
      <th>varname</th>
      <th>dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>var1</td>
      <td>int</td>
    </tr>
    <tr>
      <th>1</th>
      <td>var2</td>
      <td>float</td>
    </tr>
  </tbody>
</table>
</div>




```python
data=[]
for f in train.columns:
    # role 정의
    if f=='target':
        role='target'
    elif f=='id':
        role='id'
    else:
        role='input'
        
    # level 정의
    if 'bin' in f or f=='target':
        level='binary'
    elif 'cat' in f or f=='id':
        level='nominal'
    elif train[f].dtype==float:
        level='interval'
    elif train[f].dtype==int:
        level='ordinal'
        
    # id를 제외한 나머지는 keep에 True값 지정
    keep=True
    if f=='id':
        keep=False
        
    # 데이터타입 정의
    dtype=train[f].dtype
    
    # 모든 metadata를 포함하고있는 딕셔너리를 만든다.
    f_dict={
        'varname':f,
        'role':role,
        'level':level,
        'keep':keep,
        'dtype':dtype
    }
    data.append(f_dict)
    
meta=pd.DataFrame(data,columns=['varname','role','level','keep','dtype'])
meta.set_index('varname',inplace=True)
    
```


```python
meta
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
      <th>role</th>
      <th>level</th>
      <th>keep</th>
      <th>dtype</th>
    </tr>
    <tr>
      <th>varname</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>id</td>
      <td>nominal</td>
      <td>False</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>target</th>
      <td>target</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_01</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_02_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_03</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_04_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_05_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_06_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_07_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_08_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_09_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_10_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_11_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_12_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_13_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_14</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_15</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_16_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_17_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_ind_18_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_reg_01</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_reg_02</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_reg_03</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_01_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_02_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_03_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_04_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_05_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_06_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_07_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_08_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_09_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_10_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_11_cat</th>
      <td>input</td>
      <td>nominal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_11</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_car_12</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_13</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_14</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_car_15</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_01</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_02</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_03</th>
      <td>input</td>
      <td>interval</td>
      <td>True</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>ps_calc_04</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_05</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_06</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_07</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_08</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_09</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_10</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_11</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_12</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_13</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_14</th>
      <td>input</td>
      <td>ordinal</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_15_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_16_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_17_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_18_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_19_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ps_calc_20_bin</th>
      <td>input</td>
      <td>binary</td>
      <td>True</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
</div>




```python
# nominal변수 확인
meta[(meta.level=='nominal')&(meta.keep)].index
```




    Index(['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
           'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',
           'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
           'ps_car_10_cat', 'ps_car_11_cat'],
          dtype='object', name='varname')




```python
# role별, level별 변수 개수
pd.DataFrame({'count':meta.groupby(['role','level'])['role'].size()}).reset_index()
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
      <th>role</th>
      <th>level</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>nominal</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>input</td>
      <td>binary</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>input</td>
      <td>interval</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>input</td>
      <td>nominal</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>input</td>
      <td>ordinal</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>target</td>
      <td>binary</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



-----------------------------
## Descriptive statistics
describe함수를 사용해서 데이터를 살펴볼수도있지만, categorical 변수와 id변수에서 평균, 표준편차 등을 계산하는건 의미가 없다. <br>
따라서, categorical 변수에 대해서는 이후에 탐색할것이다.

### 1) Interval variables


```python
v=meta[(meta.level=='interval')&(meta.keep)].index
train[v].describe()
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
      <th>ps_reg_01</th>
      <th>ps_reg_02</th>
      <th>ps_reg_03</th>
      <th>ps_car_12</th>
      <th>ps_car_13</th>
      <th>ps_car_14</th>
      <th>ps_car_15</th>
      <th>ps_calc_01</th>
      <th>ps_calc_02</th>
      <th>ps_calc_03</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.610991</td>
      <td>0.439184</td>
      <td>0.551102</td>
      <td>0.379945</td>
      <td>0.813265</td>
      <td>0.276256</td>
      <td>3.065899</td>
      <td>0.449756</td>
      <td>0.449589</td>
      <td>0.449849</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.287643</td>
      <td>0.404264</td>
      <td>0.793506</td>
      <td>0.058327</td>
      <td>0.224588</td>
      <td>0.357154</td>
      <td>0.731366</td>
      <td>0.287198</td>
      <td>0.286893</td>
      <td>0.287153</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>0.250619</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.400000</td>
      <td>0.200000</td>
      <td>0.525000</td>
      <td>0.316228</td>
      <td>0.670867</td>
      <td>0.333167</td>
      <td>2.828427</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.700000</td>
      <td>0.300000</td>
      <td>0.720677</td>
      <td>0.374166</td>
      <td>0.765811</td>
      <td>0.368782</td>
      <td>3.316625</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.900000</td>
      <td>0.600000</td>
      <td>1.000000</td>
      <td>0.400000</td>
      <td>0.906190</td>
      <td>0.396485</td>
      <td>3.605551</td>
      <td>0.700000</td>
      <td>0.700000</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.900000</td>
      <td>1.800000</td>
      <td>4.037945</td>
      <td>1.264911</td>
      <td>3.720626</td>
      <td>0.636396</td>
      <td>3.741657</td>
      <td>0.900000</td>
      <td>0.900000</td>
      <td>0.900000</td>
    </tr>
  </tbody>
</table>
</div>



#### 1-1) reg variables
* ps_reg_03 변수에만 missing value가 있음
    * min값이 -1이면 missing value
* 변수간의 범위(최소,최대)가 서로 다르기때문에 스케일링(StandardScaler)을 사용해볼수있음.

#### 1-2) car variables
* ps_car_12 , ps_car_15에 missing value가있다.
* 마찬가지로, 범위가 다르기때문에 스케일링이 필요해보임

#### 1-3) calc variables
* missing value가 없음
* 위의 describe를 보면, 최대값이 0.9로 동일한것을 알수있음
* 세개의 _calc 로 끝나는 변수들의 분포가 유사해보임

> 전반적으로, interval 변수들의 범위가 크게 차이가 나지 않는것으로 보아, 아마도 로그변환 같은 데이터 변환이 적용된 데이터가아닐까 싶다

### 2) Ordinal variables


```python
v=meta[(meta.level=='ordinal') & (meta.keep)].index
train[v].describe()
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
      <th>ps_ind_01</th>
      <th>ps_ind_03</th>
      <th>ps_ind_14</th>
      <th>ps_ind_15</th>
      <th>ps_car_11</th>
      <th>ps_calc_04</th>
      <th>ps_calc_05</th>
      <th>ps_calc_06</th>
      <th>ps_calc_07</th>
      <th>ps_calc_08</th>
      <th>ps_calc_09</th>
      <th>ps_calc_10</th>
      <th>ps_calc_11</th>
      <th>ps_calc_12</th>
      <th>ps_calc_13</th>
      <th>ps_calc_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.900378</td>
      <td>4.423318</td>
      <td>0.012451</td>
      <td>7.299922</td>
      <td>2.346072</td>
      <td>2.372081</td>
      <td>1.885886</td>
      <td>7.689445</td>
      <td>3.005823</td>
      <td>9.225904</td>
      <td>2.339034</td>
      <td>8.433590</td>
      <td>5.441382</td>
      <td>1.441918</td>
      <td>2.872288</td>
      <td>7.539026</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.983789</td>
      <td>2.699902</td>
      <td>0.127545</td>
      <td>3.546042</td>
      <td>0.832548</td>
      <td>1.117219</td>
      <td>1.134927</td>
      <td>1.334312</td>
      <td>1.414564</td>
      <td>1.459672</td>
      <td>1.246949</td>
      <td>2.904597</td>
      <td>2.332871</td>
      <td>1.202963</td>
      <td>1.694887</td>
      <td>2.746652</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>10.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>11.000000</td>
      <td>4.000000</td>
      <td>13.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>25.000000</td>
      <td>19.000000</td>
      <td>10.000000</td>
      <td>13.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>



* ps_car_11에만 missing value가 있음
* 범위가 서로 다른것에 대해서는 스케일링을 적용해볼수있음

### 3) Binary variables


```python
v=meta[(meta.level=='binary')&(meta.keep)].index
train[v].describe()
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
      <th>target</th>
      <th>ps_ind_06_bin</th>
      <th>ps_ind_07_bin</th>
      <th>ps_ind_08_bin</th>
      <th>ps_ind_09_bin</th>
      <th>ps_ind_10_bin</th>
      <th>ps_ind_11_bin</th>
      <th>ps_ind_12_bin</th>
      <th>ps_ind_13_bin</th>
      <th>ps_ind_16_bin</th>
      <th>ps_ind_17_bin</th>
      <th>ps_ind_18_bin</th>
      <th>ps_calc_15_bin</th>
      <th>ps_calc_16_bin</th>
      <th>ps_calc_17_bin</th>
      <th>ps_calc_18_bin</th>
      <th>ps_calc_19_bin</th>
      <th>ps_calc_20_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
      <td>595212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.036448</td>
      <td>0.393742</td>
      <td>0.257033</td>
      <td>0.163921</td>
      <td>0.185304</td>
      <td>0.000373</td>
      <td>0.001692</td>
      <td>0.009439</td>
      <td>0.000948</td>
      <td>0.660823</td>
      <td>0.121081</td>
      <td>0.153446</td>
      <td>0.122427</td>
      <td>0.627840</td>
      <td>0.554182</td>
      <td>0.287182</td>
      <td>0.349024</td>
      <td>0.153318</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.187401</td>
      <td>0.488579</td>
      <td>0.436998</td>
      <td>0.370205</td>
      <td>0.388544</td>
      <td>0.019309</td>
      <td>0.041097</td>
      <td>0.096693</td>
      <td>0.030768</td>
      <td>0.473430</td>
      <td>0.326222</td>
      <td>0.360417</td>
      <td>0.327779</td>
      <td>0.483381</td>
      <td>0.497056</td>
      <td>0.452447</td>
      <td>0.476662</td>
      <td>0.360295</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




* 평균값을 통해 대부분의 변수에서 값이 0인것을 알수있다. imbalanced한 데이터

------------------------------------------
## Handling imbalanced classes
target=1 인것의 비율이 target=0보다 매우 적다. 이로인해 accuracy가 좋을순있다.<br>
이문제를 해결하기위해서 아래 두가지 전략이 있다.
* target=1 인 record를 oversampling한다
* target=0 인 record를 undersampling한다

> 우리의 경우 training set 데이터수가 많기때문에 undersampling하는 방법으로 가려고한다.


```python
desired_apriori=0.1

# Get the indices per target value
idx_0=train[train.target==0].index   # target=0인 값의 index
idx_1=train[train.target==1].index   # target=1인 값의 index

# target값(0,1) 에 따른 행 개수
nb_0=len(train.loc[idx_0])
nb_1=len(train.loc[idx_1])

# target=1은 0.9, target=0은 0.1
## ((1-0.1) * target=1인 데이터개수 ) / (0.1 * target=0인 데이터개수) 
## Undersampling => 0.34
undersampling_rate=((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)

# 언더샘플링한 target=0인 데이터개수 = undersampling비율 x target=0인 데이터개수
## 기존 573518 -> 195246
undersampled_nb_0=int(undersampling_rate*nb_0)
print('target=0에 대한 undersampling비율 : {}'.format(undersampling_rate))
print('undersampling이후 target=0 데이터 개수: {}'.format(undersampled_nb_0))

# target=0인 전체index 중에서 언더샘플링할 개수 지정해서 랜덤으로 index선택
undersampled_idx=shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# undersampling된 index의 리스트와 기존 target=1인 index합친 리스트만들기
idx_list=list(undersampled_idx)+list(idx_1)

# undersampling된 데이터 반환
train=train.loc[idx_list].reset_index(drop=True)
```

    target=0에 대한 undersampling비율 : 0.34043569687437886
    undersampling이후 target=0 데이터 개수: 195246


--------------------------------------------
## Data Quality Checks

Missing values를 체크해보자 (-1로 되어있음)


```python
vars_with_missing=[]

for f in train.columns:
    # 결측치 개수
    missings=train[train[f]==-1][f].count()
    if missings >0:
        vars_with_missing.append(f)
        # 결측치 비중
        missing_perc=missings/train.shape[0]
        
        print('Variable {} has {}records({:.2%}) with missing values'.format(f,missings,missing_perc))
        
print('In total, there are {}variables with missing values'.format(vars_with_missing))

```

    Variable ps_ind_02_cat has 103records(0.05%) with missing values
    Variable ps_ind_04_cat has 51records(0.02%) with missing values
    Variable ps_ind_05_cat has 2256records(1.04%) with missing values
    Variable ps_reg_03 has 38580records(17.78%) with missing values
    Variable ps_car_01_cat has 62records(0.03%) with missing values
    Variable ps_car_02_cat has 2records(0.00%) with missing values
    Variable ps_car_03_cat has 148367records(68.39%) with missing values
    Variable ps_car_05_cat has 96026records(44.26%) with missing values
    Variable ps_car_07_cat has 4431records(2.04%) with missing values
    Variable ps_car_09_cat has 230records(0.11%) with missing values
    Variable ps_car_11 has 1records(0.00%) with missing values
    Variable ps_car_14 has 15726records(7.25%) with missing values
    In total, there are ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_car_07_cat', 'ps_car_09_cat', 'ps_car_11', 'ps_car_14']variables with missing values


**결측치가 너무많은 변수는 제거하고, 일부이면 평균 또는 최빈값으로 대치**

* ps_car_03_cat 변수와 ps_car_05_cat 변수는 결측치가 각각 68%, 44%로 너무 많으므로 제거
* 나머지 categorical 변수들의 결측치는 유지
* ps_reg_03(continuous) 변수는 18%가 결측치이므로, 평균으로 대치
* ps_car_11(ordinal) 변수는 1개의 record만 결측이므로 최빈값으로 대치
* ps_car_14(continuous) 변수는 7%가 결측치이므로 평균값으로 대치


```python
# 결측치가 너무 많은 컬럼은 drop
vars_to_drop=['ps_car_03_cat','ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)

# 저장한 meta데이터에 drop한 컬럼의 상태 False로 바꾸기
meta.loc[(vars_to_drop),'keep']=False
```


```python
# 평균 또는 최빈값으로 대치
# missing_values의 default값은 np.nan인데 여기서는 -1이 결측이므로 지정해줘야함.
mean_imp=SimpleImputer(missing_values=-1, strategy='mean')
mode_imp=SimpleImputer(missing_values=-1, strategy='most_frequent')

# 다차원 배열(array)을 1차원 배열로 평평하게 펴주는 NumPy의 ravel() 함수
## ex) array([[0.83],[0.72]]) -> array([0.83],[0.72])
train['ps_reg_03']=mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_14']=mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11']=mode_imp.fit_transform(train[['ps_car_11']]).ravel()
```

### Categorical변수의 cardinality(집합의 크기) 체크
* Cardinality는 변수안에 서로다른 unique한 값의 개수를 말하는데, categorical변수에 많은 distinct value가 있으면 더미변수 개수가 너무 많아질수있음. 따라서 이런 변수들을 다뤄보려고한다.


```python
v=meta[(meta.level=='nominal')&(meta.keep)].index
for f in v:
    dist_values=train[f].value_counts().shape[0]
    print('Variable {} has {}distinct values'.format(f,dist_values))
```

    Variable ps_ind_02_cat has 5distinct values
    Variable ps_ind_04_cat has 3distinct values
    Variable ps_ind_05_cat has 8distinct values
    Variable ps_car_01_cat has 13distinct values
    Variable ps_car_02_cat has 3distinct values
    Variable ps_car_04_cat has 10distinct values
    Variable ps_car_06_cat has 18distinct values
    Variable ps_car_07_cat has 3distinct values
    Variable ps_car_08_cat has 2distinct values
    Variable ps_car_09_cat has 6distinct values
    Variable ps_car_10_cat has 3distinct values
    Variable ps_car_11_cat has 104distinct values


* ps_car_11_cat 변수만이 distinct value개수가 많다.

**아래는 distinct한 값이 너무 많은 카테고리변수 ps_car_11_cat를 처리하는 방법 (잘이해안감..)**


```python
# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
```


```python
train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target=train.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)
```

----------
## Exploratory Data Visualization

#### 1) Categorical variables
Categorical 변수를 시각화해보고 target=1인 고객의 비율을 살펴보자


```python
train[['ps_ind_02_cat','target']].groupby(['ps_ind_02_cat'],as_index=False).mean()
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
      <th>ps_ind_02_cat</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>0.388350</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.098190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.104102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.101746</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.112349</td>
    </tr>
  </tbody>
</table>
</div>




```python
v=meta[(meta.level=='nominal')&(meta.keep)].index
for f in v:
    plt.figure()
    fig,ax=plt.subplots(figsize=(20,10))
    # target=1인 값의 퍼센테이지를 구해보자
    cat_perc=train[[f,'target']].groupby([f],as_index=False).mean()
    cat_perc.sort_values(by='target',ascending=False, inplace=True)
    # 막대그래프
    sns.barplot(ax=ax, x=f,y='target',data=cat_perc,order=cat_perc[f])
    plt.ylabel('% target',fontsize=18)
    plt.xlabel(f, fontsize=18)
    plt.tick_params(axis='both',which='major',labelsize=18)
    plt.show();
```


    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_1.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_3.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_5.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_7.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_9.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_11.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_13.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_15.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_17.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_19.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_52_21.png)


결측치를 최빈값으로 대치하지 않고, 카테고리 값으로 분리해서 살펴본것은 바람직했던 것으로 보인다. 결측치가 있는 고객들은 보험청구를 요구했을 가능성이 더 높은것으로 나타났다. (어떤경우는 낮기도함)

#### 2) Interval variables
히트맵으로 interval 변수간의 상관관계를 살펴보자. 


```python
def corr_heatmap(v):
    correlations=train[v].corr()
    
    cmap=sns.diverging_palette(220,10,as_cmap=True)
    
    fig,ax=plt.subplots(figsize=(10,10))
    sns.heatmap(correlations,cmap=cmap,vmax=1.0, center=0, fmt='.2f',square=True, \
               linewidths=.5, annot=True, cbar_kws={'shrink':.75})
    plt.show();
    
v=meta[(meta.level=='interval')&(meta.keep)].index
corr_heatmap(v)
```


![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_55_0.png)


변수간에 상관관계가 높은 변수들이 있는것을 확인할 수 있다.
* ps_reg_02 and ps_reg_03 (0.7)
* ps_car_12 and ps_car13 (0.67)
* ps_car_12 and ps_car14 (0.58)
* ps_car_13 and ps_car15 (0.67)


```python
# train데이터를 샘플링해서 시각화 살펴보기
s=train.sample(frac=0.1)
```

**위에서 상관관계가 높은 변수들을 lmplot으로 시각화해본다**


```python
# 회귀모델 판단할때, 상관관계 볼때 lmplot 사용
sns.lmplot(x='ps_reg_02',y='ps_reg_03',data=s, hue='target',palette='Set1',
          scatter_kws={'alpha':0.3})
plt.show()
```


![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_59_0.png)



```python
sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
```


![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_60_0.png)



```python
sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
```


![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_61_0.png)



```python
sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
```


![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_62_0.png)


* 서로 상관이있는 변수를 어떻게 결정하고 남겨둘것인가? 우리는 차원을 줄이기위해 변수에 Principal Component Analysis (PCA) 를 적용해볼수있다. 그러나, 상관관계가 있는 변수들의 수가 적기때문에 모델이 알아서 하게 두기로한다.

PCA 참고 노트북 <br>
https://www.kaggle.com/bertcarremans/reducing-number-of-numerical-features-with-pca

(참고) 아래는 상관관계가 서로 낮은 변수를 lmplot으로 시각화해서 상관관계가 있는 위의 시각화 결과와 비교


```python
# ps_car_14 와 ps_reg_03 간의 상관관계는 0.08
sns.lmplot(x='ps_car_14', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
```


![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_65_0.png)


#### 3) ordinal 변수간의 상관관계 확인


```python
v=meta[(meta.level=='ordinal')&(meta.keep)].index
corr_heatmap(v)
```


![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_67_0.png)


* ordinal변수에서는 상관관계가 큰 변수들은 보이지 않는다. (target 0,1에 따라 그룹핑했을때의 분포는 확인해볼 필요 있음)


```python
for f in v:
    plt.figure()
    fig,ax=plt.subplots(figsize=(20,10))
    ordinal_perc=train[[f,'target']].groupby([f],as_index=False).mean()
    ordinal_perc.sort_values(by='target',ascending=False, inplace=True)
    
    sns.barplot(ax=ax, x=f, y='target',data=ordinal_perc, order=ordinal_perc[f])
    plt.ylabel('% target',fontsize=18)
    plt.xlabel(f,fontsize=18)
    plt.tick_params(axis='both',which='major',labelsize=18)
    plt.show();
```


    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_1.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_3.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_5.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_7.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_9.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_11.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_13.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_15.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_17.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_19.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_21.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_23.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_25.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_27.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_29.png)



    <Figure size 432x288 with 0 Axes>



![png](/images/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_files/kaggle-PortoSeguro%27s-safe-driver-prediction%28beginner%29_69_31.png)


* ordinal 변수의 target별 평균을 살펴보니 ps_calc_13 변수처럼 특정 값일때 target=1 평균값이 유난히 큰 특징이 있었음

----------------------------------------
## Feature engineering

#### 1) 더미변수 만들기
카테고리변수는 어떤 순서나 단위를 나타내지 않는다. 예를들어 카테고리 2가 카테고리 1의 두배라고 할 수 없다. 


```python
v=meta[(meta.level=='nominal') & (meta.keep)].index
print('더미화 하기전에 train데이터에 {}개의 변수가 있다'.format(train.shape[1]))
train=pd.get_dummies(train, columns=v, drop_first=True)
print('더미화 한 후에 train데이터에 {}개의 변수가 있다'.format(train.shape[1]))
```

    더미화 하기전에 train데이터에 57개의 변수가 있다
    더미화 한 후에 train데이터에 109개의 변수가 있다


* 따라서, 더미변수를 만듬으로써 52개의 변수가 training set에 추가되었다

#### 2) interaction 변수 만들기 (다항회귀)
https://chana.tistory.com/entry/핸즈온-머신러닝5-다항-회귀

* 사이킷런의 PolynomialFeatures를 사용하여 훈련 세트에 있는 각 특성을 제곱하여 새로운 특성으로 추가한 훈련 데이터를 만들어보자 <br>

> Q : 2차항 변수를 만드는 이유는 ? 어떤경우에 ? <br>
> A : 데이터들간의 형태가 비선형일때 데이터에 각 특성의 제곱을 추가해서 특성이 추가된 비선형 데이터를 선형회귀 모델로 훈련시키는 방법
    


```python
v=meta[(meta.level=='interval')&(meta.keep)].index
# 2차항 변수 만들기
poly=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)

# interactions변수에는 기존 데이터와 제곱한 데이터가 모두 포함된다.
interactions=pd.DataFrame(data=poly.fit_transform(train[v]),columns=poly.get_feature_names(v))
# 기존 컬럼은 제거
interactions.drop(v,axis=1,inplace=True)

# train데이터에 interaction변수를 concat한다. 즉, 기존 변수에 interval변수의 2차항 변수들을 추가
print('interaction을 추가하기 전에는 train데이터에 {} 개의 변수가 있음'.format(train.shape[1]))
train=pd.concat([train,interactions],axis=1)
print('interaction 추가한 후에 train데이터에 {}개의 변수가 있음'.format(train.shape[1]))
```

    interaction을 추가하기 전에는 train데이터에 109 개의 변수가 있음
    interaction 추가한 후에 train데이터에 164개의 변수가 있음



```python
interactions.head()
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
      <th>ps_reg_01^2</th>
      <th>ps_reg_01 ps_reg_02</th>
      <th>ps_reg_01 ps_reg_03</th>
      <th>ps_reg_01 ps_car_12</th>
      <th>ps_reg_01 ps_car_13</th>
      <th>ps_reg_01 ps_car_14</th>
      <th>ps_reg_01 ps_car_15</th>
      <th>ps_reg_01 ps_calc_01</th>
      <th>ps_reg_01 ps_calc_02</th>
      <th>ps_reg_01 ps_calc_03</th>
      <th>ps_reg_02^2</th>
      <th>ps_reg_02 ps_reg_03</th>
      <th>ps_reg_02 ps_car_12</th>
      <th>ps_reg_02 ps_car_13</th>
      <th>ps_reg_02 ps_car_14</th>
      <th>ps_reg_02 ps_car_15</th>
      <th>ps_reg_02 ps_calc_01</th>
      <th>ps_reg_02 ps_calc_02</th>
      <th>ps_reg_02 ps_calc_03</th>
      <th>ps_reg_03^2</th>
      <th>ps_reg_03 ps_car_12</th>
      <th>ps_reg_03 ps_car_13</th>
      <th>ps_reg_03 ps_car_14</th>
      <th>ps_reg_03 ps_car_15</th>
      <th>ps_reg_03 ps_calc_01</th>
      <th>ps_reg_03 ps_calc_02</th>
      <th>ps_reg_03 ps_calc_03</th>
      <th>ps_car_12^2</th>
      <th>ps_car_12 ps_car_13</th>
      <th>ps_car_12 ps_car_14</th>
      <th>ps_car_12 ps_car_15</th>
      <th>ps_car_12 ps_calc_01</th>
      <th>ps_car_12 ps_calc_02</th>
      <th>ps_car_12 ps_calc_03</th>
      <th>ps_car_13^2</th>
      <th>ps_car_13 ps_car_14</th>
      <th>ps_car_13 ps_car_15</th>
      <th>ps_car_13 ps_calc_01</th>
      <th>ps_car_13 ps_calc_02</th>
      <th>ps_car_13 ps_calc_03</th>
      <th>ps_car_14^2</th>
      <th>ps_car_14 ps_car_15</th>
      <th>ps_car_14 ps_calc_01</th>
      <th>ps_car_14 ps_calc_02</th>
      <th>ps_car_14 ps_calc_03</th>
      <th>ps_car_15^2</th>
      <th>ps_car_15 ps_calc_01</th>
      <th>ps_car_15 ps_calc_02</th>
      <th>ps_car_15 ps_calc_03</th>
      <th>ps_calc_01^2</th>
      <th>ps_calc_01 ps_calc_02</th>
      <th>ps_calc_01 ps_calc_03</th>
      <th>ps_calc_02^2</th>
      <th>ps_calc_02 ps_calc_03</th>
      <th>ps_calc_03^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.36</td>
      <td>0.36</td>
      <td>0.502892</td>
      <td>0.221269</td>
      <td>0.324362</td>
      <td>0.207413</td>
      <td>1.200000</td>
      <td>0.54</td>
      <td>0.18</td>
      <td>0.00</td>
      <td>0.36</td>
      <td>0.502892</td>
      <td>0.221269</td>
      <td>0.324362</td>
      <td>0.207413</td>
      <td>1.200000</td>
      <td>0.54</td>
      <td>0.18</td>
      <td>0.00</td>
      <td>0.702500</td>
      <td>0.309095</td>
      <td>0.453108</td>
      <td>0.289739</td>
      <td>1.676305</td>
      <td>0.754337</td>
      <td>0.251446</td>
      <td>0.000000</td>
      <td>0.136</td>
      <td>0.199365</td>
      <td>0.127483</td>
      <td>0.737564</td>
      <td>0.331904</td>
      <td>0.110635</td>
      <td>0.000000</td>
      <td>0.292252</td>
      <td>0.186880</td>
      <td>1.081207</td>
      <td>0.486543</td>
      <td>0.162181</td>
      <td>0.000000</td>
      <td>0.1195</td>
      <td>0.691375</td>
      <td>0.311119</td>
      <td>0.103706</td>
      <td>0.000000</td>
      <td>4.0</td>
      <td>1.800000</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>0.81</td>
      <td>0.27</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.81</td>
      <td>0.54</td>
      <td>0.655596</td>
      <td>0.381838</td>
      <td>0.344658</td>
      <td>0.340933</td>
      <td>0.000000</td>
      <td>0.18</td>
      <td>0.00</td>
      <td>0.54</td>
      <td>0.36</td>
      <td>0.437064</td>
      <td>0.254558</td>
      <td>0.229772</td>
      <td>0.227288</td>
      <td>0.000000</td>
      <td>0.12</td>
      <td>0.00</td>
      <td>0.36</td>
      <td>0.530625</td>
      <td>0.309051</td>
      <td>0.278958</td>
      <td>0.275943</td>
      <td>0.000000</td>
      <td>0.145688</td>
      <td>0.000000</td>
      <td>0.437064</td>
      <td>0.180</td>
      <td>0.162473</td>
      <td>0.160717</td>
      <td>0.000000</td>
      <td>0.084853</td>
      <td>0.000000</td>
      <td>0.254558</td>
      <td>0.146653</td>
      <td>0.145068</td>
      <td>0.000000</td>
      <td>0.076591</td>
      <td>0.000000</td>
      <td>0.229772</td>
      <td>0.1435</td>
      <td>0.000000</td>
      <td>0.075763</td>
      <td>0.000000</td>
      <td>0.227288</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.81</td>
      <td>0.54</td>
      <td>0.782340</td>
      <td>0.360000</td>
      <td>0.732844</td>
      <td>0.362131</td>
      <td>2.984962</td>
      <td>0.27</td>
      <td>0.81</td>
      <td>0.09</td>
      <td>0.36</td>
      <td>0.521560</td>
      <td>0.240000</td>
      <td>0.488563</td>
      <td>0.241421</td>
      <td>1.989975</td>
      <td>0.18</td>
      <td>0.54</td>
      <td>0.06</td>
      <td>0.755625</td>
      <td>0.347707</td>
      <td>0.707819</td>
      <td>0.349765</td>
      <td>2.883032</td>
      <td>0.260780</td>
      <td>0.782340</td>
      <td>0.086927</td>
      <td>0.160</td>
      <td>0.325708</td>
      <td>0.160947</td>
      <td>1.326650</td>
      <td>0.120000</td>
      <td>0.360000</td>
      <td>0.040000</td>
      <td>0.663037</td>
      <td>0.327637</td>
      <td>2.700631</td>
      <td>0.244281</td>
      <td>0.732844</td>
      <td>0.081427</td>
      <td>0.1619</td>
      <td>1.334504</td>
      <td>0.120710</td>
      <td>0.362131</td>
      <td>0.040237</td>
      <td>11.0</td>
      <td>0.994987</td>
      <td>2.984962</td>
      <td>0.331662</td>
      <td>0.09</td>
      <td>0.27</td>
      <td>0.03</td>
      <td>0.81</td>
      <td>0.09</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.36</td>
      <td>0.90</td>
      <td>1.023523</td>
      <td>0.240000</td>
      <td>0.503032</td>
      <td>0.227051</td>
      <td>2.163331</td>
      <td>0.48</td>
      <td>0.24</td>
      <td>0.06</td>
      <td>2.25</td>
      <td>2.558808</td>
      <td>0.600000</td>
      <td>1.257580</td>
      <td>0.567627</td>
      <td>5.408327</td>
      <td>1.20</td>
      <td>0.60</td>
      <td>0.15</td>
      <td>2.910000</td>
      <td>0.682349</td>
      <td>1.430181</td>
      <td>0.645532</td>
      <td>6.150610</td>
      <td>1.364698</td>
      <td>0.682349</td>
      <td>0.170587</td>
      <td>0.160</td>
      <td>0.335355</td>
      <td>0.151367</td>
      <td>1.442221</td>
      <td>0.320000</td>
      <td>0.160000</td>
      <td>0.040000</td>
      <td>0.702893</td>
      <td>0.317260</td>
      <td>3.022847</td>
      <td>0.670710</td>
      <td>0.335355</td>
      <td>0.083839</td>
      <td>0.1432</td>
      <td>1.364405</td>
      <td>0.302734</td>
      <td>0.151367</td>
      <td>0.037842</td>
      <td>13.0</td>
      <td>2.884441</td>
      <td>1.442221</td>
      <td>0.360555</td>
      <td>0.64</td>
      <td>0.32</td>
      <td>0.08</td>
      <td>0.16</td>
      <td>0.04</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.64</td>
      <td>0.64</td>
      <td>0.869253</td>
      <td>0.320000</td>
      <td>0.724622</td>
      <td>0.307870</td>
      <td>2.884441</td>
      <td>0.48</td>
      <td>0.40</td>
      <td>0.72</td>
      <td>0.64</td>
      <td>0.869253</td>
      <td>0.320000</td>
      <td>0.724622</td>
      <td>0.307870</td>
      <td>2.884441</td>
      <td>0.48</td>
      <td>0.40</td>
      <td>0.72</td>
      <td>1.180625</td>
      <td>0.434626</td>
      <td>0.984186</td>
      <td>0.418151</td>
      <td>3.917668</td>
      <td>0.651939</td>
      <td>0.543283</td>
      <td>0.977909</td>
      <td>0.160</td>
      <td>0.362311</td>
      <td>0.153935</td>
      <td>1.442221</td>
      <td>0.240000</td>
      <td>0.200000</td>
      <td>0.360000</td>
      <td>0.820432</td>
      <td>0.348577</td>
      <td>3.265825</td>
      <td>0.543466</td>
      <td>0.452888</td>
      <td>0.815199</td>
      <td>0.1481</td>
      <td>1.387552</td>
      <td>0.230903</td>
      <td>0.192419</td>
      <td>0.346354</td>
      <td>13.0</td>
      <td>2.163331</td>
      <td>1.802776</td>
      <td>3.244996</td>
      <td>0.36</td>
      <td>0.30</td>
      <td>0.54</td>
      <td>0.25</td>
      <td>0.45</td>
      <td>0.81</td>
    </tr>
  </tbody>
</table>
</div>



-----------------
## Feature Selection
* 분산이 0이거나 작은 feature제거
* 사이킷런의 VarianceThreshold를 이용하면 간편하게 분산이 0인 feature를 제거할 수 있다. 하지만, 이 경우에는 분산이 0인 변수가 없음. 따라서 분산이 1% 미만인 변수를 제거한다고하면 31개의 변수를 제거할 수 있음.

> Q : 분산이 작은 변수를 제거해야하는 이유 ? <br>

> A : 예측모델에서 중요한 특성이란, 타겟과의 상관관계가 큰 특성을 의미한다. 그런데 상관관계에 앞서 어떤 특성의 값 자체가 표본에 따라 그다지 변하지 않는다면, 예측에 별 도움이 되지 않을 가능성이 높다.  <br>
(ex. 남자를 상대로한 설문조사 데이터에서 남자 라는 성별특성은 무의미함.)<br>
따라서, 표본 변화에 따른 데이터 값의 변화량 즉, 분산이 기준치보다낮은 특성은 제거하는것


```python
selector=VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id','target'],axis=1))

# np.vectorize(사용자 정의 함수명, otypes = ...)
## Function to toggle boolean array elements
## get_support() 하면 True,False값이 나오는데 이중에서 False값만 반환하기위해 f함수지정
f=np.vectorize(lambda x: not x)

v=train.drop(['id','target'],axis=1).columns[f(selector.get_support())]
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))
```

    28 variables have too low variance.
    These variables are ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_12', 'ps_car_14', 'ps_car_11_cat_te', 'ps_ind_05_cat_2', 'ps_ind_05_cat_5', 'ps_car_01_cat_1', 'ps_car_01_cat_2', 'ps_car_04_cat_3', 'ps_car_04_cat_4', 'ps_car_04_cat_5', 'ps_car_04_cat_6', 'ps_car_04_cat_7', 'ps_car_06_cat_2', 'ps_car_06_cat_5', 'ps_car_06_cat_8', 'ps_car_06_cat_12', 'ps_car_06_cat_16', 'ps_car_06_cat_17', 'ps_car_09_cat_4', 'ps_car_10_cat_1', 'ps_car_10_cat_2', 'ps_car_12^2', 'ps_car_12 ps_car_14', 'ps_car_14^2']


### Selecting features with a Random Forest and SelectFromModel

https://blog.naver.com/PostView.nhn?blogId=bosongmoon&logNo=221807565642


```python
X_train=train.drop(['id','target'],axis=1)
y_train=train['target']

feat_labels=X_train.columns

rf=RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

rf.fit(X_train,y_train)
importances=rf.feature_importances_

# np.argsort() : 작은 것 부터 순서대로 뽑아내는 index
# [::-1] 다시 역순으로
indices=np.argsort(rf.feature_importances_)[::-1]   # 중요도 큰순으로 나열

# 순서, 30으로 나누기, 인덱스와 중요도 출력
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]], importances[indices[f]]))

```

     1) ps_car_11_cat_te               0.021159
     2) ps_car_13^2                    0.017303
     3) ps_car_13                      0.017301
     4) ps_car_12 ps_car_13            0.017235
     5) ps_car_13 ps_car_14            0.017129
     6) ps_reg_03 ps_car_13            0.017123
     7) ps_car_13 ps_car_15            0.016827
     8) ps_reg_01 ps_car_13            0.016823
     9) ps_reg_03 ps_car_14            0.016259
    10) ps_reg_03 ps_car_12            0.015592
    11) ps_reg_03 ps_car_15            0.015163
    12) ps_car_14 ps_car_15            0.015020
    13) ps_car_13 ps_calc_01           0.014759
    14) ps_car_13 ps_calc_03           0.014683
    15) ps_car_13 ps_calc_02           0.014658
    16) ps_reg_01 ps_reg_03            0.014643
    17) ps_reg_02 ps_car_13            0.014634
    18) ps_reg_01 ps_car_14            0.014362
    19) ps_reg_03^2                    0.014268
    20) ps_reg_03                      0.014229
    21) ps_reg_03 ps_calc_02           0.013812
    22) ps_reg_03 ps_calc_03           0.013765
    23) ps_reg_03 ps_calc_01           0.013726
    24) ps_car_14 ps_calc_02           0.013681
    25) ps_calc_10                     0.013630
    26) ps_car_14 ps_calc_01           0.013526
    27) ps_car_14 ps_calc_03           0.013501
    28) ps_calc_14                     0.013366
    29) ps_car_12 ps_car_14            0.012949
    30) ps_ind_03                      0.012886
    31) ps_car_14                      0.012788
    32) ps_car_14^2                    0.012761
    33) ps_reg_02 ps_car_14            0.012729
    34) ps_calc_11                     0.012661
    35) ps_reg_02 ps_reg_03            0.012514
    36) ps_ind_15                      0.012162
    37) ps_car_15 ps_calc_02           0.010903
    38) ps_car_15 ps_calc_03           0.010890
    39) ps_car_12 ps_car_15            0.010860
    40) ps_car_15 ps_calc_01           0.010850
    41) ps_car_12 ps_calc_01           0.010496
    42) ps_calc_13                     0.010482
    43) ps_car_12 ps_calc_02           0.010325
    44) ps_car_12 ps_calc_03           0.010314
    45) ps_reg_02 ps_car_15            0.010234
    46) ps_reg_01 ps_car_15            0.010154
    47) ps_calc_02 ps_calc_03          0.010094
    48) ps_calc_01 ps_calc_02          0.010016
    49) ps_calc_01 ps_calc_03          0.009951
    50) ps_calc_07                     0.009820
    51) ps_calc_08                     0.009792
    52) ps_reg_01 ps_car_12            0.009452
    53) ps_reg_02 ps_car_12            0.009277
    54) ps_reg_02 ps_calc_01           0.009203
    55) ps_reg_02 ps_calc_03           0.009179
    56) ps_reg_02 ps_calc_02           0.009153
    57) ps_reg_01 ps_calc_03           0.009089
    58) ps_calc_06                     0.009035
    59) ps_reg_01 ps_calc_01           0.009023
    60) ps_reg_01 ps_calc_02           0.009005
    61) ps_calc_09                     0.008775
    62) ps_ind_01                      0.008613
    63) ps_calc_05                     0.008315
    64) ps_calc_04                     0.008138
    65) ps_calc_12                     0.008030
    66) ps_reg_01 ps_reg_02            0.008006
    67) ps_car_15^2                    0.006175
    68) ps_car_15                      0.006167
    69) ps_calc_03                     0.005970
    70) ps_calc_01                     0.005968
    71) ps_calc_03^2                   0.005962
    72) ps_calc_01^2                   0.005958
    73) ps_calc_02                     0.005932
    74) ps_calc_02^2                   0.005919
    75) ps_car_12                      0.005427
    76) ps_car_12^2                    0.005363
    77) ps_reg_02                      0.004977
    78) ps_reg_02^2                    0.004974
    79) ps_reg_01                      0.004142
    80) ps_reg_01^2                    0.004134
    81) ps_car_11                      0.003788
    82) ps_ind_05_cat_0                0.003580
    83) ps_ind_17_bin                  0.002838
    84) ps_calc_17_bin                 0.002703
    85) ps_calc_16_bin                 0.002624
    86) ps_calc_19_bin                 0.002570
    87) ps_calc_18_bin                 0.002496
    88) ps_ind_04_cat_0                0.002403
    89) ps_ind_16_bin                  0.002394
    90) ps_car_01_cat_11               0.002383
    91) ps_ind_04_cat_1                0.002369
    92) ps_ind_07_bin                  0.002324
    93) ps_car_09_cat_2                0.002320
    94) ps_ind_02_cat_1                0.002255
    95) ps_car_01_cat_7                0.002116
    96) ps_car_09_cat_0                0.002111
    97) ps_ind_02_cat_2                0.002093
    98) ps_calc_20_bin                 0.002076
    99) ps_ind_06_bin                  0.002065
    100) ps_car_06_cat_1                0.002005
    101) ps_calc_15_bin                 0.001986
    102) ps_car_07_cat_1                0.001966
    103) ps_ind_08_bin                  0.001928
    104) ps_car_06_cat_11               0.001817
    105) ps_car_09_cat_1                0.001811
    106) ps_ind_18_bin                  0.001735
    107) ps_ind_09_bin                  0.001724
    108) ps_car_01_cat_9                0.001606
    109) ps_car_01_cat_10               0.001598
    110) ps_car_06_cat_14               0.001580
    111) ps_car_01_cat_6                0.001550
    112) ps_car_01_cat_4                0.001517
    113) ps_ind_05_cat_6                0.001512
    114) ps_ind_02_cat_3                0.001413
    115) ps_car_07_cat_0                0.001372
    116) ps_car_08_cat_1                0.001342
    117) ps_car_02_cat_1                0.001341
    118) ps_car_01_cat_8                0.001337
    119) ps_car_02_cat_0                0.001308
    120) ps_ind_05_cat_4                0.001227
    121) ps_car_06_cat_4                0.001219
    122) ps_ind_02_cat_4                0.001152
    123) ps_car_01_cat_5                0.001150
    124) ps_car_06_cat_6                0.001096
    125) ps_car_06_cat_10               0.001062
    126) ps_car_04_cat_1                0.001038
    127) ps_ind_05_cat_2                0.001023
    128) ps_car_06_cat_7                0.001018
    129) ps_car_04_cat_2                0.000986
    130) ps_car_01_cat_3                0.000910
    131) ps_car_09_cat_3                0.000878
    132) ps_car_01_cat_0                0.000858
    133) ps_ind_14                      0.000857
    134) ps_car_06_cat_15               0.000832
    135) ps_car_06_cat_9                0.000785
    136) ps_ind_05_cat_1                0.000754
    137) ps_car_10_cat_1                0.000713
    138) ps_car_06_cat_3                0.000707
    139) ps_ind_12_bin                  0.000687
    140) ps_ind_05_cat_3                0.000662
    141) ps_car_09_cat_4                0.000622
    142) ps_car_01_cat_2                0.000558
    143) ps_car_04_cat_8                0.000547
    144) ps_car_06_cat_17               0.000513
    145) ps_car_06_cat_16               0.000481
    146) ps_car_04_cat_9                0.000446
    147) ps_car_06_cat_12               0.000428
    148) ps_car_06_cat_13               0.000389
    149) ps_car_01_cat_1                0.000389
    150) ps_ind_05_cat_5                0.000314
    151) ps_car_06_cat_5                0.000282
    152) ps_ind_11_bin                  0.000214
    153) ps_car_04_cat_6                0.000201
    154) ps_ind_13_bin                  0.000152
    155) ps_car_04_cat_3                0.000146
    156) ps_car_06_cat_2                0.000144
    157) ps_car_04_cat_5                0.000093
    158) ps_car_06_cat_8                0.000091
    159) ps_car_04_cat_7                0.000080
    160) ps_ind_10_bin                  0.000072
    161) ps_car_10_cat_2                0.000059
    162) ps_car_04_cat_4                0.000041


#### SelectFromModel 
* 모델 훈련이 끝난 후 사용자가 지정한 임계값을 기반으로 특성 선택



```python
# rf모델 내 특성 중 지니계수 값이 mediand이상일 경우의 특징 선택
sfm=SelectFromModel(rf, threshold='median',prefit=True)
print('Number of features before selection: {}'.format(X_train.shape[1]))

# 학습시킨 sfm을 x_train에 적용
n_features=sfm.transform(X_train).shape[1]
print('Number of features after selection: {}'.format(n_features))

# sfm.get_support() 하면 선택된 변수일경우 True, 아니면 False
## 최종 선택 변수
selected_vars=list(feat_labels[sfm.get_support()])
```

    Number of features before selection: 162
    Number of features after selection: 81



```python
selected_vars
```




    ['ps_ind_01',
     'ps_ind_03',
     'ps_ind_15',
     'ps_reg_01',
     'ps_reg_02',
     'ps_reg_03',
     'ps_car_11',
     'ps_car_12',
     'ps_car_13',
     'ps_car_14',
     'ps_car_15',
     'ps_calc_01',
     'ps_calc_02',
     'ps_calc_03',
     'ps_calc_04',
     'ps_calc_05',
     'ps_calc_06',
     'ps_calc_07',
     'ps_calc_08',
     'ps_calc_09',
     'ps_calc_10',
     'ps_calc_11',
     'ps_calc_12',
     'ps_calc_13',
     'ps_calc_14',
     'ps_car_11_cat_te',
     'ps_reg_01^2',
     'ps_reg_01 ps_reg_02',
     'ps_reg_01 ps_reg_03',
     'ps_reg_01 ps_car_12',
     'ps_reg_01 ps_car_13',
     'ps_reg_01 ps_car_14',
     'ps_reg_01 ps_car_15',
     'ps_reg_01 ps_calc_01',
     'ps_reg_01 ps_calc_02',
     'ps_reg_01 ps_calc_03',
     'ps_reg_02^2',
     'ps_reg_02 ps_reg_03',
     'ps_reg_02 ps_car_12',
     'ps_reg_02 ps_car_13',
     'ps_reg_02 ps_car_14',
     'ps_reg_02 ps_car_15',
     'ps_reg_02 ps_calc_01',
     'ps_reg_02 ps_calc_02',
     'ps_reg_02 ps_calc_03',
     'ps_reg_03^2',
     'ps_reg_03 ps_car_12',
     'ps_reg_03 ps_car_13',
     'ps_reg_03 ps_car_14',
     'ps_reg_03 ps_car_15',
     'ps_reg_03 ps_calc_01',
     'ps_reg_03 ps_calc_02',
     'ps_reg_03 ps_calc_03',
     'ps_car_12^2',
     'ps_car_12 ps_car_13',
     'ps_car_12 ps_car_14',
     'ps_car_12 ps_car_15',
     'ps_car_12 ps_calc_01',
     'ps_car_12 ps_calc_02',
     'ps_car_12 ps_calc_03',
     'ps_car_13^2',
     'ps_car_13 ps_car_14',
     'ps_car_13 ps_car_15',
     'ps_car_13 ps_calc_01',
     'ps_car_13 ps_calc_02',
     'ps_car_13 ps_calc_03',
     'ps_car_14^2',
     'ps_car_14 ps_car_15',
     'ps_car_14 ps_calc_01',
     'ps_car_14 ps_calc_02',
     'ps_car_14 ps_calc_03',
     'ps_car_15^2',
     'ps_car_15 ps_calc_01',
     'ps_car_15 ps_calc_02',
     'ps_car_15 ps_calc_03',
     'ps_calc_01^2',
     'ps_calc_01 ps_calc_02',
     'ps_calc_01 ps_calc_03',
     'ps_calc_02^2',
     'ps_calc_02 ps_calc_03',
     'ps_calc_03^2']




```python
train = train[selected_vars + ['target']]
```

----------
## Feature scaling


```python
scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))
```




    array([[-0.45941104, -1.26665356,  1.05087653, ..., -0.72553616,
            -1.01071913, -1.06173767],
           [ 1.55538958,  0.95034274, -0.63847299, ..., -1.06120876,
            -1.01071913,  0.27907892],
           [ 1.05168943, -0.52765479, -0.92003125, ...,  1.95984463,
            -0.56215309, -1.02449277],
           ...,
           [-0.9631112 ,  0.58084336,  0.48776003, ..., -0.46445747,
             0.18545696,  0.27907892],
           [-0.9631112 , -0.89715418, -1.48314775, ..., -0.91202093,
            -0.41263108,  0.27907892],
           [-0.45941104, -1.26665356,  1.61399304, ...,  0.28148164,
            -0.11358706, -0.72653353]])



------------------------------------------------
# test-set 동일하게 전처리 후 submission파일 생성

### Data Preprocessing
* 결측치 처리 (평균, 최빈값 대치)
* distinct value가 너무 많은 변수 drop
* 더미변수 추가
* 2차항변수 추가
* train 데이터에서 다른 사람 코드로 encoded한 컬럼 ps_car_11_cat_te 은 우선 제거
* train-set의 rf, SelectFromModel 과정을 통해 주요변수 (selected_vars) 도출했고 test-set도 이 변수만 추출
* 데이터 스케일링 (StandardScaler)
    * train-set에는 fit_transform, test-set에는 transform만 적용



```python

```
