---
layout: single
title : "[그레인저 인과관계] Granger Causality를 활용한 선행관계 분석"
author_profile: true
read_time: false
comments: true
categories:
- CausalInference
---

<br>
<br>





# **01. Granger Causality 란 ,**

---

*“시간의 선행성”* 을 파악하는 방법론으로, 주로 경제학에서 사용함. **과거의 사건은 현재의 사건을 유발할 수 있지만, 미래의 사건은 현재의 사건을 유발할 수는 없다** 라는 논리에 근거하여 인과관계를 파악하는 것이 Granger 인과관계 검정이다.

예를들어, A라는 사건이 B 사건보다 시간적으로 먼저 발생했고 미래의 B를 예측(추정)할때 B의 과거 데이터와 함께 A의 과거 데이터를 함께 사용하는것이 B의 과거값만으로 예측(추정)하는 것보다 정확하면,
A로부터 B로의 인과방향이 존재한다고 간주함.
반대일경우 B로부터 A로의 인과방향이 존재한다고 간주함.

만약, 이러한 인과관계가 두 방향으로 모두 성립되면 A와 B는 상호의존적인 관계로 쌍방의 인과방향이 존재하는 것으로 본다.

### **1-1)  VAR모형과 Granger-Causality 관계 및 활용 케이스**

- Granger Causality는 VAR모형 중 하나임.
    - 한 변수가 다른 변수에 인과적 영향을 미치는 관계를 파악하는데 중점을 둠. 그러나 그레인저 인과모형은 변수 간의 양방향 인과관계를 분석하지 않기 때문에 일부 정보가 누락될 수 있음.
- VAR (벡터 자기회귀 모형)모형은 다변량 시계열 분석으로 변수간의 관계를 이해하는데 사용될 뿐만 아니라, 예측 하는데에도 사용됨.
    - VAR 모형은 예측할 변수의 과거 값뿐만 아니라, 예측할 변수와 의존성이 있는 변수의 과거값을 사용해서 예측하는 방법
    - VAR 모델은 모든 변수 간의 상호작용을 분석할 수 있으므로, 다양한 변수 간의 복잡한 관계를 파악할 수 있고, 시간적인 선후관계(temporal ordering)를 고려하여 변수 간의 인과관계를 파악할 수 있음.

- VAR모델에서는 시계열1과 시계열2가 서로 영향을 끼친다는것을 알수있고, (양방향)
- Granger Causality Test를 통해서는 시계열1이 시계열2에 영향을 주는것인지, 시계열2가 시계열1에 영향을 주는것인지를 알 수 있음. (단방향)

<aside>
💡 따라서, 그레인저 인과모형은 단방향 인과관계를 분석하는 데 적합하며, VAR 모델은 변수 간의 상호작용과 복잡한 관계를 파악하는 데 적합함.

</aside>

### 1-2) 그레인저 인과관계 활용 예시

- 닭과 달걀의 생산량의 인과관계
- 단기금리와 중기, 단기 금리와의 인과관계
- 강수량과 동물원방문객수의 인과관계
- 급여인상액과 소비금액의 인과관계
- 어떤 회사의 광고비지출액와 매출액의 인과관계
- 강수량과 인터넷사용량의 인과관계
- 어떤 광고캠페인의 수치형 설정조건과 클릭수와 인과관계

# **02. Granger 인과관계 검정에 대한 모형**

---

x가 y에 영향을 미치지 않는다는 귀무가설을 검정하기 위하여

- (a) y를 y의 과거값과 x의 과거값에 대한 회귀식을 추정하고
- (b) y를 y의 과거값에 대해서만 회귀식을 추정함.

=> (a)가 (b)보다 통계적으로 유의미하게 예측값에 긍정적인 영향을 주었는가를 검정하여 의미있다고 확인되면, 그레인저 인과가 있다고 말할 수 있음.

***⚠️주의***

> Granger Causality 는 사람들이 생각하는 원인을 바로 찾아주고 관련이 있는지를 알려주는 인과(causality)와는 다름.
>
>
> 즉, 달걀의 개체수 증가가 미래의 닭 개체수 증가에 인과영향이 있다는 사실이 밝혀졌다고 해서, 반드시 닭의 수의 요인은 달걀의 개체수다 라고 확신해서 말하는 것은 무리가 있다는 의미.
>
> 그래서 그래인저 인과관계는 명확하게 **`“그레인저 인과관계“`**라고 명시해야함.
>

# **03. 전제 조건**

---

- 그레인저 인과 검정은 시계열 데이터에 사용하는 방법론이라서 `정상성`을 만족시켜야 함.

# **04. 사용하면 좋은 이유**

---

- 가짜 상관성을 찾아낼 수 있다.
    - 선행하지 않으면 절대 인과성이 있다 할 수 없기 때문
    - 단순 상관관계를 인과관계로 해석하는 위험을 방지할 수 있음.
- 꽤 단순한 `다중 회귀 모형`으로 , 다른 인과 모형에 비해 논쟁의 여지가 적어 빠르게 검증해볼 수 있다.

# ****05. 적용 사례****

---

> 택시 산업에서는 수요측에 택시를 타려는 고객이 있고 , 공급측에는 택시 운전기사가있다.
고객 수요가 많아야 기사 입장에서는 선택지가 많아지기 때문에 몰릴수밖에 없고, 반대로 기사 공급이 많아야 배차가 더 빨리 되서 더 많은 고객이 몰릴 수 있다.

즉, 이 둘의 관계는 쌍방의 영향이 있을거라고 누구라도 쉽게 예상이 되는데 실제로 그러한지, 한쪽이 선행하는 관계는 아닐지 *(고객이 먼저 늘어야 기사가 늘어난다거나)* Granger Causality 방법론을 사용해서 확인 해보고자 한다.
>

*(참고) 아래 데이터는 일부 변형을 거쳤고, user 는 수요 seller 는 공급으로 이해하면 됨.*

## 1) user 와 seller 의 추이 및 상관계수 체크

- user 와 seller 간의 상관계수는 0.68로 꽤 높은편.
    - **참고로 , Granger causality test 에서 높은 상관관계는 필수 조건은 아님.**
      상관관계가 높아도 그레인저 인과관계가 없을수도있고, 상관관계가 낮아도 그레인저 인과관계가 있을 수 있음.

![png](/images/2023-03-19-Granger-Causality_files/Untitled 0.png)

![png](/images/2023-03-19-Granger-Causality_files/Untitled 1.png)
## 2) **정상성 확인**

### 2-1) 트렌드 및 계절성 쪼개서 확인 (seasonal decompose)

![png](/images/2023-03-19-Granger-Causality_files/Untitled 2.png)
![png](/images/2023-03-19-Granger-Causality_files/Untitled 3.png)
- Trend(추세)
    - 이동평균은 특정 기간(k) 내의 시계열 평균을 계산하는 방법
    - 결과
        - 전반적으로 우상향 하는 Trend를 확인.
- Seasonal (계절성)
    - 가법모형, 승법모형 중 가법모형 선택
        - 가법모형(addictive) : 계절성분의 진폭이 시계열의 수준에 관계없이 일정한 수준일 때 주로 사용
        - 승법모형(multiplicative) : 시계열의 수준에 따라 진폭이 달라질 때 사용
    - 한 달 주기로 계절성 확인을 위해 freq=30 으로 설정함.
    - 결과
        - Seasonal 성분은 30일을 주기로 반복되는것을 알 수 있음.
- Residual
    - 관측값에서 계절성과 추세 성분을 빼서 나머지 불규칙 성분을 계산한 값
        - 실제 관측값 - Trend - Seasonal

### **2-2. ACF 그래프로 정상성 만족여부 확인**

- 정상성을 나타내지 않는 데이터에서는 ACF가 느리게 감소하지만, 정상성을 나타내는 시계열에서는, ACF가 비교적 빠르게 0으로 떨어짐.
    - 아래 그래프를 보면 전자인것으로 보아, user와 seller  모두 정상성을 나타내지 않는것을 확인.

![png](/images/2023-03-19-Granger-Causality_files/Untitled 4.png)
(user acf-plot)

![png](/images/2023-03-19-Granger-Causality_files/Untitled 5.png)
(seller acf-plot)

### **2-3. 단위근 검정으로 정상성 만족여부 확인**

- ADF 검정의 귀무 가설은 ‘시계열에 단위근이 존재한다’ 이고, 대립가설은 ‘시계열이 정상성을 만족한다’
  따라서, p-value <0.05 여야 정상성을 만족한다고 할 수 있음.
    - user와 seller 모두 정상성을 만족하지 못하므로 별도 처리가 필요함.

![png](/images/2023-03-19-Granger-Causality_files/Untitled 6.png)
![png](/images/2023-03-19-Granger-Causality_files/Untitled 7.png)
## 3) 정상성 만족을 위한 차분

정상성을 만족하는지를 확인하는 방법으로 아래 3가지가 있음

- **ADF-Test**
    - *(참고) **ADF test** 는 추세가 있는 비정상 시계열에 대해서는 정상 시계열이 아님을 잘 검정하지만, **분산이 변하거나 계절성이 있는 시계열에 대해서는 정상성 여부를 제대로 검정해내지 못함.** 따라서, 추가적으로 아래 KPSS test 와 같은 여러 테스트를 해보는것을 추천*
        - *실제로 특정 사례에서, 1차 차분만 했을경우 **ADF-test 에서는 정상성을 만족한다고 나왔는데 KPSS-test 에서는 정상성을 만족하지 않는것으로 나옴.** 해당 데이터는 seasonality가 있다고 판단하여 계절차분(shift=7)을 했고 그 결과 ADF-test , KPSS-test 모두 정상성을 만족하는것으로 나왔음.*
- **KPSS-Test**
- **ACF-plot** 을 그려보는 방법
    - 정상성을 나타내지 않는 데이터에서는 ACF가 느리게 감소하지만, 정상성을 나타내는 시계열에서는 ACF가 비교적 빠르게 0으로 떨어짐.

![ *(user :  1차 차분, 2차 차분, 계절차분(7) 결과 )*](![png](/images/2023-03-19-Granger-Causality_files/Untitled 8.png))

*(user :  1차 차분, 2차 차분, 계절차분(7) 결과 )*

![ *(seller :  1차 차분, 2차 차분, 계절차분(7) 결과 )*](![png](/images/2023-03-19-Granger-Causality_files/Untitled 9.png))

*(seller :  1차 차분, 2차 차분, 계절차분(7) 결과 )*

## 4) Granger causality test

- grangercausalitytests 결과로 `ssr based F test` 와 `ssr based chi2 test` 두가지 결과값이 나오는데, 일반적으로 F-test가 더 일반적으로 사용되므로 F-test를 기준으로 p-value를 판단하는 것이 좋음.
- 적정 maxlag는 도메인 지식을 바탕으로 판단해도되고, **AIC** 를 기준으로 정해도 됨.
    - maxlag를 늘려가면서 결과를 확인해보는것이 좋음.

**(data-set)**

![(seller → user 그레인저 인과관계를 확인할때 데이터셋)](/images/2023-03-19-Granger-Causality_files/Untitled 10.png))

(seller → user 그레인저 인과관계를 확인할때 데이터셋)

**(code)**

![(user→ seller 그레인저 인과관계를 확인할때 데이터셋)](/images/2023-03-19-Granger-Causality_files/Untitled 11.png))

(user→ seller 그레인저 인과관계를 확인할때 데이터셋)

```python
# 1) maxlag 에 따른 결과 나열 방식
from statsmodels.tsa.stattools import grangercausalitytests

print('\n[seller -> user]')
granger_seasonal_diff_result1 = grangercausalitytests(dataset_seasonal_diff.values, maxlag=7, verbose=True)

print('\n[user -> seller]')
granger_seasonal_diff_result2 = grangercausalitytests(dataset_seasonal_diff.iloc[:,[1,0]].values, maxlag=7, verbose=True)

# ----------------------------------------------------------------------------
# 2) grangers_causation_matrix (전체 lag중에서 p-value의 min값으로 볼수도, max값으로 볼수도)
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """   
    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
#             p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]

            p_values = [test_result[i+1][0][test][1] for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
            
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    
    return df
```

## 5) 결과

- 이 분석만으로는 정확히 알 수 없었음.
    - lag에 따라 p-value가 유의할때도 있고 유의하지 않을때도 있으며, **특히 lag가 증가할수록 p-value의 유의함이 점차 사라져가는것으로 보아, 두 변수간의 관계가 더 복잡하거나, 더 복잡한 시차(lag)를 고려해야하거나, 또는 다른 제 3의 외부 요인이 영향을 준것으로 보임.**
    - 혹은 , 해당 데이터가 Granger causality test 로 검증하기에 적합하지 않을 가능성도 있음.

