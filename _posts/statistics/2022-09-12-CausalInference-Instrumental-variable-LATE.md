---
layout: single
title : "[인과추론] 인과추론 관점에서의 도구변수 (Local Average Treatment Effect)"
author_profile: true
read_time: false
comments: true
categories:
- CausalInference
published : false
---

<br>
<br>






(아래 내용은 ‘인과추론의 데이터과학' [유튜브 강의](https://www.youtube.com/watch?v=nRMZ7a4Ah8E)를 들으면서 공부 목적으로  정리한 내용입니다. 자세한 내용은 강의를 참고해주세요)


<br>

## 도구변수를 하나의 Treatment Assignment Mechanism으로 해석

⇒ 연구대상을 아래 4가지 유형(Always takers, Never takers, Compliers, Defiers)으로 구분해볼수있음

- 도구변수에 의해서 유도된 treatment가 움직이는 사람들을 Compliers 와 Defiers 라고 볼수있음.
    - 청개구리처럼, 도구변수가 1이면 control그룹으로 가고 0이면 treatment그룹으로 가는 , 반대로 움직이는 Defiers가 있으면 인과추론이 어려움. (causal effect를 제대로 구할수없게됨)
        - 따라서, 이런 Defiers는 없다는 가정을 하고 시작하고 이것을 Monotonicity assumption 이라고한다. (도구변수에 의해 한 방향으로 assignment가 되어야함)
- Always takers, Never takers는 도구변수에 의해 유도되지않음 (항상 1이거나 항상 0이거나)
- **따라서, 우리가 도구변수에 의해 추정할수 있는것은 Compliers(LATE) 라고 할수있음.**

![png](/images/2022-09-12-CausalInference-Instrumental-variable-LATE_files/Untitled 0.png)




<br>
<br>




## **LATE 의 개념을 사례로 이해해보자**

> **주제 : 전쟁 참전이 그사람의 평생 소득에 어떤 인과적인 효과를 주는지를 알고자함.**
> 

```
- 도구변수(Z) : Draft lottery (징병 우선순위 선발)
- Treatment(W) : 전쟁 참전 여부
- 임금 (Z) : 인과효과에 의한 결과
```

![png](/images/2022-09-12-CausalInference-Instrumental-variable-LATE_files/Untitled 1.png)

**<  징병 우선순위에 따른 전쟁참전 여부 >**

|  | 비선발 | 선발 |
| --- | --- | --- |
| 전쟁 비 참전 | 5948 | 1915 |
| 전쟁 참전 | 1372 | 865 |

<br>

**< 도구변수와 처치결과에 따라 추정한 평균 임금 >**

|  | 비선발 | 선발 |
| --- | --- | --- |
| 전쟁 비 참전 | 5.4472 | 5.4028 |
| 전쟁 참전 | 5.4076 | 5.4289 |

- lottery와 상관없이 징병을 가는 Always takers가 있을수있고,
- lottery와 상관없이 징병을 가지 않겠다는 Never takers가 있을수있음.
    - 전쟁이라는 context를 생각하면 Never takers가 많을듯.
- Defiers는 적을것이라고 생각할수있으므로, 이런 케이스는 없다고 가정해도 make sense할듯.
    - 우선 순위가 높은데 참전 안하고, 우선순위가 낮은데 참전하는 이런 Defiers는 있을수있지만 비율상 아주 적을거라고 합리적으로 가정할수있음. (가정이 있어야 분석이 가능하기도함)

<br>
<br>


### 1. 참전한 사람과 참전하지 않은 사람의 평균을 단순 계산했을 경우

![png](/images/2022-09-12-CausalInference-Instrumental-variable-LATE_files/Untitled 2.png)

- 이때, 참전 군인과 참전하지 않은 군인은 애초에 비교 가능하지 않기 때문에 단순 평균차로  인과관계를 밝히는것은 적절하지 않음.
    
    ⇒ 단순 평균 차이로 계산하면 위 연산에 의해 2%로 나옴.
    
- 단순 OLS를 돌려도 이 결과 그대로 2% 정도 수준으로 나옴.
    
    
<br>


### 2. 도구변수를 사용해서 2SLS를 분석할 경우

- 2SLS로 계산한 결과는 23% 수준으로 나옴.
    - 전쟁에 참전을 하는게 평생소득에 20% 손해라면, 전쟁 참여한 사람은 국가를 위해 봉사한 사람이기때문에 국가 차원에서 대대적인 보상을 주어야 한다는 근거가 될수있음.
- 이 23%의 효과가 어디에서 나타났고, 누구에게 해당되는 효과인지가 모호하고 추상적임.
    - 이 23%의 효과는 미국의 모든 남성들에게 해당되는 것일까 ?  NO
        - lottery에 순응한 사람들만 해당되는거라고 볼 수 있음.
- 도구변수가 lottery일때 ,lottery 선발 되었는데도 절대 안가겠다는 사람 (never-taker) 과 lottery 선발되지 않았는데도 무조건 가겠다는 사람 (always-taker) 이 섞여있어서 정확하지 않음. 우리가 알고자 하는것은 complier 임.
    
    ![png](/images/2022-09-12-CausalInference-Instrumental-variable-LATE_files/Untitled 3.png)

```
- complier : 국가의 부름에 수긍하여 전쟁 참전을 하겠다.(국가가 부르면가고, 안부르면 안가고)
- never-taker : 무조건 전쟁 참전하지 않겠다.
- always-taker : 무조건 전쟁 참전하겠다.
```


<br>
<br>



#### 2-1.  never-taker, always-taker, complier 각각의 비중을 구해보자.

- 정확한 never-taker, always-taker, complier 는 구분할수 없지만, (그 상황에 가지않는 이상 안 받았다면 어땠을까는 알수없으니..) 한가지 가정으로 간접적으로 추정은 가능함.
- 만약 Lottery가 완전히 랜덤으로 배정된 것이라면, 도구변수가 0일때 complier , never-taker, always-taker의 비중은 같을거라고 예상할수있음.
    - 다시 해석하면, lottery 선발되지 않은 사람들 중에서 complier, never-taker, always-taker는 모두 같은 비중으로 뽑혔을것이다.
        
        |  | 비선발 (0) | 선발 (1) |
        | --- | --- | --- |
        | 전쟁 비 참전 (0) | 5948 | 1915 |
        | 전쟁 참전 (1) | 1372 | 865 |
     
        
   - defier가 없다는 가정하에, never-taker와 always-taker의 비중을 구할 수 있음.
        - **never-taker** 비중 : 1915 / (1915 + 865) = **68.88%**
        - **always-taker** 비중 : 1372 / (1372 + 5948)  = **18.74%**
    
   - 그렇게되면, defier는 없으니까 전체 1에서 never-taker비중과 always-taker의 비중을 빼면 complier의 비중을 알수가 있음.
        - **complier** 비중 : 1 - 68.88 - 18.74 = **12.37%**
            - Draft lottery에 순응해서 전쟁에 참여한 사람은 12% 정도라는의미.
        
        |  | 비선발 (0) | 선발 (1) |
        | --- | --- | --- |
        | 전쟁 비 참전 (0) | complier / never-taker | never-taker / defier |
        | 전쟁 참전 (1) | always-taker / defier | complier / always-taker |
    

<br>
<br>


#### 2-2.  각각의 비중을 바탕으로 평균 소득 계산

(위에서구한 각 비중을 아래 임금에 곱해서 각각의 평균 임금을 계산할수있음.)

- never-taker : **5.4028**
- always-taker : **5.4076**
- compllier
    - lottery 비선발(0) 되어서 전쟁 참전하지 않은 사람의 평균임금 :
        - (0.6888 * *5.4028 + 0.1237* X) / (0.6888+0.1237) = 5.4472일때 X = **5.6948**
    - lottery 선발되어서(1) 전쟁 참전한 사람의 평균 임금 :
        - (*0.1874* * *5.4076 + 0.1237*X) / (*0.1874*+*0.1237*) =5.4289 일때 X = **5.4612**
    
    |  | 비선발 (0) | 선발 (1) |
    | --- | --- | --- |
    | 전쟁 비 참전 (0) | 5.4472 | 5.4028 |
    | 전쟁 참전 (1) | 5.4076 | 5.4289 |
    
    |  | 비선발 (0) | 선발 (1) |
    | --- | --- | --- |
    | 전쟁 비 참전 (0) | complier / never-taker | never-taker / defier |
    | 전쟁 참전 (1) | always-taker / defier | complier / always-taker |
    
<br>


#### 2-3. 결론

- COMPLIANCE TYPES : ESTIMATED AVERAGE OUTCOMES
    - lottery에 선발 → 전쟁에 참전하여 나온 평균소득과(complier(1))
    - lottery에 선발되지 않아서 → 전쟁에 참전하지 않아서 나온 평균소득 (complier(0))
        - 이 둘의 차이가 23%정도됨. 이것이 2SLS로 추정할수있었던 결과임.
        
        |  |  |
        | --- | --- |
        | never-taker = 5.4028 | defier(NA) |
        | complier(0) = 5.6948                        
        |  complier(1) = 5.4612 | always-taker = 5.4076 |

<br>

----------------------------------------------------------------

### Summary 

```
목적이 국가의 부름에 의해 순응하여 징병된 사람들에 대한 적절한 보상정책을 수립하는게 목적이라면 이 23%의 결과가 충분히 의미가있지만,

처음부터 평생 직업군인이 꿈이었던 always-taker는 23%의 효과가 적용이 안될것임.

이런식으로 2SLS가 추정하는값이 무엇인지를 이해하는것이 어떻게 해석하고 적용하는지에 중요한 역할을 함.
```


> **군대갈수 있던 모든 성인 남자들에 해당하는게 아니라, draft lottery에 순응한 (complier)사람들의 효과라고 할수있음.**

**정리하면, 이 결과는 모든 집단에 대한 결과가 아니고, 도구변수에의해 treatment  가 유도될수있는 compliers 에서의 효과인 LATE이다.**
> 

<br>


###  LATE 한계점

compliers는 특정 도구변수에 순응하는지 마는지 여부임. 그래서 도구변수가 달라지만 또다른 도구변수에 순응하는 complier는 달라질수있음. 

이게 특정 도구변수에서만 의미있고, 일반화시키기 어려움. (특별한 가정이 없는 이상)

특정 context에 국한해서 해석해야되는 한계점이 있음.

----------------------------------------------------------------


<br>

### 느낀점 및 생각정리
도구변수라는것이 독립변수와는 상관성이 높고, 종속변수와는 상관성이 낮아야하는데 이 조건을 만족하는 도구변수를 찾는것이 쉽지 않다.
예를들어, 쇼핑몰에서 팝업을 띄우고나서 이 팝업이 매출에 어떤 인과적인 효과를 주었는지를 알고싶다고 할 경우
팝업을 띄운것(독립변수)과는 상관성이 높고 매출(종속변수)과는 상관성이 낮은 어떠한 도구변수를 찾아야하는데 이것부터가 많은 고민이 필요하고
실제로 이렇다할 도구변수가 있지않을수도있다.

하지만 그럼에도 이런 방법론을 알고있으면, 어떤 인과적 효과를 밝히고싶은 문제를 직면했을때 마침 그 문제에 도구변수가 떠오를경우
이 방법을 적용 해볼 수 있을것같다.





