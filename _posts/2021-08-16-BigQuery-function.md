---
layout: single
title : "헷갈리는 BigQuery 함수 정리"
author_profile: true
read_time: false
comments: true
categories:
- BigQuery
---

## (1) TIMESTAMP / DATETIME

아래 쿼리를 실행하면 뒤에 UTC가 붙고, 안붙고의 차이 말고는 동일한 결과로 보인다.

(참고 : 빅쿼리는 UTC Timezone을 사용. KST는 UTC+9:00)

```sql
select current_datetime() cur_dt
    , current_timestamp() cur_ts
```

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816185242440.png" alt="image-20210816185242440" style="zoom:50%;" />



**Q : 여기서 만약에 UTC 시간을 우리나라 시간으로 바꾸고 싶을때, 아래 두가지중 어떤것을 사용해야 할까 ?**

**A : datetime(current_timestamp(), 'Asia/Seoul')  이것을 사용해야함.**

```sql
select timestamp(current_datetime(), 'Asia/Seoul') cur_dt_to_ts
     , datetime(current_timestamp(), 'Asia/Seoul') cur_ts_to_dt
```

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816185324417.png" alt="image-20210816185324417" style="zoom:50%;" />



- current_datetime에 timestamp() 를 씌우면 우리나라 날짜를 → UTC로 변경 했을때의 결과가 나옴
  - 2021-07-30 04:31 → 2021-07-29 19:31
- current_timestmap에 datetime() 을 씌우면 UTC 기준 날짜를 → KST로 변경했을때의 결과가 나옴
  - 2021-07-30 04:31 → 2021-07-30 13:31
    - ex) 어떤 피쳐를 UTC기준에서는 새벽 4시에 사용률이 피크였는데 우리나라 시간 기준으로 보면 오후 1시가 사용률이 피크 라는 것을 인지하고 분석할 수 있음





## (2) DATETIME_TRUNC

2021년 7월 28일을 기준으로 잡았을때,

- `day`로 지정할 경우 시간단위 제외하고 일자만 나옴
- `week`로 지정할 경우, 해당 주의 전주 일요일 날짜인 7월 25일이 나옴 (일요일이 DEFAULT이고 직접 요일 지정 가능)
- `month` 로 지정할 경우,  해당월의 첫 날짜인 7/1이 나옴.

```sql
-- DEFAULT로 지정한 dt는 2021-07-28 (수)
DECLARE dt datetime DEFAULT date_sub(current_datetime('Asia/Seoul'),interval 5 day);

select dt as dt_original
    , date_trunc(dt ,day)as dt_trunc_day
    , date_trunc(dt ,week)as dt_trunc_week
    , date_trunc(dt ,month)as dt_trunc_month
```

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816185613522.png" alt="image-20210816185613522" style="zoom:50%;" />





## (3) Window frame

### ROWS BETWEEN

4-1) rows between unbounded preceding and `current row` 는 partition by 로 나눈 그룹별로 위에서부터 순차적으로 누적합계를 계산함.

```sql
with sample_data as (
          SELECT 'A' key_group, 'A1' key, 500 value UNION ALL
          SELECT 'A' key_group, 'A1' key, 20 value UNION ALL
          SELECT 'A' key_group, 'A0' key, 100 value UNION ALL
          SELECT 'B' key_group, 'B2' key,  400 value UNION ALL
          SELECT 'B' key_group, 'B2' key, 100 value
)

select *
    , sum(value)over(partition by key_group order by key rows between unbounded preceding and current row) rows_sum
from sample_data
```

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816185838313.png" alt="image-20210816185838313" style="zoom:25%;" />

4-2) rows between unbounded preceding and `unbounded following` 는 partition by 로 나눈 그룹의 전체 누적합을 계산함.

```sql
select *
    , sum(value)over(partition by key_group order by key rows between unbounded preceding and unbounded following) rows_sum
from sample_data
```

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816185901299.png" alt="image-20210816185901299" style="zoom:25%;" />

4-3) rows between ~ 이런거 추가 안하고 그냥 sum( ) over (partition by        order by       ) 만 쓰면, DEFAULT인

```sql
RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

위 명령어로 처리되는데, 이것은 Order by 로 정렬한 결과에 동일한 값이 있을 때 합쳐서 처리한다.

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816185931817.png" alt="image-20210816185931817" style="zoom:25%;" />





## (4) APPROX_QUANTILES , PERCENTILE_CONT 차이

- 정의는, **APPROX_QUANTILES( )** 는 근사치 경계를 반환하고, **PERCENTILE_CONT( )** 는 지정된 백분위수 값을 선형 보간으로 계산한다고 되어있음.
- 그래서 두 함수 결과값이 살짝 다름.

APPROX_QUANTILES( ) 는 근사치 값이므로 소수점까지 나타내주지는 않음.

```sql
SELECT APPROX_QUANTILES(x, 4)[OFFSET(0)] q_min
    , APPROX_QUANTILES(x,4)[OFFSET(1)] q1
    , APPROX_QUANTILES(x,4)[OFFSET(2)] q2
    , APPROX_QUANTILES(x,4)[OFFSET(3)] q3
    , APPROX_QUANTILES(x,4)[OFFSET(4)] q_max
FROM UNNEST([1, 1, 1, 4, 5, 6, 7, 8, 9, 10]) AS x;
```

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816190007324.png" alt="image-20210816190007324" style="zoom:25%;" />



PERCENTILE_CONT( ) 는 반드시 over()를 동반해야됨. 따라서, 각 row 별로 percentile값이 출력됨.

```sql
select percentile_cont(x,0)over() as q_min
    , percentile_cont(x,0.25)over() as q1
    , percentile_cont(x,0.5)over() as q2
    , percentile_cont(x,0.75)over() as q3
    , percentile_cont(x,1)over() as q_max
FROM UNNEST([1, 1, 1, 4, 5, 6, 7, 8, 9, 10]) AS x;
```

<img src="/Users/suhyun/Library/Application Support/typora-user-images/image-20210816190025653.png" alt="image-20210816190025653" style="zoom:25%;" />



