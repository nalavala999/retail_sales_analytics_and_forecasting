{{ config(materialized='table', tags=['gold']) }}

with bounds as (
  select
    coalesce(min(order_date), to_date('2013-01-01')) as min_d,
    coalesce(max(order_date), to_date('2018-12-31')) as max_d
  from {{ source('silver','superstore_clean') }}
),
base as (select sequence(min_d, max_d, interval 1 day) as d from bounds),
dates as (select explode(d) as date_key from base),
final as (
  select
    date_key as date,
    year(date_key) as year,
    quarter(date_key) as quarter,
    concat('Q', quarter(date_key)) as quarter_label,
    month(date_key) as month,
    date_format(date_key, 'MMMM') as month_name,
    date_format(date_key, 'yyyy-MM') as year_month,
    weekofyear(date_key) as week_of_year,
    dayofmonth(date_key) as day_of_month,
    date_format(date_key, 'EEE') as weekday_name,
    case when dayofweek(date_key) in (1,7) then 0 else 1 end as is_weekday,
    sha2(cast(date_key as string), 256) as hash_k
  from dates
)
select
  row_number() over (order by hash_k) as date_sk,
  *
  except(hash_k)
from final;
