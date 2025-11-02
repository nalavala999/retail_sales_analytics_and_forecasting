{{ config(materialized='table', tags=['gold']) }}

with d as (
  select distinct
    order_id,
    order_date,
    ship_date,
    ship_mode,
    ship_days,
    case
      when ship_days is null then 'Unknown'
      when ship_days <= 2 then '0-2 days'
      when ship_days <= 5 then '3-5 days'
      else '6+ days'
    end as ship_speed_bucket,
    sha2(concat_ws('||', order_id, cast(order_date as string), cast(ship_date as string),
                   coalesce(ship_mode,''), coalesce(cast(ship_days as string),'')), 256) as hash_k
  from {{ source('silver','superstore_clean') }}
)
select
  row_number() over (order by hash_k) as order_sk,
  order_id,
  order_date,
  ship_date,
  ship_mode,
  ship_days,
  ship_speed_bucket
from d;
