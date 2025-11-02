{{ config(materialized='table', tags=['gold']) }}

with d as (
  select distinct
    customer_id,
    customer_name,
    segment,
    sha2(concat_ws('||', customer_id, customer_name, coalesce(segment,'')), 256) as hash_k
  from {{ source('silver','superstore_clean') }}
)
select
  row_number() over (order by hash_k) as customer_sk,
  customer_id,
  customer_name,
  segment
from d;
