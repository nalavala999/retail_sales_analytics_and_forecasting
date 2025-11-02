{{ config(materialized='table', tags=['gold']) }}

with d as (
  select distinct
    product_id,
    category,
    sub_category,
    product_name,
    sha2(concat_ws('||', product_id, category, sub_category, product_name), 256) as hash_k
  from {{ source('silver','superstore_clean') }}
)
select
  row_number() over (order by hash_k) as product_sk,
  product_id,
  category,
  sub_category,
  product_name
from d;
