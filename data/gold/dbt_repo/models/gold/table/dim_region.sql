{{ config(materialized='table', tags=['gold']) }}

with d as (
  select distinct
    country,
    region,
    state,
    city,
    postal_code,
    sha2(concat_ws('||', country, region, state, city, coalesce(postal_code,'')), 256) as hash_k
  from {{ source('silver','superstore_clean') }}
)
select
  row_number() over (order by hash_k) as region_sk,
  country,
  region,
  state,
  city,
  postal_code
from d;
