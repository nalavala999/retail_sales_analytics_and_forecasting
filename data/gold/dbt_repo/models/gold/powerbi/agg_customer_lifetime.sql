{{ config(materialized='table', tags=['gold']) }}
select
  customer_id,
  customer_name,
  segment,
  min(order_date) as first_order_date,
  max(order_date) as last_order_date,
  count(distinct order_id) as orders,
  sum(quantity) as units,
  round(sum(sales),2) as sales,
  round(sum(profit),2) as profit,
  round(sum(profit)/nullif(sum(sales),0),4) as margin_pct
from {{ source('silver','superstore_clean') }}
group by customer_id, customer_name, segment;