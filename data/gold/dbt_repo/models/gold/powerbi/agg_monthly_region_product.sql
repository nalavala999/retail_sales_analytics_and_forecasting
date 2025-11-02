{{ config(materialized='table', tags=['gold']) }}
select
  date_format(order_date,'yyyy-MM') as year_month,
  region,
  category,
  sub_category,
  sum(sales)    as sales,
  sum(profit)   as profit,
  sum(quantity) as units,
  sum(profit) / nullif(sum(sales),0) as margin_pct
from {{ source('silver','superstore_clean') }}
group by date_format(order_date,'yyyy-MM'), region, category, sub_category;
