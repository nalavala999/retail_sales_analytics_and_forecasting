{{ config(materialized='table', tags=['gold']) }}

with s as (
  select * from {{ source('silver','superstore_clean') }}
),
-- Join naturals to dims to fetch SKs (FKs in fact)
j as (
  select
    s.row_id,
    -- natural keys
    s.order_id,
    s.customer_id,
    s.product_id,
    s.country, s.region, s.state, s.city, s.postal_code,
    s.order_date,
    s.order_year, s.order_month, s.order_ym,
    s.ship_date, s.ship_days,
    s.sales, s.quantity, s.discount, s.profit
  from s
),
f as (
  select
    j.*,
    dord.order_sk,
    dc.customer_sk,
    dp.product_sk,
    dr.region_sk
  from j
  left join {{ ref('dim_order') }}   dord on j.order_id  = dord.order_id
  left join {{ ref('dim_customer') }} dc  on j.customer_id = dc.customer_id
  left join {{ ref('dim_product') }}  dp  on j.product_id  = dp.product_id
  left join {{ ref('dim_region') }}   dr  on j.country     = dr.country
                                        and j.region      = dr.region
                                        and j.state       = dr.state
                                        and j.city        = dr.city
                                        and coalesce(j.postal_code,'') = coalesce(dr.postal_code,'')
),
final as (
  select
    row_number() over (order by sha2(concat_ws('||',
        cast(row_id as string),
        order_id,
        product_id,
        cast(order_date as string)
    ), 256)) as fact_sk,

    -- FKs
    order_sk,
    customer_sk,
    product_sk,
    region_sk,

    -- naturals / dates
    row_id,
    order_id,
    order_date,
    order_year,
    order_month,
    order_ym,
    ship_date,
    ship_days,

    -- measures
    cast(sales    as double)  as sales,
    cast(quantity as int)     as quantity,
    cast(discount as double)  as discount,
    cast(profit   as double)  as profit
  from f
)
select * from final;
