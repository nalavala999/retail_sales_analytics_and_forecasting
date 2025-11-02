{{ config(materialized='view', tags=['gold']) }}
select * from {{ ref('fact_sales') }};
