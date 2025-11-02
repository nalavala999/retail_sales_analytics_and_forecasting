{{ config(materialized='view', tags=['gold']) }}
select * from {{ ref('dim_date') }};
