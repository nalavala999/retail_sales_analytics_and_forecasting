-- Rows per order_id in fact vs silver
WITH s AS (
  SELECT order_id, COUNT(*) AS cnt_s
  FROM retail_sales_analytics_and_forecasting.silver.superstore_clean
  GROUP BY order_id
),
f AS (
  SELECT order_id, COUNT(*) AS cnt_f
  FROM retail_sales_analytics_and_forecasting.gold.fact_sales
  GROUP BY order_id
)
SELECT f.order_id, cnt_s, cnt_f
FROM f JOIN s USING (order_id)
WHERE cnt_f <> cnt_s
ORDER BY cnt_f DESC
LIMIT 50;


-- Dim uniqueness checks (should be 1)
SELECT 'product' AS dim, product_id, COUNT(*) c
FROM retail_sales_analytics_and_forecasting.gold.dim_product
GROUP BY product_id HAVING COUNT(*) > 1
UNION ALL
SELECT 'customer', customer_id, COUNT(*) c
FROM retail_sales_analytics_and_forecasting.gold.dim_customer
GROUP BY customer_id HAVING COUNT(*) > 1
UNION ALL
SELECT 'order', order_id, COUNT(*) c
FROM retail_sales_analytics_and_forecasting.gold.dim_order
GROUP BY order_id HAVING COUNT(*) > 1;

SELECT country, region, state, city, postal_code, COUNT(*) c
FROM retail_sales_analytics_and_forecasting.gold.dim_region
GROUP BY 1,2,3,4,5
HAVING COUNT(*) > 1
ORDER BY c DESC
LIMIT 50;

-- Impossible discount values
SELECT *
FROM retail_sales_analytics_and_forecasting.gold.fact_sales
WHERE discount < 0 OR discount > 1
LIMIT 50;

-- Suspicious quantities
SELECT *
FROM retail_sales_analytics_and_forecasting.gold.fact_sales
WHERE quantity <= 0 OR quantity > 100
ORDER BY quantity DESC
LIMIT 50;

-- NULL sales but non-null quantity
SELECT *
FROM retail_sales_analytics_and_forecasting.gold.fact_sales
WHERE sales IS NULL AND quantity IS NOT NULL
LIMIT 50;


-- should be zero rows:
SELECT COUNT(*) bad_discounts
FROM retail_sales_analytics_and_forecasting.gold.fact_sales
WHERE discount < 0 OR discount > 1;

SELECT COUNT(*) bad_qty
FROM retail_sales_analytics_and_forecasting.gold.fact_sales
WHERE quantity <= 0 OR quantity > 100;

SELECT COUNT(*) sales_null_with_qty
FROM retail_sales_analytics_and_forecasting.gold.fact_sales
WHERE sales IS NULL AND quantity IS NOT NULL;

