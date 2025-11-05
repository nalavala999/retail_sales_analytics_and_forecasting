# Databricks notebook source
dbutils.widgets.text("CATALOG", "retail_sales_analytics_and_forecasting")
dbutils.widgets.text("ML_SCHEMA", "ml")
dbutils.widgets.text("GOLD_SCHEMA", "gold")

CATALOG     = dbutils.widgets.get("CATALOG")
ML_SCHEMA   = dbutils.widgets.get("ML_SCHEMA")
GOLD_SCHEMA = dbutils.widgets.get("GOLD_SCHEMA")

SRC_TBL = f"{CATALOG}.{ML_SCHEMA}.monthly_region_product_features"
OUT_TBL = f"{CATALOG}.{GOLD_SCHEMA}.forecast_next_month"

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import numpy as np

pdf = spark.table(SRC_TBL).toPandas()
pdf["year_month"] = pd.to_datetime(pdf["year_month"])

# Split: last 6 months per combo = validation
pdf = pdf.sort_values(["region","category","sub_category","year_month"])

def tag_last_k(g, k=6):
    g = g.sort_values("year_month", ascending=False)
    g["is_valid"] = 0
    g.iloc[:k, g.columns.get_loc("is_valid")] = 1
    return g.sort_values("year_month")

feat = pdf.groupby(["region","category","sub_category"], group_keys=False).apply(tag_last_k, k=6)
train = feat[feat["is_valid"]==0].copy()
valid = feat[feat["is_valid"]==1].copy()


# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

num_cols = [c for c in [
    "sales_lag_1","sales_lag_3","sales_lag_6","sales_lag_12",
    "roll3","roll6","roll12","month","year","is_q4","profit","units","margin_pct"
] if c in train.columns]
cat_cols = ["region","category","sub_category"]

# 1) Ensure target has no NaN rows in train/valid
train = train[train["target_next_sales"].notna()].copy()
valid = valid[valid["target_next_sales"].notna()].copy()

X_train = train[num_cols + cat_cols]; y_train = train["target_next_sales"].astype(float)
X_valid = valid[num_cols + cat_cols]; y_valid = valid["target_next_sales"].astype(float)

# 2) Version-safe OHE (new sklearn uses sparse_output, older uses sparse)
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

# 3) Add SimpleImputer for BOTH numeric and categorical cols
pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),   # fill NaNs in numeric
            ("scale",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),  # fill NaNs in cats (before OHE)
            ("ohe",    ohe)
        ]), cat_cols),
    ],
    remainder="drop",
)

lr = Pipeline([
    ("prep", pre),
    ("lr", LinearRegression())
])

lr.fit(X_train, y_train)
pred = lr.predict(X_valid)

mae  = mean_absolute_error(y_valid, pred)
rmse = np.sqrt(mean_squared_error(y_valid, pred))
r2   = r2_score(y_valid, pred)
mape = np.mean(np.abs((y_valid - pred) / np.maximum(np.abs(y_valid), 1e-6)))

print(f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} MAPE={mape*100:.2f}%")


# COMMAND ----------

from pyspark.sql import functions as F

# you already have: latest -> add preds -> out_pdf with 'ym' column
# out_pdf columns: ym (pandas datetime), region, category, sub_category, next_month_sales_fcst, rmse, mape, scored_at

# 1) Convert pandas -> Spark and ENFORCE destination types
out_sdf = (spark.createDataFrame(out_pdf)
           .withColumn("ym", F.to_date("ym"))  # <-- KEY: make it DATE, not TIMESTAMP
           .withColumn("region", F.col("region").cast("string"))
           .withColumn("category", F.col("category").cast("string"))
           .withColumn("sub_category", F.col("sub_category").cast("string"))
           .withColumn("next_month_sales_fcst", F.col("next_month_sales_fcst").cast("double"))
           .withColumn("rmse", F.col("rmse").cast("double"))
           .withColumn("mape", F.col("mape").cast("double"))
           .withColumn("scored_at", F.to_timestamp("scored_at")))

# 2) Make sure the target table exists with the right schema (run once safely)
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {OUT_TBL} (
  ym DATE,
  region STRING,
  category STRING,
  sub_category STRING,
  next_month_sales_fcst DOUBLE,
  rmse DOUBLE,
  mape DOUBLE,
  scored_at TIMESTAMP
) USING delta
""")

# 3) Write (overwrite or append)
(out_sdf.write
    .mode("overwrite")                 # or "append" if you want to accumulate runs
    .option("overwriteSchema", "true") # keeps schema consistent
    .format("delta")
    .saveAsTable(OUT_TBL))

display(spark.table(OUT_TBL).orderBy(F.desc("ym"), "region", "category", "sub_category"))
