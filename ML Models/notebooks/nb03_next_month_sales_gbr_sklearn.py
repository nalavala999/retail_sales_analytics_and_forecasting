# Databricks notebook source
dbutils.widgets.text("CATALOG", "retail_sales_analytics_and_forecasting")
dbutils.widgets.text("ML_SCHEMA", "ml")
dbutils.widgets.text("GOLD_SCHEMA", "gold")

CATALOG     = dbutils.widgets.get("CATALOG")
ML_SCHEMA   = dbutils.widgets.get("ML_SCHEMA")
GOLD_SCHEMA = dbutils.widgets.get("GOLD_SCHEMA")

SRC_TBL = f"{CATALOG}.{ML_SCHEMA}.monthly_region_product_features"
OUT_TBL = f"{CATALOG}.{GOLD_SCHEMA}.forecast_next_month"

print("Source:", SRC_TBL, "| Output:", OUT_TBL)

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

pdf = spark.table(SRC_TBL).toPandas()
print("Rows:", len(pdf))
# Expect columns: year_month (DATE), region, category, sub_category,
# sales_lag_*, roll*, month, year, is_q4, profit, units, margin_pct, target_next_sales

pdf["year_month"] = pd.to_datetime(pdf["year_month"])

# Validate required columns
req = ["region","category","sub_category","year_month","target_next_sales"]
for c in req:
    if c not in pdf.columns:
        raise ValueError(f"Missing column: {c}")

# COMMAND ----------

pdf = pdf.sort_values(["region","category","sub_category","year_month"])
def tag_last_k(g, k=6):
    g = g.sort_values("year_month", ascending=False)
    g["is_valid"] = 0
    g.iloc[:k, g.columns.get_loc("is_valid")] = 1
    return g.sort_values("year_month")

feat = pdf.groupby(["region","category","sub_category"], group_keys=False).apply(tag_last_k, k=6)

train = feat[feat["is_valid"]==0].copy()
valid = feat[feat["is_valid"]==1].copy()

print("Train:", train.shape, " Valid:", valid.shape)

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

num_cols = [c for c in [
    "sales_lag_1","sales_lag_3","sales_lag_6","sales_lag_12",
    "roll3","roll6","roll12","month","year","is_q4","profit","units","margin_pct"
] if c in train.columns]
cat_cols = ["region","category","sub_category"]

X_train = train[num_cols + cat_cols]; y_train = train["target_next_sales"].astype(float)
X_valid = valid[num_cols + cat_cols]; y_valid = valid["target_next_sales"].astype(float)

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),   # 👈 fill NaNs in numeric
            ("scale",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),  # 👈 fill NaNs in cats
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ],
    remainder="drop",
)

gbr = Pipeline([
    ("prep", pre),
    ("gbr", GradientBoostingRegressor(random_state=42))
])

gbr.fit(X_train, y_train)
pred = gbr.predict(X_valid)

mae  = mean_absolute_error(y_valid, pred)
rmse = np.sqrt(mean_squared_error(y_valid, pred))
r2   = r2_score(y_valid, pred)
mape = np.mean(np.abs((y_valid - pred) / np.maximum(np.abs(y_valid), 1e-6)))

print(f"MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} MAPE={mape*100:.2f}%")


# COMMAND ----------

latest = feat.sort_values("year_month").groupby(["region","category","sub_category"], as_index=False).tail(1).copy()
X_latest = latest[num_cols + cat_cols]
latest["next_month_sales_fcst"] = gbr.predict(X_latest)
latest["rmse"] = float(rmse)
latest["mape"] = float(mape)
latest["scored_at"] = pd.Timestamp.utcnow()

# COMMAND ----------

# ---- ONE-CELL FORECASTER & WRITER (sklearn, no Spark-ML, creates `fcst`) ----
# Inputs: retail_sales_analytics_and_forecasting.ml.monthly_region_product_features
# Output: retail_sales_analytics_and_forecasting.gold.forecast_next_month (ym DATE)
from pyspark.sql import functions as F
import pandas as pd
import numpy as np

# 0) Config (edit if needed)
CATALOG     = "retail_sales_analytics_and_forecasting"
ML_SCHEMA   = "ml"
GOLD_SCHEMA = "gold"
SRC_TBL = f"{CATALOG}.{ML_SCHEMA}.monthly_region_product_features"
OUT_TBL = f"{CATALOG}.{GOLD_SCHEMA}.forecast_next_month"
print("Source:", SRC_TBL, "| Output:", OUT_TBL)

# 1) Load features to pandas
pdf = spark.table(SRC_TBL).toPandas()
if pdf.empty:
    raise ValueError("No rows in source table.")
# Expected cols: year_month, region, category, sub_category, sales_lag_*, roll*, month, year, is_q4, profit, units, margin_pct, target_next_sales
pdf["year_month"] = pd.to_datetime(pdf["year_month"])

# 2) Tag last 6 months per (region, category, sub_category) as validation
pdf = pdf.sort_values(["region","category","sub_category","year_month"])
def tag_last_k(g, k=6):
    g = g.sort_values("year_month", ascending=False)
    g["is_valid"] = 0
    g.iloc[:k, g.columns.get_loc("is_valid")] = 1
    return g.sort_values("year_month")
feat = pdf.groupby(["region","category","sub_category"], group_keys=False).apply(tag_last_k, k=6)

train = feat[feat["is_valid"]==0].copy()
valid = feat[feat["is_valid"]==1].copy()

# 3) Build features/target
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor  # handles NaNs well
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

num_cols = [c for c in [
    "sales_lag_1","sales_lag_3","sales_lag_6","sales_lag_12",
    "roll3","roll6","roll12","month","year","is_q4","profit","units","margin_pct"
] if c in train.columns]
cat_cols = ["region","category","sub_category"]

# guard: if some numeric cols missing entirely, still proceed
if not num_cols:
    raise ValueError("No numeric feature columns found. Check your monthly feature SQL.")

X_train = train[num_cols + cat_cols]
y_train = train["target_next_sales"].astype(float)
X_valid = valid[num_cols + cat_cols]
y_valid = valid["target_next_sales"].astype(float)

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ],
    remainder="drop",
)

model = Pipeline([
    ("prep", pre),
    ("hgb", HistGradientBoostingRegressor(random_state=42))
])

model.fit(X_train, y_train)
pred = model.predict(X_valid)

mae = mean_absolute_error(y_valid, pred)
rmse = float(np.sqrt(((y_valid - pred)**2).mean()))
r2   = r2_score(y_valid, pred)
mape = float(np.mean(np.abs((y_valid - pred) / np.maximum(np.abs(y_valid), 1e-6))))
print(f"Validation: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f} MAPE={mape*100:.2f}%")

# 4) Build *latest* per group and predict next month → create `fcst`
latest = feat.sort_values("year_month").groupby(["region","category","sub_category"], as_index=False).tail(1).copy()
X_latest = latest[num_cols + cat_cols]
latest["next_month_sales_fcst"] = model.predict(X_latest)
latest["rmse"]      = rmse
latest["mape"]      = mape
latest["scored_at"] = pd.Timestamp.utcnow()

# This is your predictions DataFrame you asked for:
fcst = latest.rename(columns={"year_month":"ym"})[
    ["ym","region","category","sub_category","next_month_sales_fcst","rmse","mape","scored_at"]
]

# 5) Cast to target schema and write to Delta (fixes DATE vs TIMESTAMP issues)
out_sdf = (spark.createDataFrame(fcst)
           .withColumn("ym", F.to_date("ym"))  # ensure DATE
           .withColumn("region", F.col("region").cast("string"))
           .withColumn("category", F.col("category").cast("string"))
           .withColumn("sub_category", F.col("sub_category").cast("string"))
           .withColumn("next_month_sales_fcst", F.col("next_month_sales_fcst").cast("double"))
           .withColumn("rmse", F.col("rmse").cast("double"))
           .withColumn("mape", F.col("mape").cast("double"))
           .withColumn("scored_at", F.to_timestamp("scored_at")))

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

(out_sdf.write
    .mode("overwrite")                # or 'append' if you want to accumulate runs
    .option("overwriteSchema","true")
    .format("delta")
    .saveAsTable(OUT_TBL))

print("✓ Rows written:", out_sdf.count())
display(spark.table(OUT_TBL).orderBy(F.desc("ym")).limit(10))
