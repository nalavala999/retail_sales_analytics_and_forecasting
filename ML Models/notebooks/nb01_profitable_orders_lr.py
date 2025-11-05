# Databricks notebook source
dbutils.widgets.text("CATALOG", "retail_sales_analytics_and_forecasting")
dbutils.widgets.text("ML_SCHEMA", "ml")
CATALOG   = dbutils.widgets.get("CATALOG")
ML_SCHEMA = dbutils.widgets.get("ML_SCHEMA")

SRC_TBL = f"{CATALOG}.{ML_SCHEMA}.order_line_training"
OUT_TBL = f"{CATALOG}.{ML_SCHEMA}.order_profit_score"

print("Source:", SRC_TBL)
print("Output:", OUT_TBL)

# COMMAND ----------

from pyspark.sql import functions as F
pdf = spark.table(SRC_TBL).toPandas()
print("Rows:", len(pdf))
pdf.head(3)


# COMMAND ----------

import pandas as pd
from datetime import timedelta

pdf["order_date"] = pd.to_datetime(pdf["order_date"])
max_dt = pdf["order_date"].max()
cutoff = max_dt - pd.Timedelta(days=90)

train = pdf[pdf["order_date"] < cutoff].copy()
test  = pdf[pdf["order_date"] >= cutoff].copy()
print("Train:", train.shape, " Test:", test.shape)

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

feature_cols_num = ["sales","discount","quantity","ship_days"]
feature_cols_cat = ["category","sub_category","region","ship_mode","segment"]
target_col = "label_profitable"

# Ensure columns exist
for col in feature_cols_num + feature_cols_cat + [target_col]:
    if col not in train.columns:
        raise ValueError(f"Missing column: {col}")

X_train = train[feature_cols_num + feature_cols_cat]
y_train = train[target_col].astype(int)
X_test  = test[feature_cols_num + feature_cols_cat]
y_test  = test[target_col].astype(int)

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_cols_cat),
    ],
    remainder="drop",
)

clf = Pipeline(steps=[
    ("prep", pre),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None))
])

clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)[:,1]
y_pred  = (y_proba >= 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred, digits=3))


# COMMAND ----------

from pyspark.sql import functions as F

# enforce the exact schema that ml.order_profit_score expects
# order_id STRING, order_date DATE, prob_profit DOUBLE, is_profitable_pred INT, scored_at TIMESTAMP
out_sdf = (spark.createDataFrame(scored)
           .withColumn("order_id", F.col("order_id").cast("string"))
           .withColumn("order_date", F.to_date("order_date"))              # <-- key fix
           .withColumn("prob_profit", F.col("prob_profit").cast("double"))
           .withColumn("is_profitable_pred", F.col("is_profitable_pred").cast("int"))
           .withColumn("scored_at", F.to_timestamp("scored_at")))

# write (keep overwrite if you want to replace the whole table each run)
(out_sdf.write
       .mode("overwrite")
       .option("overwriteSchema", "true")   # safe even if schema already matches
       .format("delta")
       .saveAsTable(OUT_TBL))

display(spark.table(OUT_TBL).orderBy(F.desc("order_date")).limit(20))

