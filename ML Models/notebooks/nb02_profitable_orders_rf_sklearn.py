# Databricks notebook source
dbutils.widgets.text("CATALOG", "retail_sales_analytics_and_forecasting")
dbutils.widgets.text("ML_SCHEMA", "ml")
CATALOG   = dbutils.widgets.get("CATALOG")
ML_SCHEMA = dbutils.widgets.get("ML_SCHEMA")

SRC_TBL = f"{CATALOG}.{ML_SCHEMA}.order_line_training"
OUT_TBL = f"{CATALOG}.{ML_SCHEMA}.order_profit_score"

# COMMAND ----------

from pyspark.sql import functions as F
pdf = spark.table(SRC_TBL).toPandas()
print("Rows:", len(pdf))

# COMMAND ----------

import pandas as pd
from datetime import timedelta
pdf["order_date"] = pd.to_datetime(pdf["order_date"])
cutoff = pdf["order_date"].max() - pd.Timedelta(days=90)
train = pdf[pdf["order_date"] < cutoff].copy()
test  = pdf[pdf["order_date"] >= cutoff].copy()


# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

num = ["sales","discount","quantity","ship_days"]
cat = ["category","sub_category","region","ship_mode","segment"]

X_train = train[num+cat]; y_train = train["label_profitable"].astype(int)
X_test  = test[num+cat];  y_test  = test["label_profitable"].astype(int)

pre = ColumnTransformer(
    [("num", StandardScaler(), num),
     ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]
)

rf = Pipeline([
    ("prep", pre),
    ("rf", RandomForestClassifier(
        n_estimators=400, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced"))
])

rf.fit(X_train, y_train)
y_proba = rf.predict_proba(X_test)[:,1]
y_pred  = (y_proba >= 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred, digits=3))

# COMMAND ----------

from pyspark.sql import functions as F

# scored has: order_id, order_date (pandas datetime), prob_profit, is_profitable_pred, scored_at (UTC now)

out_sdf = (spark.createDataFrame(scored)
           # 🔧 enforce target schema for ml.order_profit_score
           .withColumn("order_id", F.col("order_id").cast("string"))
           .withColumn("order_date", F.to_date("order_date"))              # <-- key fix (DATE, not TIMESTAMP)
           .withColumn("prob_profit", F.col("prob_profit").cast("double"))
           .withColumn("is_profitable_pred", F.col("is_profitable_pred").cast("int"))
           .withColumn("scored_at", F.to_timestamp("scored_at")))

(out_sdf.write
    .mode("overwrite")                     # or "append" if you don't want to replace
    .option("overwriteSchema", "true")     # safe even if schema already matches
    .format("delta")
    .saveAsTable(OUT_TBL))

display(spark.table(OUT_TBL).orderBy(F.desc("order_date")).limit(20))
