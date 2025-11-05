# Databricks notebook source
# MAGIC %md
# MAGIC # 📈 Retail ML — Regression (Next-Month Sales Forecast)

# COMMAND ----------

dbutils.widgets.text("CATALOG", "retail_sales_analytics_and_forecasting")
dbutils.widgets.text("GOLD_SCHEMA", "gold")
dbutils.widgets.text("RUN_TAG", "dev")

CATALOG     = dbutils.widgets.get("CATALOG")
GOLD_SCHEMA = dbutils.widgets.get("GOLD_SCHEMA")
RUN_TAG     = dbutils.widgets.get("RUN_TAG")

AGG_TBL  = f"{CATALOG}.{GOLD_SCHEMA}.agg_monthly_region_product"
FORECAST_TBL = f"{CATALOG}.{GOLD_SCHEMA}.forecast_next_month"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

agg = (spark.table(AGG_TBL)
       .withColumn("ym", F.to_date(F.concat_ws("-", F.col("year_month"), F.lit("01"))))
       .select("ym","region","category","sub_category","sales","profit","units","discount_rate"))

min_months = 18
good = (agg.groupBy("region","category","sub_category")
            .agg(F.countDistinct("ym").alias("n_m"))
            .filter(F.col("n_m")>=min_months)
            .select("region","category","sub_category"))
agg = agg.join(good, ["region","category","sub_category"], "inner")

w = Window.partitionBy("region","category","sub_category").orderBy("ym")
feat = (agg
  .withColumn("sales_lag_1",  F.lag("sales",1).over(w))
  .withColumn("sales_lag_3",  F.lag("sales",3).over(w))
  .withColumn("sales_lag_6",  F.lag("sales",6).over(w))
  .withColumn("sales_lag_12", F.lag("sales",12).over(w))
  .withColumn("roll3",  F.avg("sales").over(w.rowsBetween(-2,0)))
  .withColumn("roll6",  F.avg("sales").over(w.rowsBetween(-5,0)))
  .withColumn("roll12", F.avg("sales").over(w.rowsBetween(-11,0)))
  .withColumn("month", F.month("ym"))
  .withColumn("year",  F.year("ym"))
  .withColumn("is_q4", F.when(F.col("month").isin(10,11,12),1).otherwise(0))
  .withColumn("target_next_sales", F.lead("sales",1).over(w))
  .dropna())

last6_w = Window.partitionBy("region","category","sub_category").orderBy(F.col("ym").desc())
feat = feat.withColumn("rn_desc", F.row_number().over(last6_w))
train = feat.filter(F.col("rn_desc") > 6).drop("rn_desc")
valid = feat.filter(F.col("rn_desc") <= 6).drop("rn_desc")

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
import mlflow, mlflow.spark

idx_region = StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep")
idx_cat    = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
idx_sub    = StringIndexer(inputCol="sub_category", outputCol="sub_idx", handleInvalid="keep")

ohe = OneHotEncoder(inputCols=["region_idx","category_idx","sub_idx"],
                    outputCols=["region_ohe","category_ohe","sub_ohe"])

num_features = ["sales_lag_1","sales_lag_3","sales_lag_6","sales_lag_12",
                "roll3","roll6","roll12","month","year","is_q4",
                "profit","units","discount_rate"]

assembler = VectorAssembler(inputCols=["region_ohe","category_ohe","sub_ohe"] + num_features,
                            outputCol="features")

gbt = GBTRegressor(featuresCol="features", labelCol="target_next_sales",
                   maxDepth=6, maxIter=200, stepSize=0.05, subsamplingRate=0.8, seed=42)

pipe_reg = Pipeline(stages=[idx_region, idx_cat, idx_sub, ohe, assembler, gbt])

with mlflow.start_run(run_name=f"next_month_sales_gbt_{RUN_TAG}"):
    mdl_reg = pipe_reg.fit(train)
    pred = mdl_reg.transform(valid).select("ym","region","category","sub_category",
                                           "target_next_sales","prediction")
    from pyspark.sql.functions import abs as Fabs
    ev = (pred.withColumn("ae", Fabs(F.col("target_next_sales")-F.col("prediction")))
               .withColumn("ape", Fabs((F.col("target_next_sales")-F.col("prediction"))/F.greatest(F.col("target_next_sales"), F.lit(1e-6)))))
    rmse = (ev.select(F.sqrt(F.avg((F.col("target_next_sales")-F.col("prediction"))**2)).alias("rmse"))
              .collect()[0]["rmse"])
    mape = (ev.select(F.avg("ape").alias("mape")).collect()[0]["mape"])

    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("mape", float(mape))
    mlflow.spark.log_model(mdl_reg, "model_reg")

from pyspark.sql.functions import current_timestamp

latest_w = Window.partitionBy("region","category","sub_category").orderBy(F.col("ym").desc())
latest_feats = (feat.withColumn("rank", F.row_number().over(latest_w))
                    .filter(F.col("rank")==1)
                    .drop("rank","target_next_sales"))

fcst = (mdl_reg.transform(latest_feats)
         .select("ym","region","category","sub_category",
                 F.col("prediction").alias("next_month_sales_fcst"))
         .withColumn("rmse", F.lit(float(rmse)))
         .withColumn("mape", F.lit(float(mape)))
         .withColumn("scored_at", current_timestamp()))

(fcst.write.mode("overwrite").format("delta").saveAsTable(FORECAST_TBL))