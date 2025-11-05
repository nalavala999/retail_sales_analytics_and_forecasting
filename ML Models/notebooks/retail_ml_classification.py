# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 Retail ML — Classification (Profitable Order Prediction)

# COMMAND ----------

dbutils.widgets.text("CATALOG", "retail_sales_analytics_and_forecasting")
dbutils.widgets.text("GOLD_SCHEMA", "gold")
dbutils.widgets.text("RUN_TAG", "dev")

CATALOG     = dbutils.widgets.get("CATALOG")
GOLD_SCHEMA = dbutils.widgets.get("GOLD_SCHEMA")
RUN_TAG     = dbutils.widgets.get("RUN_TAG")

FACT_TBL = f"{CATALOG}.{GOLD_SCHEMA}.fact_sales"
SCORED_ORDERS_TBL = f"{CATALOG}.{GOLD_SCHEMA}.order_profit_score"

# COMMAND ----------

from pyspark.sql import functions as F

dfc = (spark.table(FACT_TBL)
       .select("order_id","order_date","sales","discount","quantity","profit","ship_days",
               "category","sub_category","region","ship_mode","segment")
       .withColumn("label_profitable", (F.col("profit") > 0).cast("int"))
       .dropna())

# COMMAND ----------

from pyspark.sql.window import Window
w = Window.orderBy(F.col("order_date").desc())
dfc = dfc.withColumn("rn", F.row_number().over(w))
total_rows = dfc.count()
test_rows = int(total_rows * 0.2) if total_rows > 10000 else int(total_rows * 0.15)
test = (dfc.orderBy(F.col("order_date").desc()).limit(test_rows)).drop("rn")
train = (dfc.join(test.select("order_id").withColumn("flag", F.lit(1)), ["order_id"], "left")
            .filter(F.col("flag").isNull()).drop("flag","rn"))

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

counts = train.groupBy("label_profitable").count().collect()
cnt = {r["label_profitable"]: r["count"] for r in counts}
total = float(sum(cnt.values()))
w0 = total / (2.0 * cnt.get(0, 1.0))
w1 = total / (2.0 * cnt.get(1, 1.0))

train = train.withColumn("weight",
                         F.when(F.col("label_profitable")==1, F.lit(w1)).otherwise(F.lit(w0)))
test  =  test.withColumn("weight",
                         F.when(F.col("label_profitable")==1, F.lit(w1)).otherwise(F.lit(w0)))

cat_cols = ["category","sub_category","region","ship_mode","segment"]
idxs = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
ohe  = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                     outputCols=[f"{c}_ohe" for c in cat_cols])

num_cols = ["sales","discount","quantity","ship_days"]
assembler = VectorAssembler(inputCols=[f"{c}_ohe" for c in cat_cols] + num_cols,
                            outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="label_profitable",
                        weightCol="weight", maxIter=100, regParam=0.01, elasticNetParam=0.0)

pipe_cls = Pipeline(stages=idxs + [ohe, assembler, lr])

# COMMAND ----------

import mlflow, mlflow.spark
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import when

with mlflow.start_run(run_name=f"order_profitability_lr_{RUN_TAG}"):
    mlflow.log_param("class_weight_0", w0)
    mlflow.log_param("class_weight_1", w1)
    model_cls = pipe_cls.fit(train)
    pred = model_cls.transform(test)

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label_profitable", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    evaluator_pr  = BinaryClassificationEvaluator(
        labelCol="label_profitable", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    auc = evaluator_auc.evaluate(pred)
    prauc = evaluator_pr.evaluate(pred)

    def f1_at_threshold(df, thr):
        p = df.withColumn("pred_thr", when(F.col("probability")[1] >= F.lit(thr), 1).otherwise(0))
        tp = p.filter((F.col("pred_thr")==1) & (F.col("label_profitable")==1)).count()
        fp = p.filter((F.col("pred_thr")==1) & (F.col("label_profitable")==0)).count()
        fn = p.filter((F.col("pred_thr")==0) & (F.col("label_profitable")==1)).count()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1 = 2*precision*recall/(precision+recall+1e-9)
        return float(f1), float(precision), float(recall)

    grid = [0.2,0.3,0.4,0.5,0.6]
    best = max(( (thr,)+f1_at_threshold(pred,thr) for thr in grid ), key=lambda x: x[1])
    best_thr, best_f1, best_p, best_r = best

    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("PR_AUC", prauc)
    mlflow.log_metric("best_threshold", best_thr)
    mlflow.log_metric("best_F1", best_f1)
    mlflow.log_metric("best_precision", best_p)
    mlflow.log_metric("best_recall", best_r)

    mlflow.spark.log_model(model_cls, artifact_path="model_cls")

# COMMAND ----------

from pyspark.sql.functions import current_timestamp
latest_90 = spark.table(FACT_TBL).filter(F.col("order_date") >= F.add_months(F.current_date(), -3))

scored = (model_cls.transform(latest_90.select("order_id","order_date","sales","discount","quantity","profit",
                                               "ship_days","category","sub_category","region","ship_mode","segment"))
          .select("order_id","order_date",
                  F.col("probability")[1].alias("prob_profit"),
                  (F.col("probability")[1] >= F.lit(best_thr)).cast("int").alias("is_profitable_pred"))
          .withColumn("scored_at", current_timestamp())
)

(scored.write.mode("overwrite").format("delta").saveAsTable(SCORED_ORDERS_TBL))