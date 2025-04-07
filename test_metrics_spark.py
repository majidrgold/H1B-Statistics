from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

def label_frauds(spark: SparkSession, df_fraud_gt, df_input, config: dict = {}, GT_PARAMS=None):
    """
    Processes fraud ground truth and scan tags, and merges them into a labeled DataFrame.

    Args:
        spark: SparkSession instance.
        df_fraud_gt: Spark DataFrame containing known fraud cases.
        df_input: Main input Spark DataFrame to be labeled.
        config: Dictionary with config including 'data.id' to match fraud IDs.
        GT_PARAMS: Optional ground truth parameters.

    Returns:
        A Spark DataFrame with a new 'label' column (1 for fraud, 0 for not).
    """

    input_id_col = config.get("data", {}).get("id", "tran_intnl_id")
    fraud_id_col = "tran_intnl_id"

    # --- Load and process scan data ---
    df_raw_scans = spark.read.parquet(
        "s3://csc-maas-vol1/rw/bmq_gtt/fraud_view_v_wf_fraud_nonfraud_zelle_send"
    ).select("tran_intnl_id", "sessn_id", "fraudtypeind", "init_date")

    df_raw_scans = df_raw_scans \
        .withColumn("tran_intnl_id", F.trim(F.col("tran_intnl_id"))) \
        .withColumn("sessn_id", F.trim(F.col("sessn_id"))) \
        .withColumn("scan_in", F.when(F.lower(F.col("fraudtypeind")).contains("scan"), 1).cast("int"))

    df_scan_events = df_raw_scans.filter(F.col("scan_in") == 1) \
        .dropDuplicates(["tran_intnl_id", "sessn_id"])

    # --- Process fraud ground truth data ---
    df_fraud_gt = df_fraud_gt \
        .withColumn("tran_intnl_id", F.trim(F.col("tran_intnl_id"))) \
        .withColumn("sessn_id", F.trim(F.col("sessn_id"))) \
        .withColumn("label", F.lit(1))

    fraud_rank_window = Window.partitionBy("tran_intnl_id").orderBy("OPEN_DT")
    df_fraud_gt = df_fraud_gt.withColumn("rank", F.row_number().over(fraud_rank_window)) \
        .filter(F.col("rank") == 1)

    # --- Join fraud and scan events ---
    df_fraud_scan_tags = df_fraud_gt.join(
        df_scan_events.select(
            "scan_in", "fraudtypeind", "tran_intnl_id", "sessn_id", 
            F.col("init_date").alias("init_date_scam")
        ),
        on=["tran_intnl_id", "sessn_id"],
        how="outer"
    )

    # --- Assign label: 1 if fraud, 0 if scan ---
    df_fraud_scan_tags = df_fraud_scan_tags.withColumn(
        "label",
        F.when(F.col("scan_in") == 1, F.lit(0)).otherwise(F.col("label"))
    )

    # --- Join with input data ---
    df_labeled = df_input.join(
        F.broadcast(df_fraud_scan_tags),
        df_input[input_id_col] == df_fraud_scan_tags[fraud_id_col],
        how="left"
    ).na.fill({"label": 0})

    return df_labeled
