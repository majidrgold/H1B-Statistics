import os
import pytest
import numpy as np
from scipy.stats import kstest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, FloatType

from commons import _gen_random_values, _gen_classification_values
from metrics_spark import MetricsSpark
from dotenv import load_dotenv

load_dotenv(override=True)


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.appName("test").getOrCreate()
    yield spark
    spark.stop()


def _cal_vdr(df, field, threshold, amt_field):
    if amt_field is None:
        return None, "NO_FRAUD_AMT"
    total_value = 0
    hit_value = 0
    for row in df.collect():
        amount = row[amt_field]
        label = row[field + "_label"]
        score = row[field]
        total_value += amount if label == 1 else 0
        if score >= threshold and label == 1:
            hit_value += amount
    return hit_value / total_value if total_value != 0 else None


def _cal_tdr(df, field, threshold):
    tp_count = 0
    total_count = 0
    for row in df.collect():
        label = row[field + "_label"]
        score = row[field]
        if label == 1:
            total_count += 1
            if score >= threshold:
                tp_count += 1
    return tp_count / total_count if total_count != 0 else None


def _cal_tfpr(df, field, threshold):
    fp_count = 0
    tp_count = 0
    for row in df.collect():
        label = row[field + "_label"]
        score = row[field]
        if label == 0 and score >= threshold:
            fp_count += 1
        elif label == 1 and score >= threshold:
            tp_count += 1
    if tp_count == 0:
        return None, "0_TP"
    return fp_count / tp_count


def _cal_hist_proportions(reference, monitored, bins):
    if len(reference) == 0 or len(monitored) == 0:
        return np.zeros(bins), np.zeros(bins)
    bin_edges = np.linspace(min(reference), max(reference), bins + 1)

    reference_hist, _ = np.histogram(reference, bins=bin_edges)
    monitored_hist, _ = np.histogram(monitored, bins=bin_edges)

    reference_proportions = reference_hist / len(reference)
    monitored_proportions = monitored_hist / len(monitored)

    return monitored_proportions, reference_proportions


def _cal_psi(monitored_proportions, reference_proportions):
    # Avoid division by zero with a small constant
    reference_proportions = np.where(
        reference_proportions == 0, 1e-10, reference_proportions
    )
    monitored_proportions = np.where(
        monitored_proportions == 0, 1e-10, monitored_proportions
    )

    psi_values = (monitored_proportions - reference_proportions) * np.log(
        monitored_proportions / reference_proportions
    )
    psi = np.sum(psi_values)
    return psi


def _cal_ks_test(x, x_bl):
    ks_stat, p_value = kstest(x, x_bl)
    return [ks_stat, p_value]


def _gen_case(case_name, metric, func, args):
    """
    For 'vdr', 'tdr', and 'tfpr' metrics, returns raw Python data (values, labels, amt_values).
    For 'psi' and 'ks', returns numeric arrays plus expected results.
    """
    if metric in ["vdr", "tdr", "tfpr"]:
        if callable(args):
            inputs = args()
        else:
            inputs = func(**args)

        values = [float(val) for val in inputs["values"]]
        labels = [int(label) for label in inputs["labels"]]
        amt_values = [float(amt) for amt in inputs["amt_values"]]

        field = "score"
        threshold = args.get("threshold", 0.5) if isinstance(args, dict) else 0.5
        amt_field = "amount"

        # Return raw data. The test function will create Spark DataFrame and do the actual calculations.
        return {
            "case": case_name,
            "metric": metric,
            "inputs": {
                "values": values,
                "labels": labels,
                "amt_values": amt_values,
                "field": field,
                "threshold": threshold,
                "amt_field": amt_field,
            },
            "expected": {
                # We'll let the test function compute actual values and compare
                "res": None,
                "tag": (
                    args.get("tag")
                    if isinstance(args, dict) and "tag" in args
                    else None
                ),
            },
        }

    elif metric in ["psi", "ks"]:
        inputs = func(**args)
        x, x_bl = inputs["a"], inputs["b"]
        inputs_dict = {"x": x, "x_bl": x_bl}
        exp_tag = None

        if metric == "psi":
            nb = 10
            inputs_dict["hist_data"], inputs_dict["hist_bl"] = _cal_hist_proportions(
                x_bl, x, nb
            )
            exp_res = _cal_psi(inputs_dict["hist_data"][2:], inputs_dict["hist_bl"][2:])
            inputs_dict["nb"] = nb
        elif metric == "ks":
            nb = 10
            inputs_dict["hist_data"], inputs_dict["hist_bl"] = _cal_hist_proportions(
                x_bl, x, nb
            )
            exp_res = _cal_ks_test(x, x_bl)
        else:
            exp_res = None

        return {
            "case": case_name,
            "inputs": inputs_dict,
            "expected": {"res": exp_res, "tag": exp_tag},
        }


def _gen_data(metric):
    """
    Generate raw data only. Spark usage is deferred to the test functions.
    """
    if metric in ["vdr", "tdr", "tfpr"]:
        return [
            _gen_case(
                "all positive values",
                metric,
                _gen_classification_values,
                {"val_range": 10},
            ),
            _gen_case(
                "all large values",
                metric,
                _gen_classification_values,
                {"val_range": 1000000, "offset": 0.5},
            ),
            _gen_case(
                "all zero values",
                metric,
                _gen_classification_values,
                {"val_range": 10, "offset": 0.5},
            ),
            _gen_case(
                "null entry values",
                metric,
                # Instead of trying to use a lambda function to return empty arrays,
                # define a simple fixed dictionary function
                lambda: {
                    "values": np.array([]),
                    "labels": np.array([]),
                    "amt_values": np.array([]),
                },
                {},  # Empty dictionary as args, since the function doesn't need args
            ),
            _gen_case(
                "single class",
                metric,
                # Same approach here
                lambda: {
                    "values": [1, 1, 1, 1],
                    "labels": [1, 1, 1, 1],
                    "amt_values": [10, 10, 10, 10],
                },
                {},  # Empty dictionary as args
            ),
        ]
    elif metric in ["psi", "ks"]:
        return [
            _gen_case("all positive values", metric, _gen_random_values, {}),
            _gen_case(
                "all negative values",
                metric,
                _gen_random_values,
                {"val_range": -10},
            ),
            _gen_case("mixed values", metric, _gen_random_values, {"offset": 0.5}),
            _gen_case(
                "large values",
                metric,
                _gen_random_values,
                {"val_range": 10000000, "offset": 0.5},
            ),
            _gen_case(
                "small values",
                metric,
                _gen_random_values,
                {"val_range": 0.00000001, "offset": 0.5},
            ),
        ]
    else:
        raise ValueError(f"Unknown metric type: {metric}")


# Generate data at the module level without Spark
data_psi = _gen_data("psi")
data_ks = _gen_data("ks")
data_vdr = _gen_data("vdr")
data_tdr = _gen_data("tdr")
data_tfpr = _gen_data("tfpr")


@pytest.fixture()
def gen_data(request):
    return request.param["inputs"], request.param["expected"]


@pytest.mark.parametrize(
    "gen_data", argvalues=data_psi, indirect=True, ids=[t["case"] for t in data_psi]
)
def test_hist_density(gen_data):
    inputs, expected = gen_data
    field = "data"
    schema = StructType([StructField(field, FloatType(), True)])
    inputs["x"] = [float(x) for x in inputs["x"]]
    inputs["x_bl"] = [float(x) for x in inputs["x_bl"]]

    data_df = spark.createDataFrame([(x,) for x in inputs["x"]], schema)
    bl_df = spark.createDataFrame([(x,) for x in inputs["x_bl"]], schema)

    bins = MetricsSpark.hist_bins([data_df, bl_df], field, inputs["nb"])
    hist_data = MetricsSpark.hist_density(data_df, field, bins)
    hist_bl = MetricsSpark.hist_density(bl_df, field, bins)

    for res, exp_res in zip(hist_data, inputs["hist_data"]):
        assert np.isclose(res, exp_res, rtol=0.1)

    for res, exp_res in zip(hist_bl, inputs["hist_bl"]):
        assert np.isclose(res, exp_res, rtol=0.1)


@pytest.mark.parametrize(
    "gen_data", argvalues=data_ks, indirect=True, ids=[t["case"] for t in data_ks]
)
def test_ks(gen_data):
    inputs, expected = gen_data
    res = MetricsSpark.ks(inputs["hist_data"][2:], inputs["hist_bl"][2:])
    if expected["res"] is None:
        assert res is None
    else:
        assert np.isclose(res, expected["res"][0], rtol=1e-4)


@pytest.mark.parametrize(
    "gen_data", argvalues=data_psi, indirect=True, ids=[t["case"] for t in data_psi]
)
def test_psi(gen_data):
    inputs, expected = gen_data
    res = MetricsSpark.psi(inputs["hist_data"], inputs["hist_bl"])
    if expected["res"] is None:
        assert res is None
    else:
        assert np.isclose(res, expected["res"], rtol=1e-4)


@pytest.mark.parametrize(
    "gen_data", argvalues=data_vdr, indirect=True, ids=[t["case"] for t in data_vdr]
)
def test_vdr(spark, gen_data):
    inputs, expected = gen_data

    # Convert raw data to Spark DataFrame inside the test
    values = inputs["values"]
    labels = inputs["labels"]
    amt_values = inputs["amt_values"]

    if len(values) == 0:
        df = spark.createDataFrame(
            [],
            schema=StructType(
                [
                    StructField(inputs["field"], FloatType(), True),
                    StructField(inputs["field"] + "_label", FloatType(), True),
                    StructField(inputs["amt_field"], FloatType(), True),
                ]
            ),
        )
    else:
        df = spark.createDataFrame(
            [
                Row(score=val, score_label=label, amount=amt)
                for val, label, amt in zip(values, labels, amt_values)
            ]
        )

    res, tag = MetricsSpark.vdr(
        df, inputs["field"], inputs["threshold"], inputs["amt_field"]
    )

    if expected["res"] is not None:
        assert np.isclose(res, expected["res"], rtol=1e-4) or res is None
    if expected["tag"] is not None:
        assert tag == expected["tag"]


@pytest.mark.parametrize(
    "gen_data", argvalues=data_tdr, indirect=True, ids=[t["case"] for t in data_tdr]
)
def test_tdr(spark, gen_data):
    inputs, expected = gen_data
    values = inputs["values"]
    labels = inputs["labels"]

    if len(values) == 0:
        df = spark.createDataFrame(
            [],
            schema=StructType(
                [
                    StructField(inputs["field"], FloatType(), True),
                    StructField(inputs["field"] + "_label", FloatType(), True),
                    StructField(inputs["amt_field"], FloatType(), True),
                ]
            ),
        )
    else:
        df = spark.createDataFrame(
            [
                Row(score=val, score_label=label, amount=amt)
                for val, label, amt in zip(values, labels, inputs["amt_values"])
            ]
        )

    res, tag = MetricsSpark.tdr(df, inputs["field"], inputs["threshold"])

    if expected["res"] is not None:
        assert np.isclose(res, expected["res"], rtol=1e-4) or res is None
    if expected["tag"] is not None:
        assert tag == expected["tag"]


@pytest.mark.parametrize(
    "gen_data", argvalues=data_tfpr, indirect=True, ids=[t["case"] for t in data_tfpr]
)
def test_tfpr(spark, gen_data):
    inputs, expected = gen_data
    values = inputs["values"]
    labels = inputs["labels"]

    if len(values) == 0:
        df = spark.createDataFrame(
            [],
            schema=StructType(
                [
                    StructField(inputs["field"], FloatType(), True),
                    StructField(inputs["field"] + "_label", FloatType(), True),
                    StructField(inputs["amt_field"], FloatType(), True),
                ]
            ),
        )
    else:
        df = spark.createDataFrame(
            [
                Row(score=val, score_label=label, amount=amt)
                for val, label, amt in zip(values, labels, inputs["amt_values"])
            ]
        )

    res, tag = MetricsSpark.tfpr(df, inputs["field"], inputs["threshold"])



    if expected["res"] is not None:
        assert np.isclose(res, expected["res"], rtol=1e-4) or res is None
    if expected["tag"] is not None:
        assert tag == expected["tag"]



---

import xgboost as xgb
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, monotonically_increasing_id
from pyspark.sql.types import FloatType, StructType, StructField
from typing import List, Tuple


class ShapMetricsSpark:
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[xgb.XGBClassifier, List[str]]:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        features = model.get_booster().feature_names
        return model, features
    
    @staticmethod
    def predict_scores(model, df: DataFrame, features: List[str], score_col: str = 'model_score') -> DataFrame:
        features_pd = df.select(features).toPandas()
        probas = model.predict_proba(features_pd)[:, 1]
        probas_array = [[float(p)] for p in probas]
        
        schema = StructType([StructField(score_col, FloatType(), False)])
        probas_df = df.sparkSession.createDataFrame(probas_array, schema)
        
        df_with_id = df.withColumn("_row_id", monotonically_increasing_id())
        probas_with_id = probas_df.withColumn("_row_id", monotonically_increasing_id())
        
        return df_with_id.join(probas_with_id, "_row_id").drop("_row_id")
    
    @staticmethod
    def shap_values(model_path: str, df: DataFrame, features: List[str] = None, 
                  top_pct: float = 0.05, n_rows: int = None, 
                  abs_val: bool = False, score_col: str = 'model_score', 
                  nthread: int = None) -> DataFrame:
        model, model_features = ShapMetricsSpark.load_model(model_path)
        features = features or model_features
        
        if score_col not in df.columns:
            df = ShapMetricsSpark.predict_scores(model, df, features, score_col)
        
        rows_to_keep = n_rows if n_rows else int(df.count() * top_pct)
        top_df = df.orderBy(F.col(score_col).desc()).limit(rows_to_keep)
        features_pd = top_df.select(features).toPandas()
        
        dmatrix = xgb.DMatrix(
            features_pd,
            missing=np.nan,
            nthread=nthread,
        )
        
        shap_contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
        if abs_val:
            shap_contributions = np.abs(shap_contributions)
        
        shap_values = shap_contributions[:, :-1]  # Remove bias column
        shap_df = pd.DataFrame(shap_values, columns=features)
        
        return df.sparkSession.createDataFrame(shap_df)
    
    @staticmethod
    def avg_shap_by_date(model_path: str, df: DataFrame, date_col: str,
                       features: List[str] = None, top_pct: float = 0.05, 
                       abs_val: bool = False, score_col: str = 'model_score') -> DataFrame:
        if features is None:
            _, features = ShapMetricsSpark.load_model(model_path)
        
        dates = [row[date_col] for row in df.select(date_col).distinct().collect()]
        result_rows = []
        
        for date in dates:
            date_df = df.filter(col(date_col) == date)
            shap_df = ShapMetricsSpark.shap_values(
                model_path, date_df, features, top_pct, None, abs_val, score_col
            )
            
            avg_values = shap_df.select([F.avg(c).alias(c) for c in features]).collect()[0]
            
            row_dict = {date_col: date}
            for feature in features:
                row_dict[feature] = float(avg_values[feature])
            
            result_rows.append(row_dict)
        
        return df.sparkSession.createDataFrame(result_rows)
    
    @staticmethod
    def conf_intervals(shap_df: DataFrame, date_col: str, start_date: str = None, 
                     end_date: str = None, conf_level: float = 2.0) -> DataFrame:
        filtered_df = shap_df
        if start_date and end_date:
            filtered_df = shap_df.filter(
                (F.col(date_col) >= start_date) & 
                (F.col(date_col) <= end_date)
            )
        
        feature_cols = [c for c in filtered_df.columns if c != date_col]
        result_rows = []
        
        for feature in feature_cols:
            stats = filtered_df.select(
                F.avg(feature).alias("mean"),
                F.stddev(feature).alias("std_dev")
            ).collect()[0]
            
            mean_val = float(stats["mean"])
            std_val = float(stats["std_dev"])
            lower_ci = mean_val - conf_level * std_val
            upper_ci = mean_val + conf_level * std_val
            
            result_rows.append({
                "feature": feature,
                "mean": mean_val,
                "std_dev": std_val,
                "lower_ci": lower_ci,
                "upper_ci": upper_ci
            })
        
        return shap_df.sparkSession.createDataFrame(result_rows)


---
from metrics_spark import MetricsSpark
from shap_metrics_spark import ShapMetricsSpark

# Use general metrics
bins = MetricsSpark.hist_bins(df, "score", 30)
psi_value = MetricsSpark.psi(hist_a, hist_b)

# Use SHAP-specific metrics
shap_values = ShapMetricsSpark.shap_values(model_path, df, top_pct=0.05)
