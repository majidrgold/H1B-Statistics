import pytest
import numpy as np
import math
from scipy.stats import kstest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, FloatType

from src.analysis.metrics_spark import MetricsSpark
from test_metric_spark.commons import _SEED_A, _SEED_B

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder.appName("Test").master("local[*]").getOrCreate()
    yield spark
    spark.stop()

# -------------------------
# Helper Functions
# -------------------------
def _gen_random_values(val_range=10, n=1000, offset=0, **kwargs):
    """Generate random values for both classification and distribution metrics"""
    np.random.seed(_SEED_A)
    values = np.random.uniform(-val_range, val_range, n) + offset
    labels = np.random.choice([0, 1], size=n)
    amt_values = np.random.uniform(2, 100, n)
    
    # For distribution metrics (PSI, KS)
    np.random.seed(_SEED_B)
    values_b = np.random.uniform(-val_range, val_range, n) + offset + 0.2
    
    return {
        "values": values,
        "labels": labels,
        "amt_values": amt_values,
        "a": values,
        "b": values_b
    }

def _cal_hist_proportions(reference, monitored, bins):
    """Calculate histogram proportions for PSI and KS metrics"""
    bin_edges = np.linspace(min(reference), max(reference), bins + 1)
    reference_hist, _ = np.histogram(reference, bins=bin_edges)
    monitored_hist, _ = np.histogram(monitored, bins=bin_edges)
    
    reference_props = reference_hist / len(reference)
    monitored_props = monitored_hist / len(monitored)
    
    # Avoid division by zero
    monitored_props = np.where(monitored_props == 0, 1e-4, monitored_props)
    reference_props = np.where(reference_props == 0, 1e-4, reference_props)
    
    return monitored_props, reference_props

def _cal_psi(monitored_props, reference_props):
    """Calculate PSI"""
    return np.sum((monitored_props - reference_props) * np.log(monitored_props / reference_props))

def _cal_ks_test(x, x_bl):
    """Calculate KS statistic"""
    ks_stat, _ = kstest(x, x_bl)
    return ks_stat

# -------------------------
# Test Case Generators
# -------------------------
def _gen_case_classification(case_name, metric, func, args, spark_session):
    """Generate test cases for classification metrics (VDR, TDR, TFPR)"""
    inputs = func(**args)
    values = [float(val) for val in inputs["values"]]
    labels = [int(label) for label in inputs["labels"]]
    amt_values = [float(amt) for amt in inputs["amt_values"]]
    
    df = spark_session.createDataFrame([
        Row(score=val, score_label=label, amount=amt)
        for val, label, amt in zip(values, labels, amt_values)
    ])
    
    field = "score"
    threshold = args.get("threshold", 0.5)
    amt_field = "amount"
    
    return {
        "case": case_name,
        "metric": metric,
        "inputs": {
            "df": df,
            "field": field,
            "threshold": threshold,
            "amt_field": amt_field
        },
        "expected": {"res": None, "tag": args.get("tag")}
    }

def _gen_case_distribution(case_name, metric, func, args):
    """Generate test cases for distribution metrics (PSI, KS)"""
    inputs = func(**args)
    x, x_bl = inputs["a"], inputs["b"]
    
    nb = 10  # Standard number of bins
    hist_data, hist_bl = _cal_hist_proportions(x_bl, x, nb)
    
    if metric == "psi":
        exp_res = _cal_psi(hist_data[2:], hist_bl[2:])
    elif metric == "ks":
        exp_res = _cal_ks_test(x, x_bl)
    
    return {
        "case": case_name,
        "metric": metric,
        "inputs": {
            "hist_data": hist_data,
            "hist_bl": hist_bl,
            "x": x,
            "x_bl": x_bl,
            "nb": nb
        },
        "expected": {"res": exp_res}
    }

def _gen_data(metric, spark_session=None):
    """Generate standard test cases for all metrics"""
    test_cases = [
        ("all_positive_values", {}),
        ("all_negative_values", {"val_range": -10}),
        ("mixed_values", {"offset": 0.5}),
        ("large_values", {"val_range": 10000000, "offset": 0.5}),
        ("small_values", {"val_range": 0.00000001, "offset": 0.5})
    ]
    
    data = []
    for case_name, args in test_cases:
        if metric in ["vdr", "tdr", "tfpr"]:
            data.append(_gen_case_classification(case_name, metric, _gen_random_values, args, spark_session))
        else:
            data.append(_gen_case_distribution(case_name, metric, _gen_random_values, args))
    
    return data

# -------------------------
# Tests
# -------------------------
@pytest.mark.parametrize("metric", ["vdr", "tdr", "tfpr"])
def test_classification_metrics(spark_session, metric):
    """Test classification metrics (VDR, TDR, TFPR)"""
    for test_case in _gen_data(metric, spark_session):
        inputs = test_case["inputs"]
        
        if metric == "vdr":
            res, tag = MetricsSpark.vdr(inputs["df"], inputs["field"], 
                                      inputs["threshold"], inputs["amt_field"])
        elif metric == "tdr":
            res, tag = MetricsSpark.tdr(inputs["df"], inputs["field"], 
                                      inputs["threshold"])
        elif metric == "tfpr":
            res, tag = MetricsSpark.tfpr(inputs["df"], inputs["field"], 
                                       inputs["threshold"])
        
        if res is not None:
            assert math.isclose(res, test_case["expected"]["res"], rel_tol=1e-9)
        if "tag" in test_case["expected"]:
            assert tag == test_case["expected"]["tag"]

@pytest.mark.parametrize("metric", ["psi", "ks"])
def test_distribution_metrics(metric):
    """Test distribution metrics (PSI, KS)"""
    for test_case in _gen_data(metric):
        inputs = test_case["inputs"]
        expected = test_case["expected"]["res"]
        
        if metric == "psi":
            res = MetricsSpark.psi(inputs["hist_data"], inputs["hist_bl"])
        else:  # ks
            res = MetricsSpark.ks(inputs["hist_data"][2:], inputs["hist_bl"][2:])
        
        if expected is not None:
            assert math.isclose(res, expected, rel_tol=1e-9)
