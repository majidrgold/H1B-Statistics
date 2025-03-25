import os
import pytest
import numpy as np
import math
from scipy.stats import kstest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, FloatType
from src.analysis.metrics_spark import MetricsSpark
from dotenv import load_dotenv

# Load environment variables (for Part 2)
load_dotenv(override=True)

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture(scope="module")
def spark_session():
    """Spark session fixture for Part 1 (classification metrics)"""
    spark = SparkSession.builder.appName("Test").master("local[4]").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def spark():
    """Spark session fixture for Part 2 (distribution metrics)"""
    spark = SparkSession.builder.appName("test").getOrCreate()
    yield spark
    spark.stop()

# -------------------------
# Helper Functions - Classification Metrics (Part 1)
# -------------------------
def _gen_random_values_classification(val_range=10, n=1000, offset=0, **kwargs):
    """Generate random values for classification metrics"""
    np.random.seed(0)
    values = np.random.uniform(-val_range, val_range, n) + offset
    labels = np.random.choice([0, 1], size=n)
    amt_values = np.random.uniform(2, 100, n)
    return {"values": values, "labels": labels, "amt_values": amt_values}

def _cal_vdr(df, field, threshold, amt_field):
    """Calculate VDR for validation"""
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
    
    tag = None if total_value != 0 else "NO_FRAUD_AMT"
    return hit_value / total_value if total_value != 0 else None, tag

def _cal_tdr(df, field, threshold):
    """Calculate TDR for validation"""
    tp_count = 0
    total_count = 0
    for row in df.collect():
        label = row[field + "_label"]
        score = row[field]
        if label == 1:
            total_count += 1
            if score >= threshold:
                tp_count += 1
    
    tag = None if total_count != 0 else "NO_FRAUD"
    return tp_count / total_count if total_count != 0 else None, tag

def _cal_tfpr(df, field, threshold):
    """Calculate TFPR for validation"""
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
    return fp_count / tp_count, None

def _gen_case_classification(case_name, metric, func, args, spark_session):
    """Generate test cases for classification metrics"""
    if callable(args):
        # Handle lambda case
        inputs = args()
    else:
        inputs = func(**args)
    
    values = [float(val) for val in inputs["values"]] if len(inputs["values"]) > 0 else []
    labels = [int(label) for label in inputs["labels"]] if len(inputs["labels"]) > 0 else []
    amt_values = [float(amt) for amt in inputs["amt_values"]] if len(inputs["amt_values"]) > 0 else []
    
    df = spark_session.createDataFrame(
        [
            Row(score=val, score_label=label, amount=amt)
            for val, label, amt in zip(values, labels, amt_values)
        ]
    ) if values else spark_session.createDataFrame([], schema=None)
    
    field = "score"
    threshold = args.get("threshold", 0.5) if isinstance(args, dict) else 0.5
    amt_field = "amount"
    
    if metric == "vdr":
        res, tag = _cal_vdr(df, field, threshold, amt_field)
    elif metric == "tdr":
        res, tag = _cal_tdr(df, field, threshold)
    elif metric == "tfpr":
        res, tag = _cal_tfpr(df, field, threshold)
    else:
        res, tag = None, None
    
    return {
        "case": case_name,
        "metric": metric,
        "inputs": {
            "df": df,
            "field": field,
            "threshold": threshold,
            "amt_field": amt_field
        },
        "expected": {"res": res, "tag": tag}
    }

def gen_data_classification(metric, spark_session):
    """Generate test data for classification metrics"""
    return [
        _gen_case_classification(
            "all positive values",
            metric,
            _gen_random_values_classification,
            {"val_range": 10},
            spark_session,
        ),
        _gen_case_classification(
            "all large values",
            metric,
            _gen_random_values_classification,
            {"val_range": 1000000, "offset": 0.5},
            spark_session,
        ),
        _gen_case_classification(
            "all zero values",
            metric,
            _gen_random_values_classification,
            {"val_range": 10, "offset": 0.5},
            spark_session,
        ),
        _gen_case_classification(
            "null entry values",
            metric,
            _gen_random_values_classification,
            lambda: {"values": np.array([]), "labels": np.array([]), "amt_values": np.array([])},
            spark_session,
        ),
        _gen_case_classification(
            "single class",
            metric,
            _gen_random_values_classification,
            lambda: {
                "values": np.array([1, 1, 1, 1]),
                "labels": np.array([1, 1, 1, 1]),
                "amt_values": np.array([10, 10, 10, 10]),
            },
            spark_session,
        ),
    ]

# -------------------------
# Helper Functions - Distribution Metrics (Part 2)
# -------------------------
def _cal_hist_proportions(reference, monitored, bins):
    """Calculate histogram proportions for distribution metrics"""
    n_nulls = np.sum(np.equal(monitored, None))
    r_nulls = np.sum(np.equal(reference, None))

    bin_edges = np.linspace(min(reference), max(reference), bins + 1)
    reference_hist, _ = np.histogram(reference, bins=bin_edges)
    monitored_hist, _ = np.histogram(monitored, bins=bin_edges)

    reference_proportions = reference_hist / len(reference)
    monitored_proportions = monitored_hist / len(monitored)

    return monitored_proportions, reference_proportions

def _cal_psi(monitored_proportions, reference_proportions):
    """Calculate PSI statistic"""
    psi_values = (monitored_proportions - reference_proportions) * np.log(
        monitored_proportions / reference_proportions
    )
    psi = np.sum(psi_values)
    return psi

def _cal_ks_test(x, x_bl):
    """Calculate KS test statistic"""
    ks_stat, p_value = kstest(x, x_bl)
    return [ks_stat, p_value]

def _gen_case_distribution(case_name, metric, func, args):
    """Generate test cases for distribution metrics"""
    # For part 2, we use the function from commons
    from .commons import _gen_random_values
    
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

def gen_data_distribution(metric):
    """Generate test data for distribution metrics"""
    # For part 2, we use the function from commons
    from .commons import _gen_random_values
    
    data = [
        _gen_case_distribution("all positive values", metric, _gen_random_values, {}),
        _gen_case_distribution(
            "all negative values", metric, _gen_random_values, {"val_range": -10}
        ),
        _gen_case_distribution("mixed values", metric, _gen_random_values, {"offset": 0.5}),
        _gen_case_distribution(
            "large values",
            metric,
            _gen_random_values,
            {"val_range": 10000000, "offset": 0.5},
        ),
        _gen_case_distribution(
            "small values",
            metric,
            _gen_random_values,
            {"val_range": 0.00000001, "offset": 0.5},
        ),
    ]
    return data

# Generate test data for distribution metrics
data_psi = gen_data_distribution("psi")
data_ks = gen_data_distribution("ks")

# -------------------------
# Fixtures for test parameters
# -------------------------
@pytest.fixture()
def gen_data_class(request):
    """Fixture for classification metrics tests"""
    return request.param

@pytest.fixture()
def gen_data_dist(request):
    """Fixture for distribution metrics tests"""
    return request.param["inputs"], request.param["expected"]

# -------------------------
# Tests - Classification Metrics (Part 1)
# -------------------------
@pytest.mark.parametrize(
    "gen_data_class", 
    gen_data_classification("vdr", spark_session()), 
    indirect=True, 
    ids=lambda x: f"vdr-{x['case']}"
)
def test_vdr(spark_session, gen_data_class):
    """Test VDR metric"""
    inputs = gen_data_class["inputs"]
    expected = gen_data_class["expected"]
    
    df = inputs["df"]
    field = inputs["field"]
    threshold = inputs["threshold"]
    amt_field = inputs["amt_field"]
    
    res, tag = MetricsSpark.vdr(df, field, threshold, amt_field)
    
    if expected["res"] is None:
        assert res is None
    else:
        assert math.isclose(res, expected["res"], rel_tol=1e-9)
    
    if expected["tag"] is not None:
        assert tag == expected["tag"]

@pytest.mark.parametrize(
    "gen_data_class", 
    gen_data_classification("tdr", spark_session()), 
    indirect=True, 
    ids=lambda x: f"tdr-{x['case']}"
)
def test_tdr(spark_session, gen_data_class):
    """Test TDR metric"""
    inputs = gen_data_class["inputs"]
    expected = gen_data_class["expected"]
    
    df = inputs["df"]
    field = inputs["field"]
    threshold = inputs["threshold"]
    
    res, tag = MetricsSpark.tdr(df, field, threshold)
    
    if expected["res"] is None:
        assert res is None
    else:
        assert math.isclose(res, expected["res"], rel_tol=1e-9)
    
    if expected["tag"] is not None:
        assert tag == expected["tag"]

@pytest.mark.parametrize(
    "gen_data_class", 
    gen_data_classification("tfpr", spark_session()), 
    indirect=True, 
    ids=lambda x: f"tfpr-{x['case']}"
)
def test_tfpr(spark_session, gen_data_class):
    """Test TFPR metric"""
    inputs = gen_data_class["inputs"]
    expected = gen_data_class["expected"]
    
    df = inputs["df"]
    field = inputs["field"]
    threshold = inputs["threshold"]
    
    res, tag = MetricsSpark.tfpr(df, field, threshold)
    
    if expected["res"] is None:
        assert res is None
    else:
        assert math.isclose(res, expected["res"], rel_tol=1e-9)
    
    if expected["tag"] is not None:
        assert tag == expected["tag"]

# -------------------------
# Tests - Distribution Metrics (Part 2)
# -------------------------
@pytest.mark.parametrize(
    "gen_data_dist", argvalues=data_psi, indirect=True, ids=[t["case"] for t in data_psi]
)
def test_hist_density(spark, gen_data_dist):
    """Test histogram density calculation"""
    inputs, expected = gen_data_dist
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
        assert abs(res - exp_res) <= 0.1

    for res, exp_res in zip(hist_bl, inputs["hist_bl"]):
        assert abs(res - exp_res) <= 0.1

@pytest.mark.parametrize(
    "gen_data_dist", argvalues=data_psi, indirect=True, ids=[t["case"] for t in data_psi]
)
def test_psi(gen_data_dist):
    """Test PSI metric"""
    inputs, expected = gen_data_dist
    res = MetricsSpark.psi(inputs["hist_data"], inputs["hist_bl"])
    if expected["res"] is None:
        assert res == expected["res"]
    else:
        assert abs(expected["res"] - res) < 0.0001

@pytest.mark.parametrize(
    "gen_data_dist", argvalues=data_ks, indirect=True, ids=[t["case"] for t in data_ks]
)
def test_ks(gen_data_dist):
    """Test KS metric"""
    inputs, expected = gen_data_dist
    res = MetricsSpark.ks(inputs["hist_data"][2:], inputs["hist_bl"][2:])
    if expected["res"] is None:
        assert res == expected["res"]
    else:
        assert abs(expected["res"][0] - res) < 0.0001
