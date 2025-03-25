import pytest
import numpy as np
import math
from scipy.stats import kstest
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, FloatType
from src.analysis.metrics_spark import MetricsSpark

# Import common utilities and constants
from test_metric_spark.commons import _SEED_A, _SEED_B, CornerCases


@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder.appName("Test").master("local[*]").getOrCreate()
    yield spark
    spark.stop()


def _gen_random_values(val_range=10, n=1000, offset=0, **kwargs):
    """Generate random values for testing purposes"""
    np.random.seed(_SEED_A)
    values = np.random.uniform(-val_range, val_range, n) + offset
    labels = np.random.choice([0, 1], size=n)
    amt_values = np.random.uniform(2, 100, n)

    # For distribution metrics (PSI, KS), generate a second set with a slight shift
    np.random.seed(_SEED_B)
    values_b = np.random.uniform(-val_range, val_range, n) + offset + 0.2

    return {
        "values": values,
        "labels": labels,
        "amt_values": amt_values,
        "a": values,
        "b": values_b,
    }


# ===== Calculation helpers for classification metrics =====
def _cal_vdr(df, field, threshold, amt_field):
    """Calculate Value Detection Rate for validation"""
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
    """Calculate True Detection Rate for validation"""
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
    """Calculate True-False Positive Ratio for validation"""
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


# ===== Calculation helpers for distribution metrics =====
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
    """Calculate Population Stability Index for validation"""
    psi = np.sum(
        (monitored_props - reference_props) * np.log(monitored_props / reference_props)
    )
    return psi


def _cal_ks_test(x, x_bl):
    """Calculate Kolmogorov-Smirnov statistic for validation"""
    ks_stat, _ = kstest(x, x_bl)
    return ks_stat


def validate_test_inputs(values, labels, amt_values=None):
    """Validate test input data"""
    if not len(values) == len(labels):
        raise ValueError("Values and labels must have same length")
    if amt_values is not None and len(values) != len(amt_values):
        raise ValueError("Amount values must have same length as values")
    if not all(isinstance(l, (int, np.integer)) for l in labels):
        raise ValueError("Labels must be integers")


# ===== Test case generators =====
def _gen_case_classification(case_name, metric, func, args, spark_session):
    """Generate test cases for classification metrics"""
    inputs = func(**args)
    values = [float(val) for val in inputs["values"]]
    labels = [int(label) for label in inputs["labels"]]
    amt_values = [float(amt) for amt in inputs["amt_values"]]

    validate_test_inputs(values, labels, amt_values)

    df = spark_session.createDataFrame(
        [
            Row(score=val, score_label=label, amount=amt)
            for val, label, amt in zip(values, labels, amt_values)
        ]
    ).cache()

    field = "score"
    threshold = args.get("threshold", 0.5)
    amt_field = "amount"

    if metric == "vdr":
        expected_res, expected_tag = _cal_vdr(df, field, threshold, amt_field)
    elif metric == "tdr":
        expected_res, expected_tag = _cal_tdr(df, field, threshold)
    elif metric == "tfpr":
        expected_res, expected_tag = _cal_tfpr(df, field, threshold)
    else:
        expected_res, expected_tag = None, None

    df.unpersist()  # Clean up DataFrame
    return {
        "case": case_name,
        "metric": metric,
        "inputs": {
            "df": df,
            "field": field,
            "threshold": threshold,
            "amt_field": amt_field if metric == "vdr" else None,
        },
        "expected": {"res": expected_res, "tag": expected_tag},
    }


def _gen_case_distribution(case_name, metric, func, args, spark_session):
    """Generate test cases for distribution metrics (PSI, KS)"""
    inputs = func(**args)
    x = [float(val) for val in inputs["a"]]
    x_baseline = [float(val) for val in inputs["b"]]

    # Validate inputs
    if len(x) == 0 or len(x_baseline) == 0:
        raise ValueError("Empty input data")

    # Create dataframes
    schema = StructType([StructField("val", FloatType(), True)])
    data_df = spark_session.createDataFrame([(val,) for val in x], schema).cache()
    ref_df = spark_session.createDataFrame(
        [(val,) for val in x_baseline], schema
    ).cache()

    bins = 10  # Standard number of bins

    # Calculate histograms
    hist_data = MetricsSpark.hist_density(data_df, "val", bins)
    hist_bl = MetricsSpark.hist_density(ref_df, "val", bins)

    # Calculate expected results
    if metric == "psi":
        monitored_props, reference_props = _cal_hist_proportions(x, x_baseline, bins)
        expected_res = _cal_psi(monitored_props, reference_props)
    elif metric == "ks":
        expected_res = _cal_ks_test(x, x_baseline)

    # Cleanup
    data_df.unpersist()
    ref_df.unpersist()

    return {
        "case": case_name,
        "metric": metric,
        "inputs": {
            "hist_data": hist_data,
            "hist_bl": hist_bl,
            "df": data_df,
            "ref_df": ref_df,
            "field": "val",
        },
        "expected": {"res": expected_res},
    }


def _gen_data(metric, spark_session):
    """Generate standard set of test cases for all metrics"""
    if metric in ["vdr", "tdr", "tfpr"]:
        # Classification metrics
        gen_func = _gen_case_classification
    else:
        # Distribution metrics
        gen_func = _gen_case_distribution

    data = [
        gen_func("all_positive_values", metric, _gen_random_values, {}, spark_session),
        gen_func(
            "all_negative_values",
            metric,
            _gen_random_values,
            {"val_range": -10},
            spark_session,
        ),
        gen_func(
            "mixed_values", metric, _gen_random_values, {"offset": 0.5}, spark_session
        ),
        gen_func(
            "large_values",
            metric,
            _gen_random_values,
            {"val_range": 10000000, "offset": 0.5},
            spark_session,
        ),
        gen_func(
            "small_values",
            metric,
            _gen_random_values,
            {"val_range": 0.00000001, "offset": 0.5},
            spark_session,
        ),
    ]

    return data


@pytest.fixture
def gen_data(request):
    return request.param


# Parameterization for all metrics
def pytest_generate_tests(metafunc):
    if "gen_data" in metafunc.fixturenames:
        if metafunc.function.__name__ == "test_metrics":
            # Get spark session fixture
            spark = metafunc.getfixturevalue("spark_session")

            # Generate test cases for all metrics
            metrics = ["vdr", "tdr", "tfpr", "psi", "ks"]
            all_data = []
            for metric in metrics:
                all_data.extend(_gen_data(metric, spark))

            ids = [f"{d['metric']}-{d['case']}" for d in all_data]
            metafunc.parametrize("gen_data", all_data, ids=ids)


# Single unified test function for all metrics
def test_metrics(spark_session, gen_data):
    """Test both classification and distribution metrics"""
    metric = gen_data["metric"]
    inputs = gen_data["inputs"]
    expected = gen_data["expected"]

    # Classification metrics
    if metric in ["vdr", "tdr", "tfpr"]:
        df = inputs["df"]
        field = inputs["field"]
        threshold = inputs["threshold"]

        if metric == "vdr":
            amt_field = inputs["amt_field"]
            res, tag = MetricsSpark.vdr(df, field, threshold, amt_field)
        elif metric == "tdr":
            res, tag = MetricsSpark.tdr(df, field, threshold)
        elif metric == "tfpr":
            res, tag = MetricsSpark.tfpr(df, field, threshold)

        # Assert results
        if expected["res"] is None:
            assert res is None
        else:
            assert math.isclose(res, expected["res"], rel_tol=1e-9)

        # Assert tags
        if "tag" in expected and expected["tag"] is not None:
            assert tag == expected["tag"]

    # Distribution metrics
    elif metric in ["psi", "ks"]:
        hist_data = inputs["hist_data"]
        hist_bl = inputs["hist_bl"]
        expected_res = expected["res"]

        if metric == "psi":
            res = MetricsSpark.psi(hist_data, hist_bl)
        elif metric == "ks":
            # Exclude first two bins (poor and null bins) for KS test
            res = MetricsSpark.ks(hist_data[2:], hist_bl[2:])

        # Assert results
        if expected_res is None:
            assert res is None or np.isnan(res)
        else:
            assert math.isclose(res, expected_res, rel_tol=1e-9)
