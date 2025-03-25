import pytest
import numpy as np
import math
from scipy.stats import kstest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, FloatType
from src.analysis.metrics_spark import MetricsSpark

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture(scope="module")
def spark_session():
    """Create a Spark session for testing"""
    spark = SparkSession.builder.appName("Test").master("local[4]").getOrCreate()
    yield spark
    spark.stop()

# -------------------------
# Shared Helper Functions
# -------------------------
def _gen_random_values(val_range=10, n=1000, offset=0, **kwargs):
    """Generate random values for testing with consistent seeds"""
    np.random.seed(0)
    values = np.random.uniform(-val_range, val_range, n) + offset
    labels = np.random.choice([0, 1], size=n)
    amt_values = np.random.uniform(2, 100, n)
    
    # For distribution metrics, generate a second set
    np.random.seed(1)  # Different seed for baseline
    values_b = np.random.uniform(-val_range, val_range, n) + offset + 0.2
    
    return {
        "values": values, 
        "labels": labels, 
        "amt_values": amt_values,
        "a": values,
        "b": values_b
    }

# -------------------------
# Classification Metrics Helpers
# -------------------------
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

# -------------------------
# Distribution Metrics Helpers
# -------------------------
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
    
    # Create dataframe for testing
    df = spark_session.createDataFrame([
        Row(score=val, score_label=label, amount=amt)
        for val, label, amt in zip(values, labels, amt_values)
    ])
    
    field = "score"
    threshold = args.get("threshold", 0.5)
    amt_field = "amount"
    
    # Calculate expected results
    if metric == "vdr":
        exp_res, exp_tag = _cal_vdr(df, field, threshold, amt_field)
    elif metric == "tdr":
        exp_res, exp_tag = _cal_tdr(df, field, threshold)
    elif metric == "tfpr":
        exp_res, exp_tag = _cal_tfpr(df, field, threshold)
    else:
        exp_res, exp_tag = None, None
    
    return {
        "case": case_name,
        "metric": metric,
        "inputs": {
            "df": df,
            "field": field,
            "threshold": threshold,
            "amt_field": amt_field
        },
        "expected": {"res": exp_res, "tag": exp_tag}
    }

def _gen_case_distribution(case_name, metric, func, args, spark_session):
    """Generate test cases for distribution metrics (PSI, KS)"""
    inputs = func(**args)
    x = [float(val) for val in inputs["a"]]
    x_bl = [float(val) for val in inputs["b"]]
    
    # Create dataframes
    field = "data"
    schema = StructType([StructField(field, FloatType(), True)])
    data_df = spark_session.createDataFrame([(val,) for val in x], schema)
    bl_df = spark_session.createDataFrame([(val,) for val in x_bl], schema)
    
    # Generate histograms
    nb = 10  # Standard number of bins
    bins = MetricsSpark.hist_bins(data_df, field, nb)
    hist_data = MetricsSpark.hist_density(data_df, field, bins)
    hist_bl = MetricsSpark.hist_density(bl_df, field, bins)
    
    # Calculate expected results
    if metric == "psi":
        monitored_props, reference_props = _cal_hist_proportions(x_bl, x, nb)
        exp_res = _cal_psi(monitored_props, reference_props)
    elif metric == "ks":
        exp_res = _cal_ks_test(x, x_bl)
    else:
        exp_res = None
    
    return {
        "case": case_name,
        "metric": metric,
        "inputs": {
            "hist_data": hist_data,
            "hist_bl": hist_bl,
            "df": data_df,
            "ref_df": bl_df,
            "field": field,
        },
        "expected": {"res": exp_res}
    }

def _gen_data(metric, spark_session):
    """Generate test cases for all metrics"""
    if metric in ["vdr", "tdr", "tfpr"]:
        # Classification metrics - use test cases from first developer
        test_cases = [
            ("all_positive_values", {"val_range": 10}),
            ("all_large_values", {"val_range": 1000000, "offset": 0.5}),
            ("all_zero_values", {"val_range": 10, "offset": 0.5}),
            ("single_class", {}, lambda: {
                "values": [1, 1, 1, 1],
                "labels": [1, 1, 1, 1],
                "amt_values": [10, 10, 10, 10],
                "a": [1, 1, 1, 1],
                "b": [1.2, 1.2, 1.2, 1.2]
            }),
        ]
        
        data = []
        for case_name, args, *custom_func in test_cases:
            func = custom_func[0] if custom_func else _gen_random_values
            data.append(_gen_case_classification(case_name, metric, func, args, spark_session))
        
    else:
        # Distribution metrics - use test cases from second developer
        test_cases = [
            ("all_positive_values", {}),
            ("all_negative_values", {"val_range": -10}),
            ("mixed_values", {"offset": 0.5}),
            ("large_values", {"val_range": 10000000, "offset": 0.5}),
            ("small_values", {"val_range": 0.00000001, "offset": 0.5})
        ]
        
        data = []
        for case_name, args in test_cases:
            data.append(_gen_case_distribution(case_name, metric, _gen_random_values, args, spark_session))
    
    return data

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def gen_data(request):
    return request.param

# -------------------------
# Test Generation
# -------------------------
def pytest_generate_tests(metafunc):
    if "gen_data" in metafunc.fixturenames:
        if metafunc.function.__name__ == "test_metrics":
            # Get spark session 
            spark = SparkSession.builder.appName("Test").master("local[4]").getOrCreate()
            
            # Generate all test cases
            metrics = ["vdr", "tdr", "tfpr", "psi", "ks"]
            all_data = []
            for metric in metrics:
                all_data.extend(_gen_data(metric, spark))
            
            # Create IDs for better test reporting
            ids = [f"{d['metric']}-{d['case']}" for d in all_data]
            metafunc.parametrize("gen_data", all_data, ids=ids)
            
            # Clean up the spark session
            spark.stop()

# -------------------------
# Unified Test Function
# -------------------------
def test_metrics(spark_session, gen_data):
    """Unified test function for all metrics"""
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
            assert res is None
        else:
            # For KS test, first element of array is the statistic
            if metric == "ks" and isinstance(expected_res, list):
                expected_res = expected_res[0]
            assert math.isclose(res, expected_res, rel_tol=1e-9)
