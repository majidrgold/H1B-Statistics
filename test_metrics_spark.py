import os
import pytest
import numpy as np
from scipy.stats import kstest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, FloatType

from .commons import _gen_random_values, _gen_classification_values
from src.analysis.metrics_spark import MetricsSpark
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
    n_nulls = np.sum(monitored == None)
    r_nulls = np.sum(reference == None)

    bin_edges = np.linspace(min(reference), max(reference), bins + 1)
    reference_hist, _ = np.histogram(reference, bins=bin_edges)
    monitored_hist, _ = np.histogram(monitored, bins=bin_edges)

    reference_proportions = reference_hist / len(reference)
    monitored_proportions = monitored_hist / len(monitored)

    return monitored_proportions, reference_proportions


def _cal_psi(monitored_proportions, reference_proportions):
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


def _gen_case(case_name, metric, func, args, spark=None):
    if metric in ["vdr", "tdr", "tfpr"]:
        if spark is None:
            raise ValueError(
                f"Spark session required for classification metric '{metric}'"
            )

        if callable(args):
            inputs = args()
        else:
            inputs = func(**args)

        values = [float(val) for val in inputs["values"]]
        labels = [int(label) for label in inputs["labels"]]
        amt_values = [float(amt) for amt in inputs["amt_values"]]

        df = spark.createDataFrame(
            [
                Row(score=val, score_label=label, amount=amt)
                for val, label, amt in zip(values, labels, amt_values)
            ]
        )

        field = "score"
        threshold = args.get("threshold", 0.5) if isinstance(args, dict) else 0.5
        amt_field = "amount"

        if metric == "vdr":
            result = _cal_vdr(df, field, threshold, amt_field)
            if isinstance(result, tuple):
                res, tag = result  # Unpack tuple
            else:
                res, tag = result, None  # Single value
        elif metric == "tfpr":
            result = _cal_tfpr(df, field, threshold)
            if isinstance(result, tuple):
                res, tag = result  # Unpack tuple
            else:
                res, tag = result, None  # Single value
        elif metric == "tdr":
            res = _cal_tdr(df, field, threshold)
            tag = None
        else:
            res, tag = None, None

        return {
            "case": case_name,
            "metric": metric,
            "inputs": {
                "df": df,
                "field": field,
                "threshold": threshold,
                "amt_field": amt_field,
            },
            "expected": {
                "res": res,
                "tag": (
                    tag
                    if tag is not None
                    else args.get("tag") if isinstance(args, dict) else None
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


def _gen_data(metric, spark=None):
    if metric in ["vdr", "tdr", "tfpr"]:
        if spark is None:
            raise ValueError(
                f"Spark session required for classification metric '{metric}'"
            )

        return [
            _gen_case(
                "all positive values",
                metric,
                _gen_classification_values,
                {"val_range": 10},
                spark,
            ),
            _gen_case(
                "all large values",
                metric,
                _gen_classification_values,
                {"val_range": 1000000, "offset": 0.5},
                spark,
            ),
            _gen_case(
                "all zero values",
                metric,
                _gen_classification_values,
                {"val_range": 10, "offset": 0.5},
                spark,
            ),
            _gen_case(
                "null entry values",
                metric,
                lambda: {
                    "values": np.array([]),
                    "labels": np.array([]),
                    "amt_values": np.array([]),
                },
                spark,
            ),
            _gen_case(
                "single class",
                metric,
                lambda: {
                    "values": [1, 1, 1, 1],
                    "labels": [1, 1, 1, 1],
                    "amt_values": [10, 10, 10, 10],
                },
                spark,
            ),
        ]

    elif metric in ["psi", "ks"]:
        return [
            _gen_case("all positive values", metric, _gen_random_values, {}),
            _gen_case(
                "all negative values", metric, _gen_random_values, {"val_range": -10}
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


@pytest.fixture
def data_psi():
    return _gen_data("psi")


@pytest.fixture
def data_ks():
    return _gen_data("ks")


@pytest.fixture
def data_vdr(spark):
    return _gen_data("vdr", spark)


@pytest.fixture
def data_tdr(spark):
    return _gen_data("tdr", spark)


@pytest.fixture
def data_tfpr(spark):
    return _gen_data("tfpr", spark)


@pytest.fixture()
def gen_data(request):
    return request.param["inputs"], request.param["expected"]


@pytest.mark.parametrize(
    "gen_data",
    lambda request: request.getfixturevalue("data_psi"),
    indirect=True,
    ids=lambda t: t["case"],
)
def test_hist_density(spark, gen_data):
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
    "gen_data",
    lambda request: request.getfixturevalue("data_psi"),
    indirect=True,
    ids=lambda t: t["case"],
)
def test_psi(gen_data):
    inputs, expected = gen_data
    res = MetricsSpark.psi(inputs["hist_data"], inputs["hist_bl"])
    if expected["res"] is None:
        assert res is None
    else:
        assert np.isclose(res, expected["res"], rtol=1e-4)


@pytest.mark.parametrize(
    "gen_data",
    lambda request: request.getfixturevalue("data_ks"),
    indirect=True,
    ids=lambda t: t["case"],
)
def test_ks(gen_data):
    inputs, expected = gen_data
    res = MetricsSpark.ks(inputs["hist_data"][2:], inputs["hist_bl"][2:])
    if expected["res"] is None:
        assert res is None
    else:
        assert np.isclose(res, expected["res"][0], rtol=1e-4)


@pytest.mark.parametrize(
    "gen_data",
    lambda request: request.getfixturevalue("data_vdr"),
    indirect=True,
    ids=lambda t: t["case"],
)
def test_vdr(spark, gen_data):
    inputs, expected = gen_data

    df = inputs["df"]
    field = inputs["field"]
    threshold = inputs["threshold"]
    amt_field = inputs["amt_field"]

    res, tag = MetricsSpark.vdr(df, field, threshold, amt_field)

    if expected["res"] is None:
        assert res is None
    else:
        assert np.isclose(res, expected["res"], rtol=1e-4)

    if "tag" in expected and expected["tag"] is not None:
        assert tag == expected["tag"]


@pytest.mark.parametrize(
    "gen_data",
    lambda request: request.getfixturevalue("data_tdr"),
    indirect=True,
    ids=lambda t: t["case"],
)
def test_tdr(spark, gen_data):
    inputs, expected = gen_data

    df = inputs["df"]
    field = inputs["field"]
    threshold = inputs["threshold"]

    res, tag = MetricsSpark.tdr(df, field, threshold)

    if expected["res"] is None:
        assert res is None
    else:
        assert np.isclose(res, expected["res"], rtol=1e-4)

    if "tag" in expected and expected["tag"] is not None:
        assert tag == expected["tag"]


@pytest.mark.parametrize(
    "gen_data",
    lambda request: request.getfixturevalue("data_tfpr"),
    indirect=True,
    ids=lambda t: t["case"],
)
def test_tfpr(spark, gen_data):
    inputs, expected = gen_data

    df = inputs["df"]
    field = inputs["field"]
    threshold = inputs["threshold"]

    res, tag = MetricsSpark.tfpr(df, field, threshold)

    if expected["res"] is None:
        assert res is None
    else:
        assert np.isclose(res, expected["res"], rtol=1e-4)

    if "tag" in expected and expected["tag"] is not None:
        assert tag == expected["tag"]
