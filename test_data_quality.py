import pytest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from src.analysis.data_quality import validate_data_quality_spark
from test_metric_spark.commons import (
    _SEED_A,
    CornerCasesQuality,
    _gen_random_values,
    _gen_category_values,
)

VALUE_RANGE = "[0, 10]"
SIZE = 1000


@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder.appName("Test").master("local[*]").getOrCreate()
    yield spark
    spark.stop()


def _gen_case(case_name, metric, func, args=None, spark_session=None):
    """Generate test case with Spark DataFrame"""
    if args is None:
        inputs = func()
    else:
        inputs = func(**args)

    # Get values from inputs
    values = inputs.get("a", [])

    # Create appropriate schema based on data type
    if isinstance(values[0], (float, int)) if values else True:
        schema = StructType([StructField("val", FloatType(), True)])
        df = (
            spark_session.createDataFrame(
                [(float(val),) for val in values], schema
            ).cache()
            if values
            else spark_session.createDataFrame([], schema)
        )
        ftype = "numerical"
    else:
        schema = StructType([StructField("val", StringType(), True)])
        df = (
            spark_session.createDataFrame(
                [(str(val),) for val in values], schema
            ).cache()
            if values
            else spark_session.createDataFrame([], schema)
        )
        ftype = "categorical"

    return {
        "case": case_name,
        "inputs": {
            "df": df,
            "field": "val",
            "ftype": inputs.get("ftype", ftype),
            "val_range": (
                inputs.get("val_range", VALUE_RANGE)
                if ftype == "numerical"
                else inputs.get("cats")
            ),
        },
        "expected": {
            "r_mv": inputs.get("exp_mv"),
            "r_rv": inputs.get("exp_rv"),
            "tag": inputs.get("tag"),
        },
    }


def generate_test_cases(spark_session):
    """Generate all test cases"""
    np.random.seed(_SEED_A)

    data = [
        # Corner cases
        _gen_case(
            "empty_data",
            None,
            CornerCasesQuality.empty_data,
            spark_session=spark_session,
        ),
        _gen_case(
            "invalid_ftype",
            None,
            CornerCasesQuality.invalid_ftype,
            spark_session=spark_session,
        ),
        _gen_case(
            "mismatched_datatype",
            None,
            CornerCasesQuality.mismatched_datatype,
            spark_session=spark_session,
        ),
        # Numerical cases
        _gen_case(
            "numerical_missing_value",
            None,
            _gen_random_values,
            {"offset": 0.5, "add_nan": True, "exp_mv": 0.1},
            spark_session,
        ),
        _gen_case(
            "numerical_inf_value",
            None,
            _gen_random_values,
            {"offset": 0.5, "add_inf": True, "exp_rv": 0.1},
            spark_session,
        ),
        _gen_case(
            "numerical_range_violation",
            None,
            _gen_random_values,
            {"offset": 0.5, "exp_rv": 0.5},
            spark_session,
        ),
        # Categorical cases
        _gen_case(
            "categorical_missing_value",
            None,
            _gen_category_values,
            {"add_nan": True, "exp_mv": 0.1},
            spark_session,
        ),
        _gen_case(
            "categorical_rv_value",
            None,
            _gen_category_values,
            {"add_rv": True, "exp_rv": 0.1},
            spark_session,
        ),
        _gen_case(
            "categorical_range_violation",
            None,
            _gen_category_values,
            {"exp_rv": 0.0},
            spark_session,
        ),
    ]

    return data


@pytest.fixture
def gen_data(request):
    return request.param


def pytest_generate_tests(metafunc):
    if "gen_data" in metafunc.fixturenames:
        if metafunc.function.__name__ == "test_data_quality":
            spark = (
                SparkSession.builder.appName("Test").master("local[*]").getOrCreate()
            )
            all_data = generate_test_cases(spark)
            ids = [d["case"] for d in all_data]
            metafunc.parametrize("gen_data", all_data, ids=ids)
            spark.stop()


def test_data_quality(spark_session, gen_data):
    """Test data quality validation"""
    inputs = gen_data["inputs"]
    expected = gen_data["expected"]

    df = inputs["df"]
    field = inputs["field"]
    ftype = inputs["ftype"]
    val_range = inputs["val_range"]

    # Run validation
    r_mv, r_rv, df_clean = validate_data_quality_spark(df, field, ftype, val_range)

    # Check results
    if expected["r_mv"] is not None:
        assert np.isclose(r_mv, expected["r_mv"], rtol=1e-2)
    if expected["r_rv"] is not None:
        assert np.isclose(r_rv, expected["r_rv"], rtol=1e-2)
    if expected["tag"] is not None:
        assert df_clean is not None

    # Cleanup
    df.unpersist()
    if df_clean is not None:
        df_clean.unpersist()
