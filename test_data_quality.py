import pytest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from pyspark.sql.functions import col, isnan
from src.analysis.data_quality import validate_data_quality_spark
from test_metric_spark.commons import (
    _SEED_A,
    CornerCasesQuality,
    _gen_random_values,
    _gen_category_values,
)

VALUE_RANGE = "[0, 10]"
SIZE = 1000


@pytest.fixture(scope="session")
def spark_session():
    """Create a session-scoped Spark session"""
    spark = SparkSession.builder.appName("Test").master("local[*]").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def test_cases(spark_session):
    """Generate all test cases once per session"""
    np.random.seed(_SEED_A)
    return generate_test_cases(spark_session)


def _gen_case(case_name, func, args=None, spark_session=None):
    """Generate test case with Spark DataFrame"""
    try:
        if args is None:
            inputs = func()
        else:
            inputs = func(**args)

        # Get values from inputs
        values = inputs.get("a", [])

        # Set default ftype based on input data
        if values and isinstance(values[0], (float, int)):
            schema = StructType([StructField("val", FloatType(), True)])
            df = spark_session.createDataFrame(
                [(float(val),) for val in values], schema
            ).cache()
            ftype = "numerical"
        else:
            schema = StructType([StructField("val", StringType(), True)])
            if values:
                df = spark_session.createDataFrame(
                    [(str(val),) for val in values], schema
                ).cache()
            else:
                df = spark_session.createDataFrame([], schema).cache()
            ftype = "categorical"

        # Override ftype if provided in inputs
        ftype = inputs.get("ftype", ftype)

        # Determine value range based on ftype
        if ftype == "numerical":
            val_range = inputs.get("val_range", VALUE_RANGE)
        else:
            val_range = inputs.get("cats")

        return {
            "case": case_name,
            "inputs": {
                "df": df,
                "field": "val",
                "ftype": ftype,
                "val_range": val_range,
            },
            "expected": {
                "r_mv": inputs.get("exp_mv"),
                "r_rv": inputs.get("exp_rv"),
                "should_fail": inputs.get("should_fail", False),
            },
        }
    except Exception as e:
        raise ValueError(f"Error generating test case '{case_name}': {str(e)}")


def generate_test_cases(spark_session):
    """Generate all test cases"""
    # Removed redundant seed setting

    data = [
        # Corner cases
        _gen_case(
            "empty_data",
            CornerCasesQuality.empty_data,
            spark_session=spark_session,
        ),
        _gen_case(
            "invalid_ftype",
            CornerCasesQuality.invalid_ftype,
            spark_session=spark_session,
        ),
        _gen_case(
            "mismatched_datatype",
            CornerCasesQuality.mismatched_datatype,
            spark_session=spark_session,
        ),
        # Numerical cases
        _gen_case(
            "numerical_missing_value",
            _gen_random_values,
            {"offset": 0.5, "add_nan": True, "exp_mv": 0.1},
            spark_session,
        ),
        _gen_case(
            "numerical_inf_value",
            _gen_random_values,
            {"offset": 0.5, "add_inf": True, "exp_rv": 0.1},
            spark_session,
        ),
        _gen_case(
            "numerical_range_violation",
            _gen_random_values,
            {"offset": 0.5, "exp_rv": 0.5},
            spark_session,
        ),
        # Categorical cases
        _gen_case(
            "categorical_missing_value",
            _gen_category_values,
            {"add_nan": True, "exp_mv": 0.1},
            spark_session,
        ),
        _gen_case(
            "categorical_rv_value",
            _gen_category_values,
            {"add_rv": True, "exp_rv": 0.1},
            spark_session,
        ),
        _gen_case(
            "categorical_range_violation",
            _gen_category_values,
            {"exp_rv": 0.0},
            spark_session,
        ),
    ]

    return data


def pytest_generate_tests(metafunc):
    """Generate tests dynamically"""
    if "gen_data" in metafunc.fixturenames:
        if metafunc.function.__name__ == "test_data_quality":
            test_cases = metafunc.getfixturevalue("test_cases")
            ids = [d["case"] for d in test_cases]
            metafunc.parametrize("gen_data", test_cases, ids=ids)


@pytest.fixture
def gen_data(request):
    """Fixture to get test data"""
    return request.param


def test_data_quality(spark_session, gen_data):
    """Test data quality validation"""
    inputs = gen_data["inputs"]
    expected = gen_data["expected"]

    df = inputs["df"]
    field = inputs["field"]
    ftype = inputs["ftype"]
    val_range = inputs["val_range"]
    df_clean = None

    try:
        # Handle empty DataFrame case which could cause division by zero
        if df.count() == 0 and not expected.get("should_fail", False):
            # Skip testing or use a different assertion for empty dataframes
            # Here we're expecting the function to handle empty dataframes gracefully
            r_mv, r_rv, df_clean = validate_data_quality_spark(
                df, field, ftype, val_range
            )
            # For empty dataframes we expect both rates to be 0
            assert (
                r_mv == 0 and r_rv == 0
            ), "Empty DataFrame should result in zero rates"
            return

        # Run validation with proper exception handling
        try:
            r_mv, r_rv, df_clean = validate_data_quality_spark(
                df, field, ftype, val_range
            )
        except Exception as e:
            if expected.get("should_fail", False):
                # If we expect this test to fail, we pass
                pytest.skip(f"Expected failure: {str(e)}")
                return
            else:
                # Otherwise, re-raise the exception
                raise

        # Check results
        if expected["r_mv"] is not None:
            assert np.isclose(
                r_mv, expected["r_mv"], rtol=1e-2
            ), f"Missing value rate mismatch: got {r_mv}, expected {expected['r_mv']}"

        if expected["r_rv"] is not None:
            assert np.isclose(
                r_rv, expected["r_rv"], rtol=1e-2
            ), f"Range violation rate mismatch: got {r_rv}, expected {expected['r_rv']}"

        # Verify the DataFrame content if available
        if df_clean is not None and df_clean.count() > 0:
            # For numerical data, verify no nulls or NaNs remain
            if ftype == "numerical":
                null_count = df_clean.filter(
                    df_clean[field].isNull() | isnan(col(field))
                ).count()
                assert null_count == 0, "Clean DataFrame still has null/NaN values"

            # For categorical data with a range, verify all values are in range
            if ftype == "categorical" and val_range is not None:
                invalid_count = df_clean.filter(
                    ~df_clean[field].isin(val_range)
                ).count()
                assert (
                    invalid_count == 0
                ), "Clean DataFrame contains values outside expected range"

    finally:
        # Always clean up with better exception handling
        if df is not None:
            try:
                df.unpersist()
            except Exception as e:
                print(f"Warning: Failed to unpersist input DataFrame: {str(e)}")

        if df_clean is not None:
            try:
                df_clean.unpersist()
            except Exception as e:
                print(f"Warning: Failed to unpersist clean DataFrame: {str(e)}")
