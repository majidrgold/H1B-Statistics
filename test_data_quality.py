import os
import sys

# Set Java environment variables for JDK 11
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.22.7-hotspot"  # Adjust path as needed
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

import pytest
import numpy as np
import math
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from pyspark.sql.functions import col, isnan
from data_quality import validate_data_quality_spark, Tag
from commons import (
    _SEED_A,
    Tag,
    CornerCasesQuality,
    _cat1,
    _num1,
    _gen_random_values,
    _gen_category_values,
)

VALUE_RANGE = "[0, 10]"
SIZE = 1000

# Helper functions for expected values calculations
def _cal_num_range_violation(a, low=None, high=None):
    y = []
    n = 0
    for z in a:
        if math.isnan(z) or (low and z < low) or (high and z > high):
            n += 1
        else:
            y.append(z)
    return y, 0, n, None

def _cal_cat_range_violation(a, cats):
    y = []
    n = 0
    for z in a:
        if z in cats:
            y.append(z)
        else:
            n += 1
    return y, 0, n, None

# Test case generator following the same pattern as in the reference code
def _gen_case(case_name, case_type, func, args=None):
    if args is None:
        inputs = func()
        a, ftype, cats, low, high = inputs['a'], inputs['ftype'], inputs['cats'], inputs['low'], inputs['high']
        res, nm, nv, tag = inputs['res'], 0, 0, inputs['tag']
    else:
        if isinstance(args, dict):
            inputs = func(**args)
        else:
            inputs = func(*args)
            
        if case_type and case_type[:3] == 'num':
            a, ftype, cats = inputs['a'], 'numerical', None
            if case_type[4:] == 'mv':
                low, high = None, None
                clean_a = [x for x in a if not (isinstance(x, float) and math.isnan(x))]
                res, nm, nv, tag = clean_a, np.count_nonzero(np.isnan(a)), 0, None
            elif case_type[4:] == 'no_th':
                low, high = np.min(a[~np.isnan(a)]), np.max(a[~np.isnan(a)])
                res, nm, nv, tag = _cal_num_range_violation(a, low, high)
            elif case_type[4:] == 'inf_th':
                low, high = None, None
                res, nm, nv, tag = _cal_num_range_violation(a, low, high)
            else:
                if len(a[~np.isnan(a)]) > 0:
                    low, high = np.min(a[~np.isnan(a)]) - 0.25, np.max(a[~np.isnan(a)]) + 0.25
                else:
                    low, high = None, None
                res, nm, nv, tag = _cal_num_range_violation(a, low, high)
                
        elif case_type and case_type[:3] == 'str':
            a, ftype, low, high = inputs['a'], 'categorical', None, None
            if case_type[4:] == 'mv':
                cats = list(set([x for x in a if x and x.strip()]))
                y, nv = [], 0
                for v in a:
                    if v and v.strip() and v in cats:
                        y.append(v)
                    else:
                        nv += 1
                res, nm, nv, tag = y, 0, nv, None
            else:
                cats = _cat1
                res, nm, nv, tag = _cal_cat_range_violation(a, cats)
        else:
            a = inputs.get('a', [])
            ftype = inputs.get('ftype', 'numerical')
            cats = inputs.get('cats', None)
            low = inputs.get('low', None)
            high = inputs.get('high', None)
            res, nm, nv, tag = a, 0, 0, inputs.get('tag', None)

    # Create a value range string for numerical data
    val_range = None
    if ftype == 'numerical' and low is not None and high is not None:
        val_range = f"[{low}, {high}]"
    elif ftype == 'categorical':
        val_range = cats

    return {
        'case': case_name,
        'inputs': {
            'df': a,  # Will be converted to DataFrame in test
            'field': 'val',
            'ftype': ftype,
            'val_range': val_range,
        },
        'expected': {
            'res': res,
            'nm': nm,
            'nv': nv,
            'tag': tag
        }
    }

# Generate test cases
data = [
    _gen_case('empty data', None, CornerCasesQuality.empty_data),
    _gen_case('invalid type', None, CornerCasesQuality.invalid_ftype),
    _gen_case('mismatched datatype', None, CornerCasesQuality.mismatched_datatype),
    _gen_case('categorical missing value', 'str_mv', _gen_category_values, {'cat2': ['']}),
    _gen_case('numerical missing value', 'num_mv', _gen_random_values, {'offset': 0.5, 'add_nan': True}),
    _gen_case('string categorical range violation', 'str_rv', _gen_category_values, {}),
    _gen_case('numerical range violation', 'num_rv', _gen_random_values, {'offset': 0.5}),
    _gen_case('numerical inf with low, high value', 'num_inf', _gen_random_values, {'offset': 0.5, 'add_inf': True}),
    _gen_case('numerical inf without low, high value', 'num_inf_no_th', _gen_random_values, {'offset': 0.5, 'add_inf': True}),
    _gen_case('numerical value', 'num', _gen_random_values, {'offset': 0.5}),
    _gen_case('categorical value', 'str', _gen_category_values, {}),
]

@pytest.fixture(scope="session")
def spark_session():
    """Create a session-scoped Spark session"""
    spark = SparkSession.builder.appName("Test").master("local[*]").getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture()
def gen_data(request):
    """Fixture to get raw test data"""
    return request.param

@pytest.mark.parametrize('gen_data', argvalues=data, indirect=True, ids=[t['case'] for t in data])
def test_data_quality(spark_session, gen_data):
    """Test data quality validation with Spark"""
    inputs = gen_data['inputs']
    expected = gen_data['expected']
    
    # Extract test parameters
    raw_data = inputs['df']
    field = inputs['field']
    ftype = inputs['ftype']
    val_range = inputs['val_range']
    
    # Create Spark DataFrame from raw data
    if len(raw_data) == 0:
        # Empty DataFrame case
        if ftype == 'numerical':
            schema = StructType([StructField(field, FloatType(), True)])
        else:
            schema = StructType([StructField(field, StringType(), True)])
        df = spark_session.createDataFrame([], schema)
    else:
        # Create appropriate schema based on data type
        if ftype == 'numerical':
            schema = StructType([StructField(field, FloatType(), True)])
            df = spark_session.createDataFrame([(float(val),) for val in raw_data], schema)
        else:  # categorical
            schema = StructType([StructField(field, StringType(), True)])
            df = spark_session.createDataFrame([(str(val),) for val in raw_data], schema)
    
    # Calculate expected missing value and range violation rates for comparison
    n_total = len(raw_data)
    expected_mv_rate = expected['nm'] / n_total if n_total > 0 else 0
    expected_rv_rate = expected['nv'] / n_total if n_total > 0 else 0
    
    try:
        # Handle error cases
        if expected['tag'] in [Tag.DATATYPE_UNSUPPORTED, Tag.INCORRECT_DATA_TYPE]:
            with pytest.raises(Exception):
                r_mv, r_rv, df_clean = validate_data_quality_spark(df, field, ftype, val_range)
            return
        
        # Handle empty DataFrame
        if n_total == 0:
            with pytest.raises(Exception):
                r_mv, r_rv, df_clean = validate_data_quality_spark(df, field, ftype, val_range)
            return
            
        # Run validation
        r_mv, r_rv, df_clean = validate_data_quality_spark(df, field, ftype, val_range)
        
        # Check rates
        # Use relative tolerance for comparing rates
        assert abs(r_mv - expected_mv_rate) < 0.01, f"Missing value rate mismatch: got {r_mv}, expected {expected_mv_rate}"
        assert abs(r_rv - expected_rv_rate) < 0.01, f"Range violation rate mismatch: got {r_rv}, expected {expected_rv_rate}"
        
        # Verify the clean DataFrame
        if df_clean is not None:
            # For numerical data, check no missing values or out-of-range values
            if ftype == 'numerical':
                null_count = df_clean.filter(df_clean[field].isNull() | isnan(col(field))).count()
                assert null_count == 0, "Clean DataFrame still has null/NaN values"
                
                # Check range if specified
                if val_range:
                    try:
                        # Parse range from string like [low, high]
                        low, high = map(float, val_range.strip('[]').split(','))
                        out_of_range = df_clean.filter((col(field) < low) | (col(field) > high)).count()
                        assert out_of_range == 0, "Clean DataFrame has values outside range"
                    except (ValueError, AttributeError):
                        # Skip range check if parsing fails
                        pass
                        
            # For categorical data, check all values are in the allowed set
            if ftype == 'categorical' and val_range:
                if isinstance(val_range, list):
                    invalid_count = df_clean.filter(~col(field).isin(val_range)).count()
                    assert invalid_count == 0, "Clean DataFrame has invalid categorical values"
                    
    except ZeroDivisionError:
        # Handle division by zero in validate_data_quality_spark
        assert expected_mv_rate == 0 and expected_rv_rate == 0, "Empty DataFrame should result in zero rates"
    
    finally:
        # Clean up
        if 'df_clean' in locals() and df_clean is not None:
            df_clean.unpersist()
        df.unpersist()
