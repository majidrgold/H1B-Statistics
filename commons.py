import random
import numpy as np
from src.analysis.constants import Tag

_SEED_A = 42
_SEED_B = 43

_cat1 = ["a", "b", "c", "d", "e", "f", "g", "h"]
_cat2 = ["i", "j", "k"]
_num1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_num2 = [13, 14]


def _gen_random_values(val_range=10, n=1000, offset=0, **kwargs):
    # acceptable arguments:
    # - offset2 (int/float)
    # - add_weight (bool)
    # - add_nan / add_nan2 (bool)
    # - add_inf / add_inf2 (bool)
    # - change_type (bool)
    # - p (custom payload)

    np.random.seed(_SEED_A)
    n = kwargs.get("n", n)
    a = val_range * (np.random.rand(n) - offset)

    np.random.seed(_SEED_B)
    offset2 = kwargs.get("offset2", None)
    n2 = kwargs.get("n2", n)
    if offset2 is not None:
        b = val_range * (np.random.rand(n2) - offset2)
    else:
        b = val_range * (np.random.rand(n2) - offset)

    add_weight = kwargs.get("add_weight", False)
    w = np.random.rand(n) if add_weight else None

    # Add NaNs and Infs
    if kwargs.get("add_nan", False):
        r_nan = kwargs.get("r_nan", 0.1)
        loc = list(range(n))
        random.shuffle(loc)
        a[: int(n * r_nan)] = np.nan

    if kwargs.get("add_nan2", False):
        r_nan2 = kwargs.get("r_nan2", 0.1)
        loc = list(range(n2))
        random.shuffle(loc)
        b[: int(n2 * r_nan2)] = np.nan

    if kwargs.get("add_inf", False):
        r_inf = kwargs.get("r_inf", 0.1)
        loc = list(range(n))
        random.shuffle(loc)
        a[: int(n * r_inf)] = np.inf

    if kwargs.get("add_inf2", False):
        r_inf2 = kwargs.get("r_inf2", 0.1)
        loc = list(range(n2))
        random.shuffle(loc)
        b[: int(n2 * r_inf2)] = np.inf

    if kwargs.get("change_type", False):
        a = [str(i) for i in a]

    p = kwargs.get("p", None)
    result = {"a": a, "b": b, "w": w, "p": p}

    return result


def _gen_category_values(cat1=_cat1, cat2=None, n=1000, **kwargs):
    np.random.seed(_SEED_A)
    prob = kwargs.get("prob", None)
    a = random.choices(cat1, weights=prob, k=n) if prob else random.choices(cat1, k=n)

    n2 = kwargs.get("n2", n)
    overlap = kwargs.get("overlap", True)
    if cat2 is None:
        b = a.copy()
    else:
        b = random.choices(cat1 + cat2, k=n2) if overlap else random.choices(cat2, k=n2)

    # Add NaNs
    if kwargs.get("add_nan", False):
        r_nan = kwargs.get("r_nan", 0.1)
        loc = list(range(n))
        random.shuffle(loc)
        for i in loc[: int(n * r_nan)]:
            a[i] = ""

    if kwargs.get("add_nan2", False):
        r_nan2 = kwargs.get("r_nan2", 0.1)
        loc = list(range(n2))
        random.shuffle(loc)
        for i in loc[: int(n2 * r_nan2)]:
            b[i] = ""

    if kwargs.get("add_rv", False):
        r_rv = kwargs.get("r_rv", 0.1)
        loc = list(range(n))
        random.shuffle(loc)
        for i in loc[: int(n * r_rv)]:
            a[i] = "rv"

    if kwargs.get("add_rv2", False):
        r_rv2 = kwargs.get("r_rv2", 0.1)
        loc = list(range(n2))
        random.shuffle(loc)
        for i in loc[: int(n2 * r_rv2)]:
            b[i] = "rv"

    return {"a": a, "b": b, "overlap": overlap}


def _gen_classification_values(val_range=10, n=1000, offset=0, **kwargs):

    pos_ratio = kwargs.get("pos_ratio", 0.5)
    amt_min = kwargs.get("amt_min", 2.0)
    amt_max = kwargs.get("amt_max", 100.0)

    np.random.seed(_SEED_A)
    values = np.random.uniform(-val_range, val_range, n) + offset
    labels = np.random.choice([0, 1], size=n, p=[1 - pos_ratio, pos_ratio])
    amt_values = np.random.uniform(amt_min, amt_max, n)

    return {"values": values, "labels": labels, "amt_values": amt_values}


class CornerCases:
    def empty_data():
        return {
            "a": [],
            "b": [],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": Tag.EMPTY_DATA,
        }

    def not_enough_data():
        return {
            "a": [1],
            "b": [2],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": Tag.NOT_ENOUGH_DATA,
        }

    def incorrect_datatype():
        return {
            "a": ["1", "2", "3"],
            "b": [1, 2, 3],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": Tag.INCORRECT_DATA_TYPE,
        }

    def include_nan():
        return {
            "a": [1, 2, 3, np.nan],
            "b": [1, 2, 3, 4],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": Tag.NAN_IN_DATA,
        }

    def mismatched_length():
        return {
            "a": [1, 2, 3],
            "b": [1, 2],
            "res": None,
            "res2": None,
            "tag": Tag.SIZE_MISMATCH,
        }

    def incorrect_input():
        return {
            "a": [1, 2],
            "b": [3, 4],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": Tag.INCORRECT_INPUT,
        }

    def single_class():
        return {
            "a": [1, 1, 1, 1],
            "b": [0.9, 0.8, 0.4, 0.3],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": Tag.SINGLE_CLASS,
        }

    def int_label():
        return {
            "a": [1, 1, 0, 0],
            "b": [0.9, 0.8, 0.4, 0.3],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": None,
        }

    def str_label():
        return {
            "a": ["1", "1", "0", "0"],
            "b": [0.9, 0.8, 0.4, 0.3],
            "ftype": "numerical",
            "btype": "bins",
            "nb": 10,
            "res": None,
            "res2": None,
            "tag": None,
        }


class CornerCasesQuality:
    def empty_data():
        return {
            "a": [],
            "b": [],
            "ftype": "categorical",
            "cats": None,
            "low": None,
            "high": None,
            "res": [],
            "res2": [],
            "tag": Tag.EMPTY_DATA,
        }

    def invalid_ftype():
        return {
            "a": ["1", "2"],
            "b": ["1", "2"],
            "ftype": "str",
            "low": None,
            "high": None,
            "res": [],
            "res2": [],
            "tag": Tag.DATATYPE_UNSUPPORTED,
        }

    def mismatched_datatype():
        return {
            "a": ["a", "b", 1.5, 2],
            "b": [],
            "type": "continuous",
            "cats": None,
            "low": None,
            "high": None,
            "res": [],
            "res2": [],
            "tag": Tag.INCORRECT_DATA_TYPE,
        }
