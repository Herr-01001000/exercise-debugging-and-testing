import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from predict import square_root_unscented_predict
from predict import _calculate_sigma_points
from predict import _calculate_sigma_weights
from predict import _cobb_douglas


FACTORS = list('cni')


@pytest.fixture
def setup_predict():

    out = {}
    out['state'] = pd.Series(data=[12, 10, 8], index=FACTORS)
    out['root_cov'] = pd.DataFrame(
        data=[[6, 0, 0], [3, 5, 0], [2, 1, 4.0]],
        columns=FACTORS,
        index=FACTORS,
    )
    params = {
        'c': {'gammas': pd.Series(data=[0.5] * 3, index=FACTORS), 'a': 0.5},
        'n': {'gammas': pd.Series(data=[1.5, 1, 0], index=FACTORS), 'a': 0.1},
        'i': {'gammas': pd.Series(data=[0, 0, 1.0], index=FACTORS), 'a': 1.2},
    }
    out['params'] = params
    out['shock_sds'] = pd.Series(data=[1, 2, 3.0], index=FACTORS)
    out['kappa'] = 1

    return out


@pytest.fixture
def expected_predict():
    out = {}
    out['mean'] = pd.Series(data=[13.42979972, 56.04386809, 9.6], index=FACTORS)
    out['cov'] = pd.DataFrame(
        data=[
            [126.640480, 508.664745, 33.255376],
            [508.664745, 2717.324849, 70.437680],
            [33.255376, 70.437680, 32.040000],
        ],
        columns=FACTORS,
        index=FACTORS
    )
    out['sigma_points'] = pd.DataFrame(
        data=[
            [12.0, 10.0, 8.0],
            [24.0, 10.0, 8.0],
            [18.0, 20.0, 8.0],
            [16.0, 12.0, 16.0],
            [0, 10.0, 8.0],
            [6.0, 0, 8.0],
            [8.0, 8.0, 0]

        ],
        columns=FACTORS,
        index=range(7)
    )
    out['sigma_weights'] = pd.Series(
        data=[0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        index=range(7)
    )
    out['cobb_douglas'] = pd.Series(data=[15.4919, 21.9089, 26.8328, 27.7128, 0, 0, 0])

    return out


def test_square_root_unscented_predict_mean(setup_predict, expected_predict):
    calc_mean, calc_root_cov = square_root_unscented_predict(**setup_predict)
    assert_series_equal(calc_mean, expected_predict['mean'])


def test_square_root_unscented_predict_cov_values(setup_predict, expected_predict):
    calc_mean, calc_root_cov = square_root_unscented_predict(**setup_predict)
    calc_cov = calc_root_cov.dot(calc_root_cov.T)
    assert_frame_equal(calc_cov, expected_predict['cov'])


def test_square_root_unscented_predict_sigma_points(setup_predict, expected_predict):
    calc_sigma_points = _calculate_sigma_points(setup_predict['state'], setup_predict['root_cov'], setup_predict['kappa'])
    assert_frame_equal(calc_sigma_points, expected_predict['sigma_points'])


def test_square_root_unscented_predict_sigma_weights(setup_predict, expected_predict):
    calc_sigma_weights = _calculate_sigma_weights(setup_predict['state'], setup_predict['kappa'])
    assert_series_equal(calc_sigma_weights, expected_predict['sigma_weights'])


def test_square_root_unscented_predict_cobb_douglas(setup_predict, expected_predict):
    calc_sigma_points = _calculate_sigma_points(setup_predict['state'], setup_predict['root_cov'], setup_predict['kappa'])
    calc_cobb_douglas = _cobb_douglas(calc_sigma_points, pd.Series(data=[0.5] * 3, index=FACTORS), 0.5)
    assert_series_equal(calc_cobb_douglas, expected_predict['cobb_douglas'])
