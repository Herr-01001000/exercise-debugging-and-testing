import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from update import square_root_linear_update

FACTORS = list('cni')


@pytest.fixture
def setup_update():

    out = {}
    out['state'] = pd.Series(data=[13.42979972, 56.04386809, 9.6], index=FACTORS)
    out['root_cov'] = pd.DataFrame(
        data=[
            [-11.2535, 0, 0],
            [-45.2007, -25.9657, 0],
            [-2.95512, 2.43151, -4.17073],
        ],
        columns=FACTORS,
        index=FACTORS
    )
    out['measurement'] = 30.0
    out['loadings'] = pd.Series(data=[0.1, 0.5, 0.4], index=FACTORS)
    out['meas_var'] = 4.0

    return out


# Use the formulae for the normal linear update to generate test cases.
predicted_measurement = setup_update()['loadings'].dot(setup_update()['state'])
residual = setup_update()['measurement'] - predicted_measurement
cov = setup_update()['root_cov'].dot(setup_update()['root_cov'].T)
f = cov.dot(setup_update()['loadings'])
var_y = setup_update()['loadings'].dot(f) + setup_update()['meas_var']
kalman_gain = f.div(var_y)
updated_state = setup_update()['state'] + kalman_gain.multiply(residual)
updated_cov = cov - f.to_frame().dot(f.to_frame().T).div(var_y)


@pytest.fixture
def expected_update():

    out = {}
    out['mean'] = updated_state
    out['cov'] = updated_cov
    out['residual'] = residual
    out['variance_y'] = var_y
    out['kalman_gain'] = kalman_gain

    return out


def test_square_root_linear_update_mean(setup_update, expected_update):
    calc_mean, calc_root_cov = square_root_linear_update(**setup_update)
    assert_series_equal(calc_mean, expected_update['mean'])


def test_square_root_linear_update_cov_values(setup_update, expected_update):
    calc_mean, calc_root_cov = square_root_linear_update(**setup_update)
    calc_cov = calc_root_cov.dot(calc_root_cov.T)
    assert_frame_equal(calc_cov, expected_update['cov'])
