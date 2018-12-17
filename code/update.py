"""Functions for the update step of a square root unscented Kalman filter.

"""

import pandas as pd
import numpy as np


def square_root_linear_update(state, root_cov, measurement, loadings, meas_var):
    """Update *state* and *root_cov* with a *measurement*.

    Args:
        state (pd.Series): pre-update estimate of the unobserved state vector
        root_cov (pd.DataFrame): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        measurement (float): the measurement to incorporate
        loadings (pd.Series): the factor loadings
        meas_var (float): the variance of measurement errors

    Returns:
        updated_state (pd.Series)
        updated_root_cov (pd.DataFrame)

    """

    residual = _calculate_residual(state, measurement, loadings)
    root_variance_y, scaled_f, updated_root_cov_transpose = _qr_decomposition(
        state, root_cov, loadings, meas_var
    )
    kalman_gain = _calculate_kalman_gain(scaled_f, root_variance_y)
    updated_state = _update_state(state, kalman_gain, residual)
    updated_root_cov = _update_root_cov(updated_root_cov_transpose)

    return updated_state, updated_root_cov


def _calculate_residual(state, measurement, loadings):
    """Calculate *residual* by *state*, *measurement* and *loadings*.

    Args:
        state (pd.Series): pre-update estimate of the unobserved state vector
        measurement (float): the measurement to incorporate
        loadings (pd.Series): the factor loadings

    Returns:
        residual (float)
    """

    predict_measurement = loadings.dot(state)
    residual = measurement - predict_measurement

    return residual


def _qr_decomposition(state, root_cov, loadings, meas_var):
    """Calculate the elements in QR decomposition of matrix M
    by *state*, *root_cov*, *measurement*, *loadings* and *meas_var*.

    Args:
        state (pd.Series): pre-update estimate of the unobserved state vector
        root_cov (pd.DataFrame): lower triangular matrix square-root of the
            covariance matrix of the state vector before the update
        loadings (pd.Series): the factor loadings
        meas_var (float): the variance of measurement errors

    Returns:
        root_variance_y (pd.DataFrame): square root of variance of y
        scaled_f (pd.DataFrame): intermediate result
        updated_root_cov_transpose (pd.DataFrame): transpose of
            the square root of updated covariance matrix
    """

    f_star = root_cov.T.dot(loadings)
    helper_matrix_left = np.append(np.sqrt(meas_var), f_star)
    helper_matrix_right = np.vstack([np.zeros(len(state)), root_cov.T])
    helper_matrix = np.c_[helper_matrix_left, helper_matrix_right]

    qr = pd.DataFrame(
            data=np.linalg.qr(helper_matrix, mode='r'),
            columns=['0', 'c', 'n', 'i'],
            index=['0', 'c', 'n', 'i'],
        )

    qr_left, qr_right = np.split(qr, [1], axis=1)
    root_variance_y, zeros = np.split(qr_left, [1])
    scaled_f, updated_root_cov_transpose = np.split(qr_right, [1])

    return root_variance_y, scaled_f, updated_root_cov_transpose


def _calculate_kalman_gain(scaled_f, root_variance_y):
    """Calculate *kalman gain* by *scaled_f* and *root_variance_y*.

    Args:
        root_variance_y (pd.DataFrame): square root of variance of y
        scaled_f (pd.DataFrame): intermediate result

    Returns:
        kalman_gain (pd.Series): scaled Kalman gain
    """
    kalman_gain = scaled_f.T.div(root_variance_y['0']).squeeze().rename()

    return kalman_gain


def _update_state(state, kalman_gain, residual):
    updated_state = state + kalman_gain.multiply(residual, axis=0)

    return updated_state


def _update_root_cov(updated_root_cov_transpose):
    return updated_root_cov_transpose.T
