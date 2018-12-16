"""Functions for the update step of a square root unscented Kalman filter.

"""

import pandas as pd
import numpy as np


def square_root_linear_update(state, root_cov, measurement, loadings):
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
        updated_root_cov (pd.Series)

    """

    return updated_state, updated_root_cov