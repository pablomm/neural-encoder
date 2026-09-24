"""Adapted from mvlearn 0.4.1 (utils/utils.py), licensed under MIT.

Copyright (c) 2020 The mvlearn developers. See LICENSE in this directory.
Source: https://github.com/mvlearn/mvlearn/tree/v0.4.1
"""

import numpy as np
from sklearn.utils import check_array


def check_Xs(
    Xs,
    multiview=False,
    enforce_views=None,
    copy=False,
    return_dimensions=False,
):
    r"""
    Checks Xs and ensures it to be a list of 2D matrices.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    multiview : boolean, (default=False)
        If True, throws error if just 1 data matrix given.

    enforce_views : int, (default=not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    copy : boolean, (default=False)
        If True, the returned Xs is a copy of the input Xs,
        and operations on the output will not affect
        the input.
        If False, the returned Xs is a view of the input Xs,
        and operations on the output will change the input.

    return_dimensions : boolean, (default=False)
        If True, the function also returns the dimensions of the multiview
        dataset. The dimensions are n_views, n_samples, n_features where
        n_samples and n_views are respectively the number of views and the
        number of samples, and n_features is a list of length n_views
        containing the number of features of each view.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).

    n_views : int
        The number of views in the dataset. Returned only if
        ``return_dimensions`` is ``True``.

    n_samples : int
        The number of samples in the dataset. Returned only if
        ``return_dimensions`` is ``True``.

    n_features : list
        List of length ``n_views`` containing the number of features in
        each view. Returned only if ``return_dimensions`` is ``True``.
    """
    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = f"If not list, input must be of type np.ndarray,\
                not {type(Xs)}"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    n_views = len(Xs)
    if n_views == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if multiview:
        if n_views == 1:
            msg = "Must provide at least two data matrices"
            raise ValueError(msg)
        if enforce_views is not None and n_views != enforce_views:
            msg = "Wrong number of views. Expected {} but found {}".format(
                enforce_views, n_views
            )
            raise ValueError(msg)

    Xs = [check_array(X, allow_nd=False, copy=copy) for X in Xs]

    if not len(set([X.shape[0] for X in Xs])) == 1:
        msg = "All views must have the same number of samples"
        raise ValueError(msg)

    if return_dimensions:
        n_samples = Xs[0].shape[0]
        n_features = [X.shape[1] for X in Xs]
        return Xs, n_views, n_samples, n_features
    else:
        return Xs


def param_as_list(param, n_views, single_None=False):
    """
    Returns a parameter as a list

    Parameters
    ----------

    Returns
    -------
    """
    if isinstance(param, list):
        assert len(param) == n_views, \
            f"params {param} must be of length n_views={n_views}"
    elif not (param is None and single_None):
        param = [param] * n_views
    return param
