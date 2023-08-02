"""
Test that the approximate form of welch_t_test is consistent
enough with the exact form
"""
import pytest

import itertools
import numpy as np
from unittest.mock import patch

import cell_type_mapper.utils.stats_utils as stats_utils


@pytest.mark.parametrize(
        "boring_t,big_nu",
        itertools.product([3.0, 3.5, 2.5, 1.2],
                          [10, 1000, None]))
def test_t_test_approx(boring_t, big_nu):

    rng = np.random.default_rng(671231)

    t_values = np.linspace(-4.0, 4.0, 1000)
    nu_values = rng.choice(
            [5, 6, 10, 12, 400, 500, 1000, 2000, 2220],
            len(t_values),
            replace=True)

    def _alt_nu(*args, **kwargs):
        return t_values, nu_values


    with patch('cell_type_mapper.utils.stats_utils._calculate_tt_nu',
               new=_alt_nu):

        (_,
         _,
         exact_p) = stats_utils.approximate_welch_t_test(
            mean1=None,
            var1=None,
            n1=None,
            mean2=None,
            var2=None,
            n2=None)


        (_,
         _,
         approx_p) = stats_utils.approximate_welch_t_test(
            mean1=None,
            var1=None,
            n1=None,
            mean2=None,
            var2=None,
            n2=None,
            boring_t=boring_t,
            big_nu=big_nu)

    exact = np.logical_or(
        t_values < -boring_t,
        t_values > boring_t)

    inexact = np.logical_not(exact)

    if big_nu is not None:
        exact = np.logical_and(
            exact,
            nu_values < big_nu)

    assert exact.sum() > 0
    assert inexact.sum() > 0

    np.testing.assert_allclose(
        approx_p[exact],
        exact_p[exact],
        atol=0.0,
        rtol=1.0e-6)

    np.testing.assert_allclose(
        approx_p[inexact],
        np.ones(inexact.sum()),
        atol=0.0,
        rtol=1.0e-6)
