"""
Test that the approximate form of welch_t_test is consistent
enough with the exact form
"""
import pytest

import itertools
import numpy as np
import scipy.stats
from unittest.mock import patch

import cell_type_mapper.utils.stats_utils as stats_utils


@pytest.mark.parametrize(
        "boring_t,big_nu",
        itertools.product([3.0, 3.5, 2.5, 1.2],
                          [10, 1000, None]))
def test_t_test_approx(boring_t, big_nu):

    rng = np.random.default_rng(671231)

    t_values = np.linspace(-20.0, 20.0, 32000)
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
         exact_p) = stats_utils.exact_welch_t_test(
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

    if big_nu is None:
        # check that we are not breaking designation of
        # which genes pass the p value threshold
        # (if we specify non-none big_nu, we know this
        # test will break)

        approx_corrected = stats_utils.correct_ttest(approx_p)
        exact_corrected = stats_utils.correct_ttest(exact_p)

        # the p-value associated with our boring_t
        cutoff_p = scipy.stats.norm.cdf(-1*boring_t)

        approx_pass = (approx_corrected < cutoff_p)
        exact_pass = (exact_corrected < cutoff_p)
        assert approx_pass.sum() > 0
        np.testing.assert_array_equal(
            approx_pass,
            exact_pass)

        np.testing.assert_allclose(
            approx_corrected[exact],
            exact_corrected[exact],
            atol=0.0,
            rtol=1.0e-6)


def test_when_boring_t_is_zero():
    rng = np.random.default_rng(671231)
    n_genes = 220
    mean1 = rng.random(n_genes)
    var1 = rng.random(n_genes)
    n1 = 45
    mean2 = rng.random(n_genes)
    var2 = rng.random(n_genes)
    n2 = 36
    (_,
     _,
     exact_p) = stats_utils.exact_welch_t_test(
         mean1=mean1,
         var1=var1,
         n1=n1,
         mean2=mean2,
         var2=var2,
         n2=n2)
    (_,
     _,
     approx_p) = stats_utils.approximate_welch_t_test(
         mean1=mean1,
         var1=var1,
         n1=n1,
         mean2=mean2,
         var2=var2,
         n2=n2,
         boring_t=0.0,
         big_nu=None)

    np.testing.assert_allclose(
        approx_p,
        exact_p,
        atol=0.0,
        rtol=1.0e-6)


@pytest.mark.parametrize(
    'p_value', [0.01, 0.02, 0.005, 0.003])
def test_boring_t_calc(p_value):
    boring_t = stats_utils.boring_t_from_p_value(p_value)
    assert boring_t > 0
    truth = scipy.stats.norm.cdf(-1*boring_t)
    np.testing.assert_allclose(
        truth, 0.5*p_value, atol=0.0, rtol=1.0e-3)


@pytest.mark.parametrize(
    "p_th", [0.01, 0.0001, 0.000001])
def test_approx_correct_ttest(p_th):
    rng = np.random.default_rng(22312)
    log_vals = rng.integers(-10, 0, 32000)
    raw = np.power(10.0, log_vals)

    shld_match = (raw < p_th)
    assert shld_match.sum() > 0

    exact = stats_utils.correct_ttest(raw)
    approx = stats_utils.approx_correct_ttest(raw, p_th=p_th)

    assert (exact[raw >= p_th] >= p_th).all()
    assert not np.allclose(exact, raw)

    np.testing.assert_allclose(
        exact[shld_match],
        approx[shld_match],
        rtol=0.0,
        atol=1.0e-6)
