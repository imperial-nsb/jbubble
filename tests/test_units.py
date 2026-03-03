"""Tests for jbubble.units — unit scaling helpers."""

import pytest
from jbubble import Units


def test_default_l_scale():
    u = Units()
    assert u.L_scale == pytest.approx(1e-6)


def test_default_t_scale():
    u = Units()
    assert u.T_scale == pytest.approx(1e-6)


def test_default_m_scale():
    u = Units()
    assert u.M_scale == pytest.approx(1e-15)


def test_pressure_scale_derived_correctly():
    u = Units()
    expected = u.M_scale / (u.L_scale * u.T_scale**2)
    assert u.P_scale == pytest.approx(expected)


def test_velocity_scale_derived_correctly():
    u = Units()
    assert u.vel_scale == pytest.approx(u.L_scale / u.T_scale)


def test_acc_scale_derived_correctly():
    u = Units()
    assert u.acc_scale == pytest.approx(u.L_scale / u.T_scale**2)


def test_chi_scale_equals_sigma_scale():
    u = Units()
    assert u.chi_scale == pytest.approx(u.sigma_scale)


def test_freq_scale_is_reciprocal_t():
    u = Units()
    assert u.freq_scale == pytest.approx(1.0 / u.T_scale)


def test_rho_scale_derived():
    u = Units()
    expected = u.M_scale / u.L_scale**3
    assert u.rho_scale == pytest.approx(expected)


def test_custom_scales_propagate():
    u = Units(L_scale=1e-5, T_scale=1e-5, M_scale=1e-12)
    assert u.rho_scale == pytest.approx(u.M_scale / u.L_scale**3)
    assert u.P_scale == pytest.approx(u.M_scale / (u.L_scale * u.T_scale**2))
    assert u.vel_scale == pytest.approx(u.L_scale / u.T_scale)
