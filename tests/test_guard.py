import numpy as np
from gbcl_core import guarded_action, transport_draws


def test_guard_projects_to_upper_endpoint():
    action, weight = guarded_action(100.0, 130.0, 0.01)
    assert np.isclose(action, 101.0)
    assert np.isclose(weight, 1.0 / 30.0)


def test_guard_projects_symmetrically():
    upper, _ = guarded_action(100.0, 130.0, 0.01)
    lower, _ = guarded_action(100.0, 70.0, 0.01)
    assert np.isclose(upper, 101.0)
    assert np.isclose(lower, 99.0)


def test_transported_mean_is_guarded_action():
    reference = np.array([90.0, 110.0])
    raw = np.array([120.0, 140.0])
    action, weight = guarded_action(reference.mean(), raw.mean(), 0.05)
    transported = transport_draws(reference, raw, weight)
    assert np.isclose(transported.mean(), action)
