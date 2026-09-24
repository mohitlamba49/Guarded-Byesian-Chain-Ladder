import os
import pandas as pd
from gbcl_core import build_calendar_states


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_latest_calendar_has_only_unresolved_state():
    triangle = pd.read_csv(os.path.join(ROOT, "examples", "example_triangle.csv"), index_col="AccidentYear")
    states, calendars, valuation = build_calendar_states(triangle)
    terminal = [state for state in states if state["ChangeCalendar"] == valuation]
    assert len(terminal) == 1
    assert terminal[0]["StateType"] == "Unresolved"


def test_state_priors_sum_to_one_and_balance_years():
    triangle = pd.read_csv(os.path.join(ROOT, "examples", "example_triangle.csv"), index_col="AccidentYear")
    states, calendars, valuation = build_calendar_states(triangle)
    assert abs(sum(state["PriorProbability"] for state in states) - 1.0) < 1e-12
    changed = pd.DataFrame(states[1:])
    year_mass = changed.groupby("ChangeCalendar")["PriorProbability"].sum()
    assert year_mass.max() - year_mass.min() < 1e-12
