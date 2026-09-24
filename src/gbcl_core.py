"""Reference implementation of the Guarded Bayesian Chain-Ladder (GBCL).

The code follows Sections 2 and Appendix A of the accompanying manuscript.  It is
deliberately function-based: the state construction, collapsed Gibbs sampler,
terminal-continuation scenario and constrained Bayes action can be inspected without
an application framework.

Gamma distributions use the shape-rate parameterization in the manuscript.  NumPy's
``Generator.gamma`` expects shape-scale arguments, so rates are inverted when draws
are generated.
"""

import math
import numpy as np
import pandas as pd


def incremental_triangle(cumulative):
    incremental = cumulative.diff(axis=1)
    incremental.iloc[:, 0] = cumulative.iloc[:, 0]
    return incremental


def validate_triangle(cumulative):
    triangle = cumulative.copy().astype(float)
    triangle.columns = np.arange(triangle.shape[1])
    if triangle.shape[0] < 3 or triangle.shape[1] < 3:
        raise ValueError("The triangle must have at least three origins and developments.")
    if not np.issubdtype(np.asarray(triangle.index).dtype, np.number):
        raise ValueError("The index must contain numeric origin-period labels.")
    observed = triangle.notna().to_numpy()
    for row in observed:
        seen_missing = False
        for value in row:
            if not value:
                seen_missing = True
            if value and seen_missing:
                raise ValueError("Observed cells must precede missing cells within every row.")
    incremental = incremental_triangle(triangle)
    observed_incremental = incremental.stack().to_numpy()
    if np.any(observed_incremental < 0):
        raise ValueError("The Gamma-conjugate implementation requires nonnegative increments.")
    if np.any(incremental.sum(axis=1) <= 0):
        raise ValueError("Every observed origin-period total must be positive.")
    if np.any(incremental.sum(axis=0) <= 0):
        raise ValueError("Every observed development-period total must be positive.")
    return triangle, incremental


def calendar_matrix(triangle):
    origins = np.asarray(triangle.index, dtype=int)
    developments = np.arange(triangle.shape[1], dtype=int)
    return origins[:, None] + developments[None, :]


def fit_odp_ipf(incremental, tolerance=1e-10, max_iterations=10000):
    y = incremental.to_numpy(dtype=float)
    observed = np.isfinite(y)
    n_origins, n_developments = y.shape
    origin_effect = np.ones(n_origins)
    development_effect = np.ones(n_developments)
    for iteration in range(max_iterations):
        previous = np.concatenate([origin_effect, development_effect])
        origin_effect = np.array([np.nansum(y[i]) / np.sum(development_effect[observed[i]]) for i in range(n_origins)])
        development_effect = np.array([np.nansum(y[:, j]) / np.sum(origin_effect[observed[:, j]]) for j in range(n_developments)])
        normalization = development_effect[0]
        development_effect = development_effect / normalization
        origin_effect = origin_effect * normalization
        current = np.concatenate([origin_effect, development_effect])
        relative_change = np.max(np.abs(current - previous) / (1.0 + np.abs(previous)))
        if relative_change < tolerance:
            break
    fitted = origin_effect[:, None] * development_effect[None, :]
    parameters = n_origins + n_developments - 1
    degrees_freedom = int(observed.sum() - parameters)
    if degrees_freedom <= 0:
        raise ValueError("The observed triangle has no degrees of freedom for ODP dispersion.")
    pearson = np.where(observed, (y - fitted) ** 2 / np.maximum(fitted, 1e-15), np.nan)
    dispersion = float(np.nansum(pearson) / degrees_freedom)
    if not np.isfinite(dispersion) or dispersion <= 0:
        raise ValueError("The baseline ODP dispersion estimate is not positive and finite.")
    return origin_effect, development_effect, dispersion, iteration + 1


def classical_chain_ladder(cumulative):
    triangle, _ = validate_triangle(cumulative)
    factors = []
    for j in range(triangle.shape[1] - 1):
        current = triangle.iloc[:, j]
        following = triangle.iloc[:, j + 1]
        use = current.notna() & following.notna() & (current > 0)
        if use.sum() == 0:
            raise ValueError(f"No valid observations for development {j} to {j + 1}.")
        factors.append(float(following[use].sum() / current[use].sum()))
    cdf = np.ones(triangle.shape[1])
    for j in range(triangle.shape[1] - 2, -1, -1):
        cdf[j] = factors[j] * cdf[j + 1]
    latest = triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
    latest_development = triangle.apply(lambda row: int(row.dropna().index[-1]), axis=1)
    ultimate = pd.Series([latest.loc[i] * cdf[latest_development.loc[i]] for i in triangle.index], index=triangle.index)
    reserve = ultimate - latest
    return {"factors": np.asarray(factors), "cdf": cdf, "latest": latest, "ultimate": ultimate, "reserve": reserve}


def build_calendar_states(triangle, pi0=0.70, minimum_temporary_cells=3, minimum_before_cells=10, minimum_after_cells=10):
    observed = triangle.notna().to_numpy()
    future = ~observed
    calendars = calendar_matrix(triangle)
    valuation = int(np.max(calendars[observed]))
    candidates = []
    for change_year in sorted(np.unique(calendars[observed])):
        change_year = int(change_year)
        if change_year >= valuation:
            continue
        temporary_observed = observed & (calendars == change_year)
        if temporary_observed.sum() >= minimum_temporary_cells:
            candidates.append({"StateType": "Temporary", "ChangeCalendar": change_year, "ObservedMask": temporary_observed, "FutureMask": np.zeros_like(future), "StateLabel": f"Temporary_{change_year}"})
        persistent_observed = observed & (calendars >= change_year)
        persistent_before = observed & (calendars < change_year)
        if persistent_before.sum() >= minimum_before_cells and persistent_observed.sum() >= minimum_after_cells:
            candidates.append({"StateType": "Persistent", "ChangeCalendar": change_year, "ObservedMask": persistent_observed, "FutureMask": future & (calendars >= change_year), "StateLabel": f"Persistent_{change_year}"})
    terminal_observed = observed & (calendars == valuation)
    if terminal_observed.sum() >= minimum_temporary_cells:
        candidates.append({"StateType": "Unresolved", "ChangeCalendar": valuation, "ObservedMask": terminal_observed, "FutureMask": future, "StateLabel": f"RecentUnresolved_{valuation}"})
    no_change = {"StateType": "NoChange", "ChangeCalendar": None, "ObservedMask": np.zeros_like(observed), "FutureMask": np.zeros_like(future), "StateLabel": "NoChange", "PriorProbability": float(pi0)}
    states = [no_change]
    if candidates:
        years = sorted({state["ChangeCalendar"] for state in candidates})
        for year in years:
            states_in_year = [state for state in candidates if state["ChangeCalendar"] == year]
            prior = (1.0 - pi0) / len(years) / len(states_in_year)
            for state in states_in_year:
                state["PriorProbability"] = prior
                states.append(state)
    elif pi0 < 1.0:
        states[0]["PriorProbability"] = 1.0
    for state_id, state in enumerate(states):
        state["StateID"] = state_id
    prior_total = sum(state["PriorProbability"] for state in states)
    if not np.isclose(prior_total, 1.0):
        raise RuntimeError("Calendar-state prior probabilities do not sum to one.")
    return states, calendars, valuation


def no_change_state(triangle):
    observed = triangle.notna().to_numpy()
    state = {"StateID": 0, "StateType": "NoChange", "ChangeCalendar": None, "ObservedMask": np.zeros_like(observed), "FutureMask": np.zeros_like(observed), "StateLabel": "NoChange", "PriorProbability": 1.0}
    return [state]


def conditional_state_probabilities(origin_effect, development_effect, scaled_y, scaled_dispersion, states, kappa):
    observed = np.isfinite(scaled_y)
    base_mean = origin_effect[:, None] * development_effect[None, :]
    total_mean = float(np.sum(base_mean[observed]))
    log_probabilities = []
    for state in states:
        if state["StateType"] == "NoChange":
            log_likelihood = -total_mean / scaled_dispersion
        else:
            affected = state["ObservedMask"]
            affected_claims = float(np.sum(scaled_y[affected]))
            affected_mean = float(np.sum(base_mean[affected]))
            posterior_shape = kappa + affected_claims / scaled_dispersion
            posterior_rate = kappa + affected_mean / scaled_dispersion
            log_likelihood = -(total_mean - affected_mean) / scaled_dispersion
            log_likelihood += kappa * np.log(kappa) - math.lgamma(kappa)
            log_likelihood += math.lgamma(posterior_shape) - posterior_shape * np.log(posterior_rate)
        log_probabilities.append(np.log(state["PriorProbability"]) + log_likelihood)
    log_probabilities = np.asarray(log_probabilities)
    maximum = np.max(log_probabilities)
    probabilities = np.exp(log_probabilities - maximum)
    probabilities = probabilities / probabilities.sum()
    return probabilities, base_mean


def run_one_chain(incremental, triangle, origin_initial, development_initial, dispersion, states, kappa, n_iter, burn_in, thin, seed, prior_shape=0.001, prior_rate=0.001):
    random = np.random.default_rng(seed)
    y = incremental.to_numpy(dtype=float)
    observed = np.isfinite(y)
    first_development = y[:, 0]
    scale = float(np.nanmedian(first_development[first_development > 0]))
    scaled_y = y / scale
    scaled_dispersion = dispersion / scale
    origin_effect = origin_initial / scale
    development_effect = development_initial.copy()
    origin_effect = origin_effect * random.lognormal(0.0, 0.02, size=len(origin_effect))
    development_effect[1:] = development_effect[1:] * random.lognormal(0.0, 0.02, size=len(development_effect) - 1)
    development_effect[0] = 1.0
    row_totals = np.nansum(scaled_y, axis=1)
    column_totals = np.nansum(scaled_y, axis=0)
    origin_draws = []
    development_draws = []
    multiplier_draws = []
    state_draws = []
    probability_draws = []
    for iteration in range(n_iter):
        probabilities, base_mean = conditional_state_probabilities(origin_effect, development_effect, scaled_y, scaled_dispersion, states, kappa)
        state_id = int(random.choice(len(states), p=probabilities))
        state = states[state_id]
        affected = state["ObservedMask"]
        if state["StateType"] == "NoChange":
            multiplier = 1.0
        else:
            affected_claims = float(np.sum(scaled_y[affected]))
            affected_mean = float(np.sum(base_mean[affected]))
            posterior_shape = kappa + affected_claims / scaled_dispersion
            posterior_rate = kappa + affected_mean / scaled_dispersion
            multiplier = float(random.gamma(posterior_shape, 1.0 / posterior_rate))
        multiplier_matrix = np.where(affected, multiplier, 1.0)
        for i in range(triangle.shape[0]):
            posterior_shape = prior_shape + row_totals[i] / scaled_dispersion
            posterior_rate = prior_rate + np.sum(development_effect[observed[i]] * multiplier_matrix[i, observed[i]]) / scaled_dispersion
            origin_effect[i] = random.gamma(posterior_shape, 1.0 / posterior_rate)
        development_effect[0] = 1.0
        for j in range(1, triangle.shape[1]):
            posterior_shape = prior_shape + column_totals[j] / scaled_dispersion
            posterior_rate = prior_rate + np.sum(origin_effect[observed[:, j]] * multiplier_matrix[observed[:, j], j]) / scaled_dispersion
            development_effect[j] = random.gamma(posterior_shape, 1.0 / posterior_rate)
        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            origin_draws.append(origin_effect.copy() * scale)
            development_draws.append(development_effect.copy())
            multiplier_draws.append(multiplier)
            state_draws.append(state_id)
            probability_draws.append(probabilities.copy())
    return {"origin": np.asarray(origin_draws), "development": np.asarray(development_draws), "multiplier": np.asarray(multiplier_draws), "state": np.asarray(state_draws), "probability": np.asarray(probability_draws)}


def run_chains(incremental, triangle, origin_initial, development_initial, dispersion, states, kappa=25.0, n_chains=4, n_iter=5000, burn_in=1000, thin=5, seed=20260904):
    chains = []
    for chain in range(n_chains):
        draws = run_one_chain(incremental, triangle, origin_initial, development_initial, dispersion, states, kappa, n_iter, burn_in, thin, seed + 1000 * chain)
        chains.append(draws)
    output = {}
    for name in ["origin", "development", "multiplier", "state", "probability"]:
        output[name + "_chains"] = np.asarray([chain[name] for chain in chains])
        chain_array = output[name + "_chains"]
        output[name] = chain_array.reshape((-1,) + chain_array.shape[2:])
    return output


def rhat(chains):
    values = np.asarray(chains, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    n_chains, n_draws, _ = values.shape
    if n_chains < 2 or n_draws < 2:
        return np.full(values.shape[-1], np.nan)
    chain_means = values.mean(axis=1)
    within = values.var(axis=1, ddof=1).mean(axis=0)
    between = n_draws * chain_means.var(axis=0, ddof=1)
    estimate = (n_draws - 1.0) * within / n_draws + between / n_draws
    return np.sqrt(np.divide(estimate, within, out=np.ones_like(estimate), where=within > 0))


def convergence_diagnostics(draws):
    components = [rhat(draws["origin_chains"]), rhat(draws["development_chains"][:, :, 1:]), rhat(draws["multiplier_chains"])]
    finite = np.concatenate([component[np.isfinite(component)] for component in components])
    maximum_rhat = float(np.max(finite)) if len(finite) else np.nan
    pooled_probability = draws["probability_chains"].mean(axis=(0, 1))
    chain_probability = draws["probability_chains"].mean(axis=1)
    delta_probability = float(np.max(np.abs(chain_probability - pooled_probability[None, :])))
    return {"MaximumRhat": maximum_rhat, "MaximumStateProbabilityDifference": delta_probability}


def posterior_state_summary(states, probability_draws, state_draws, multiplier_draws, threshold=0.75):
    posterior = probability_draws.mean(axis=0)
    rows = []
    for state_id, state in enumerate(states):
        selected = state_draws == state_id
        changed_multiplier = multiplier_draws[selected]
        rows.append({"StateID": state_id, "StateLabel": state["StateLabel"], "StateType": state["StateType"], "ChangeCalendar": state["ChangeCalendar"], "PriorProbability": state["PriorProbability"], "PosteriorProbability": posterior[state_id], "ObservedAffectedCells": int(state["ObservedMask"].sum()), "FutureAffectedCells": int(state["FutureMask"].sum()), "SampledVisits": int(selected.sum()), "ConditionalMultiplierMean": float(changed_multiplier.mean()) if selected.any() else np.nan})
    table = pd.DataFrame(rows)
    p_no_change = float(table.loc[table["StateType"] == "NoChange", "PosteriorProbability"].sum())
    p_change = 1.0 - p_no_change
    leading = table.loc[table["PosteriorProbability"].idxmax()]
    label = leading["StateLabel"] if p_change >= threshold else "NoChange"
    prior_no_change = float(table.loc[table["StateType"] == "NoChange", "PriorProbability"].sum())
    prior_change = 1.0 - prior_no_change
    bayes_factor = (p_change / p_no_change) / (prior_change / prior_no_change) if p_no_change > 0 and prior_change > 0 else np.nan
    summary = {"ProbabilityNoChange": p_no_change, "ProbabilityAnyChange": p_change, "LeadingState": leading["StateLabel"], "ReportedState": label, "ChangeBayesFactor": bayes_factor}
    return table, summary


def reserve_draws(triangle, draws, states, dispersion, q=0.50, seed=20260905):
    random = np.random.default_rng(seed)
    future = triangle.isna().to_numpy()
    n_draws = len(draws["state"])
    parameter_by_origin = np.zeros((n_draws, triangle.shape[0]))
    predictive_by_origin = np.zeros((n_draws, triangle.shape[0]))
    for draw in range(n_draws):
        state = states[int(draws["state"][draw])]
        future_mask = state["FutureMask"]
        if state["StateType"] == "Unresolved":
            continues = bool(random.binomial(1, q))
            future_mask = future_mask if continues else np.zeros_like(future_mask)
        multiplier_matrix = np.where(future_mask, draws["multiplier"][draw], 1.0)
        mean = draws["origin"][draw, :, None] * draws["development"][draw, None, :] * multiplier_matrix
        parameter_by_origin[draw] = np.sum(np.where(future, mean, 0.0), axis=1)
        poisson_mean = np.where(future, np.maximum(mean / dispersion, 0.0), 0.0)
        predictive = random.poisson(poisson_mean) * dispersion
        predictive_by_origin[draw] = predictive.sum(axis=1)
    return {"parameter_by_origin": parameter_by_origin, "predictive_by_origin": predictive_by_origin, "parameter_total": parameter_by_origin.sum(axis=1), "predictive_total": predictive_by_origin.sum(axis=1)}


def guard_weight(reference_mean, raw_mean, impact_limit):
    if raw_mean == reference_mean or np.isinf(impact_limit):
        return 1.0
    return float(min(1.0, impact_limit * reference_mean / abs(raw_mean - reference_mean)))


def guarded_action(reference_mean, raw_mean, impact_limit=0.01):
    weight = guard_weight(reference_mean, raw_mean, impact_limit)
    return reference_mean + weight * (raw_mean - reference_mean), weight


def transport_draws(reference_draws, raw_draws, weight):
    reference_draws = np.asarray(reference_draws)
    raw_draws = np.asarray(raw_draws)
    if reference_draws.shape != raw_draws.shape:
        raise ValueError("Reference and raw draws must have the same shape.")
    return reference_draws + weight * (raw_draws - reference_draws)


def fit_gbcl(cumulative, q=0.50, impact_limit=0.01, pi0=0.70, kappa=25.0, reporting_threshold=0.75, n_chains=4, n_iter=5000, burn_in=1000, thin=5, seed=20260904):
    triangle, incremental = validate_triangle(cumulative)
    origin_initial, development_initial, dispersion, ipf_iterations = fit_odp_ipf(incremental)
    states, calendars, valuation = build_calendar_states(triangle, pi0=pi0)
    baseline_states = no_change_state(triangle)
    raw_draws = run_chains(incremental, triangle, origin_initial, development_initial, dispersion, states, kappa=kappa, n_chains=n_chains, n_iter=n_iter, burn_in=burn_in, thin=thin, seed=seed)
    baseline_draws = run_chains(incremental, triangle, origin_initial, development_initial, dispersion, baseline_states, kappa=kappa, n_chains=n_chains, n_iter=n_iter, burn_in=burn_in, thin=thin, seed=seed + 500000)
    raw_reserves = reserve_draws(triangle, raw_draws, states, dispersion, q=q, seed=seed + 700000)
    baseline_reserves = reserve_draws(triangle, baseline_draws, baseline_states, dispersion, q=0.0, seed=seed + 900000)
    reference_mean = float(baseline_reserves["parameter_total"].mean())
    raw_mean = float(raw_reserves["parameter_total"].mean())
    action, weight = guarded_action(reference_mean, raw_mean, impact_limit)
    guarded_parameter = transport_draws(baseline_reserves["parameter_total"], raw_reserves["parameter_total"], weight)
    guarded_predictive = transport_draws(baseline_reserves["predictive_total"], raw_reserves["predictive_total"], weight)
    state_table, state_summary = posterior_state_summary(states, raw_draws["probability"], raw_draws["state"], raw_draws["multiplier"], threshold=reporting_threshold)
    summary = {"ValuationCalendar": valuation, "FiniteHorizonDevelopmentPeriods": triangle.shape[1], "ODPDispersion": dispersion, "IPFIterations": ipf_iterations, "Q": q, "ImpactLimit": impact_limit, "Kappa": kappa, "NoChangePrior": pi0, "BayesianODPReserve": reference_mean, "RawGBCLReserve": raw_mean, "GuardedReserve": action, "ImpactWeight": weight, "RawChangePercent": 100.0 * (raw_mean - reference_mean) / reference_mean, "ReportedChangePercent": 100.0 * (action - reference_mean) / reference_mean, "GuardedPredictiveLower2.5": float(np.percentile(guarded_predictive, 2.5)), "GuardedPredictiveUpper97.5": float(np.percentile(guarded_predictive, 97.5))}
    summary.update(state_summary)
    return {"summary": pd.Series(summary), "state_summary": state_table, "diagnostics": pd.Series(convergence_diagnostics(raw_draws)), "triangle": triangle, "incremental": incremental, "calendar_matrix": calendars, "states": states, "raw_draws": raw_draws, "baseline_draws": baseline_draws, "raw_reserves": raw_reserves, "baseline_reserves": baseline_reserves, "guarded_parameter_draws": guarded_parameter, "guarded_predictive_draws": guarded_predictive, "classical_chain_ladder": classical_chain_ladder(triangle)}
