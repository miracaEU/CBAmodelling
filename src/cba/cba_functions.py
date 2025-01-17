"""Estimate direct damages to physical assets exposed to hazards

"""

import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
import warnings

pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def prod(val):
    res = 1
    for ele in val:
        res *= ele
    return res


def calculate_growth_rate_factor(
    growth_rate=3.0,
    start_year=2025,
    end_year=2100,
):
    """Set growth rates for yearly and period maintenance costs"""
    growth_rates = []

    if isinstance(growth_rate, float):
        for year in range(start_year, end_year + 1):
            growth_rates.append(
                (
                    year,
                    1.0 * math.pow(1.0 + 1.0 * growth_rate / 100.0, year - start_year),
                )
            )
    else:
        gr_end_y = growth_rate[-1][0]
        gr_end_v = growth_rate[-1][1]
        if end_year > gr_end_y:
            ext_y = list(np.arange(gr_end_y + 1, end_year + 1, 1))
            rates = [gr_end_v] * len(ext_y)
            growth_rate += list(zip(ext_y, rates))
        for i, (year, rate) in enumerate(growth_rate):
            if year > start_year:
                growth_rates.append(
                    (year, prod([1 + v[1] / 100.0 for v in growth_rate[: i + 1]]))
                )
            else:
                growth_rates.append((year, 1))

    return growth_rates


def calculate_discounting_rate_factor(
    discount_rate=4.0,
    start_year=2025,
    end_year=2100,
    maintain_period=5.0,
    skip_year_one=False,
):

    discount_rates = []
    maintain_years = np.arange(start_year + 1, end_year + 1, maintain_period)
    for year in range(start_year, end_year + 1):
        if year in maintain_years:
            discount_rates.append(
                1.0 / math.pow(1.0 + 1.0 * discount_rate / 100.0, year - start_year)
            )
        else:
            if skip_year_one is True:
                discount_rates.append(0)
            else:
                discount_rates.append(1)

    return np.array(discount_rates)


def estimate_risk_time_series(
    risk_dataframe,
    risk_scenario_columns,
    risk_value_columns,
    year_column,
    start_year,
    end_year,
):

    timeseries = np.arange(start_year, end_year + 1, 1)
    risk_dataframe[risk_scenario_columns] = risk_dataframe[
        risk_scenario_columns
    ].astype(str)
    scenarios = risk_dataframe[risk_scenario_columns].drop_duplicates(keep="first")

    index_columns = [
        c for c in risk_dataframe.columns.values.tolist() if c not in risk_value_columns
    ]
    idx_cols = [c for c in index_columns if c != year_column]
    damages_time_series = []
    for sc in scenarios.itertuples():
        query_values = []
        for rs in risk_scenario_columns:
            query_values.append((rs, getattr(sc, rs)))

        query_string = " and ".join([f"{qv[0]}=='{qv[1]}'" for qv in query_values])
        df = risk_dataframe.query(query_string)
        years = sorted(list(set(df[year_column].values.tolist())))
        dam_dfs = []
        for rv in risk_value_columns:
            df_t = df[index_columns + [rv]]
            df_t = (
                df_t.set_index(idx_cols)
                .pivot(columns=year_column)[rv]
                .reset_index()
                .rename_axis(None, axis=1)
            ).fillna(0)
            series = np.array([list(timeseries) * len(df_t.index)]).reshape(
                len(df_t.index), len(timeseries)
            )
            df_t[series[0]] = interp1d(
                years, df_t[years], fill_value="extrapolate", bounds_error=False
            )(series[0])
            df_t[series[0]] = df_t[series[0]].clip(lower=0.0)
            df_t = df_t.melt(id_vars=idx_cols, var_name=year_column, value_name=rv)
            df_t = df_t.sort_values(year_column)
            dam_dfs.append(df_t.set_index(index_columns))
        dam_dfs = pd.concat(dam_dfs, axis=1).reset_index()
        damages_time_series.append(dam_dfs)

    damages_time_series = pd.concat(damages_time_series, axis=0, ignore_index=False)

    return damages_time_series.sort_values(index_columns)


def estimate_adaptation_cost_time_series(
    cost_dataframe,
    cost_scenario_columns,
    initial_investment_column="initial_investment",
    periodic_investment_columns=[
        "periodic_maintenance",
        "periodic_maintenance_interval",
    ],
    recurrent_investment_columns=[
        "recurrent_maintenance",
        "recurrent_maintenance_interval",
    ],
    start_year=2025,
    end_year=2100,
    timeseries_column="year",
):

    timeseries = np.arange(start_year, end_year + 1, 1)
    cost_dataframe[cost_scenario_columns] = cost_dataframe[
        cost_scenario_columns
    ].astype(str)
    scenarios = cost_dataframe[cost_scenario_columns].drop_duplicates(keep="first")
    cost_time_series = []
    for sc in scenarios.itertuples():
        query_values = []
        for rs in cost_scenario_columns:
            query_values.append((rs, getattr(sc, rs)))

        query_string = " and ".join([f"{qv[0]}=='{qv[1]}'" for qv in query_values])
        df = cost_dataframe.query(query_string)
        ini_cost_series = [df[initial_investment_column].values[0]] + [0.0] * (
            len(timeseries) - 1
        )
        if len(periodic_investment_columns) > 0:
            if len(periodic_investment_columns) == 2:
                maintain_period = float(df[periodic_investment_columns[1]].values[0])
            else:
                maintain_period = 1.0
            maintain_years = np.arange(start_year + 1, end_year + 1, maintain_period)
            maintain_years = calculate_discounting_rate_factor(
                discount_rate=0.0,
                start_year=start_year,
                end_year=end_year,
                maintain_period=maintain_period,
                skip_year_one=True,
            )
            per_cost_series = list(
                (df[periodic_investment_columns[0]].values[0]) * maintain_years
            )
        else:
            per_cost_series = [0.0] * len(timeseries)

        if len(recurrent_investment_columns) > 0:
            if len(recurrent_investment_columns) == 2:
                maintain_period = float(df[recurrent_investment_columns[1]].values[0])
            else:
                maintain_period = 1.0
            maintain_years = calculate_discounting_rate_factor(
                discount_rate=0.0,
                start_year=start_year,
                end_year=end_year,
                maintain_period=maintain_period,
                skip_year_one=True,
            )
            rec_cost_series = list(
                (df[recurrent_investment_columns[0]].values[0]) * maintain_years
            )
        else:
            rec_cost_series = [0.0] * len(timeseries)

        cost_time_series.append(
            tuple(
                [qv[1] for qv in query_values]
                + [list(timeseries), ini_cost_series, per_cost_series, rec_cost_series]
            )
        )

    cost_time_series = pd.DataFrame(
        cost_time_series,
        columns=cost_scenario_columns
        + [
            timeseries_column,
            initial_investment_column,
            periodic_investment_columns[0],
            recurrent_investment_columns[0],
        ],
    )
    cost_time_series = cost_time_series.explode(
        [
            timeseries_column,
            initial_investment_column,
            periodic_investment_columns[0],
            recurrent_investment_columns[0],
        ]
    )

    return cost_time_series.sort_values(cost_scenario_columns)
