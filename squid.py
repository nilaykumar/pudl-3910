"""
-*- coding: utf-8 -*-
Copyright (C) 2024 RMI

A script that creates csv files that can be used to create a 'Squid' figure that shows
utility forecast errors

Settings for the script are below along with their descriptions
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>5.10,<5.25",
#   "kaleido>0.2,<0.2.2",
#   "rmi.etoolbox @ git+https://github.com/rmi/etoolbox.git",
# ]
# ///
from typing import NamedTuple

import pandas as pd
import numpy as np

from etoolbox.utils.pudl import pd_read_pudl


class SquidData(NamedTuple):
    mean_forecast_errors: pd.DataFrame
    w_average_error: pd.DataFrame
    all_forecasts: pd.DataFrame
    peak_demand: pd.DataFrame


def make_squid(
    first_year: int = 2006,
    last_year: int = 2023,
    load_weight_year: int = 2015,
    years_to_drop: tuple = (),
    *,
    export_csv=True,
):
    """Generate data for squid chart.

    Method to calculate average forecast errors for each FERC planning area
    reporting in FERC 714

    Parameters
    ----------
    first_year : int
        first report_date year to use
    last_year : int
        last report_date year to use
    load_weight_year : int
        the year of peak load data that will be used to weight each respondent
    years_to_drop : tuple (default: ())
        years for which the forecasts will not be included in the squid chart
    export_csv : bool
        export data as csvs to make charts
    """

    # calculate peak load for all respondents in all years
    peak_demand = (
        pd_read_pudl("out_ferc714__hourly_planning_area_demand")
        .assign(
            year=lambda x: x.datetime_utc.dt.year,
            report_year=lambda x: pd.to_datetime(x.report_date).dt.year,
        )
        .rename(columns={"demand_mwh": "actual"})
        .groupby(["respondent_id_ferc714", "report_year", "year"], as_index=False)
        .actual.max()
        .query("year == report_year")
    )

    # load and reshape FERC demand forecast csv
    forecast = (
        pd_read_pudl("out_ferc714__respondents_with_fips")
        .groupby("respondent_id_ferc714", as_index=False)[["respondent_name_ferc714"]]
        .first()
        .merge(
            pd_read_pudl("core_ferc714__yearly_planning_area_demand_forecast")
            .query("report_year >= @first_year & report_year <= @last_year")
            .assign(
                forecast=lambda x: np.maximum(
                    x.summer_peak_demand_forecast_mw, x.summer_peak_demand_forecast_mw
                ),
            )
            .rename(columns={"forecast_year": "year"})[
                ["respondent_id_ferc714", "report_year", "year", "forecast"]
            ],
            on="respondent_id_ferc714",
            how="right",
            validate="1:m",
        )
    )
    # Make squid df
    squid_all_years = (
        forecast.merge(
            peak_demand[["respondent_id_ferc714", "year", "actual"]],
            on=["respondent_id_ferc714", "year"],
            how="inner",
            validate="m:1",
        )
        .assign(
            error=lambda x: np.where(
                (x.actual != 0.0) & (x.forecast != 0.0),
                (x.forecast - x.actual) / x.actual,
                0.0,
            ),
            plan_year=lambda x: x.year - x.report_year,
        )
        .pivot(
            index=["respondent_id_ferc714", "respondent_name_ferc714", "year"],
            columns="plan_year",
            values="error",
        )
        .dropna(how="all")
        .dropna(axis=0, subset=[1])
    )
    if export_csv:
        # export all the forecasts errors
        squid_all_years.to_csv("squid_all_forecast_errors.csv")
    # calculate mean forecast errors for each forecast increment for each respondent
    # and remove respondents with very high and very low values
    squid = (
        squid_all_years.query("year not in @years_to_drop")
        .groupby(["respondent_id_ferc714", "respondent_name_ferc714"])
        .mean()
        .fillna(value=0)
    )
    # remove wild outliers that are more likely to be data errors or change of
    # planning area than forecast errors
    squid = (
        squid[
            (squid[1] >= -0.3)
            & (squid[1] <= 0.3)
            & (squid[9] >= -1.0)
            & (squid[9] <= 1.0)
        ]
        .replace(0.0, np.nan)
        .dropna(thresh=5)
    )
    # zero year should always be zero
    squid[0] = 0.0

    # calculate the weighted average forecast error of all respondents
    for_avg = (
        squid.reset_index()
        .merge(
            peak_demand.query("year == @load_weight_year")[
                ["respondent_id_ferc714", "actual"]
            ].rename(columns={"actual": "load"}),
            on=["respondent_id_ferc714"],
            how="inner",
            validate="1:1",
        )
        .set_index("respondent_id_ferc714")
    )
    squidavg = (
        (
            for_avg[list(range(0, 11))].multiply(for_avg.load, axis=0).sum()
            / for_avg.load.sum()
        )
        .to_frame()
        .T.set_axis(
            pd.MultiIndex.from_tuples(
                [("weighted_average", "Average")], names=squid.index.names
            )
        )
    )
    if export_csv:
        squid.to_csv("squid_data.csv")
        squidavg.to_csv("squid_weighted_average.csv")
    return SquidData(squid, squidavg, forecast, peak_demand)
