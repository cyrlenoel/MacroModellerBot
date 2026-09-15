"""Assemble the two-sector open-economy HANK / RANK DAG."""

from __future__ import annotations

import sequence_jacobian as sj

from .fiscal import government
from .households import household_ha, household_rank
from .monetary import ex_post_rate, fisher, taylor_res
from .open_economy import market_clearing, nfa_uip, trade_demand
from .production import hours, labor_income, mrs_block, prices, wage_pcs

UNKNOWN_LIST = ["Y_T", "Y_NT", "w_T", "w_NT", "pi", "Q", "i", "B", "nfa"]
TARGET_LIST = [
    "asset_mkt",
    "goods_T",
    "pc_T",
    "pc_NT",
    "uip_res",
    "cpi_res",
    "i_res",
    "B_res",
    "nfa_res",
]


def build_model(household, name: str):
    """Combine household + aggregate blocks. `household` is HA, RANK, or sticky JacobianDict."""
    return sj.create_model(
        [
            household,
            taylor_res,
            fisher,
            ex_post_rate,
            prices,
            government,
            hours,
            labor_income,
            mrs_block,
            wage_pcs,
            trade_demand,
            nfa_uip,
            market_clearing,
        ],
        name=name,
    )


def hank_fire_model():
    return build_model(household_ha, "hank_fire")


def rank_model():
    return build_model(household_rank, "rank")


def ss_to_calibration(ss) -> dict:
    """Pack the SteadyState dataclass plus SSJ household extras into a dict."""
    d = ss.as_sj()
    d["beta"] = ss.beta  # used by wage PC and RANK Euler
    d["sd_e"] = ss.sigma_e
    d["r_ante"] = ss.r
    d["Y_T_d"] = ss.Y_T
    d["Y_NT_d"] = ss.Y_NT
    return d
