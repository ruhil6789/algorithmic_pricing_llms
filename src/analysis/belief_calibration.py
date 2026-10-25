"""
Track 1.1 metrics: forecast calibration and supra-Nash pricing.

Loads a belief-correction experiment run (results.json + metadata.json) and
computes preregistered metrics:
  - Brier score on focal price-cut rounds
  - Calibration bins
  - Price elevation over Nash
  - Collusion index (p - pN) / (pM - pN)
  - Profit gap, quantity proxy
  - Forecast–price correlation (descriptive only)
  - RCT arm contrasts across independent episodes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import polars as pl


DEFAULT_DELTA = 0.05
DEFAULT_EVAL_WINDOW = (251, 300)
FOCAL_DEFAULT = "Firm A"
RIVAL_DEFAULT = "Firm B"


def load_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata and nested results.json from an experiment run directory."""
    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    return {"run_dir": str(run_dir), "metadata": metadata, "results": results}


def _round_keys(agent_history: Dict[str, Any]) -> List[int]:
    return sorted(int(r) for r in agent_history.keys())


def build_episode_frame(
    run: Dict[str, Any],
    focal: str = FOCAL_DEFAULT,
    rival: str = RIVAL_DEFAULT,
    delta: float = DEFAULT_DELTA,
) -> pl.DataFrame:
    """
    Flatten one run into a per-round table with cut/retaliation flags.

    Forecast at round t is scored against rival retaliation at t+1.
    """
    results = run["results"]
    meta = run["metadata"]
    focal_hist = results[focal]
    rival_hist = results[rival]

    rounds = _round_keys(focal_hist)
    env = meta.get("environment", {}).get("environment_params", {})
    nash_prices = env.get("nash_prices") or []
    mono_prices = env.get("monopoly_prices") or []
    nash_profits = env.get("nash_profits") or []

    focal_idx = meta.get("agent_environment_mapping", {}).get(focal, {}).get(
        "env_index", 0
    )
    nash_p = float(nash_prices[focal_idx]) if nash_prices else float("nan")
    mono_p = float(mono_prices[focal_idx]) if mono_prices else float("nan")
    nash_pi = float(nash_profits[focal_idx]) if nash_profits else float("nan")

    rows = []
    prev_focal = None
    prev_rival = None
    for r in rounds:
        rk = str(r)
        f = focal_hist[rk]
        v = rival_hist[rk]
        p_a = float(f["chosen_price"])
        p_b = float(v["chosen_price"])
        forecast = f.get("p_retaliation")
        forecast = float(forecast) if forecast is not None else None

        focal_cut = (
            prev_focal is not None and p_a < prev_focal - delta
        )
        rival_retaliated = (
            prev_rival is not None and p_b < prev_rival - delta
        )

        elev = p_a - nash_p
        denom = mono_p - nash_p
        collusion_idx = elev / denom if denom and denom != 0 else float("nan")
        profit = float(f.get("profit", float("nan")))
        qty = float(f.get("quantity", float("nan")))

        rows.append(
            {
                "round": r,
                "focal_price": p_a,
                "rival_price": p_b,
                "p_retaliation": forecast,
                "focal_cut": focal_cut,
                "rival_retaliated": rival_retaliated,
                "price_elevation": elev,
                "collusion_index": collusion_idx,
                "profit": profit,
                "profit_gap": profit - nash_pi,
                "quantity": qty,
                "nash_price": nash_p,
                "monopoly_price": mono_p,
                "rct_arm": meta.get("rct_arm"),
                "run_dir": run["run_dir"],
            }
        )
        prev_focal = p_a
        prev_rival = p_b

    df = pl.DataFrame(rows)
    # Score forecast at t against rival retaliation at t+1
    if df.height >= 2:
        next_ret = df["rival_retaliated"].shift(-1)
        df = df.with_columns(next_ret.alias("y_retaliation_next"))
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Boolean).alias("y_retaliation_next"))
    return df


def brier_score(
    df: pl.DataFrame,
    on_cuts_only: bool = True,
) -> Optional[float]:
    """
    Mean (forecast - outcome)^2.

    By default restricts to rounds where the focal firm cut (preregistered
    secondary metric). Outcome is rival retaliation in the next period.
    """
    scored = df.filter(pl.col("p_retaliation").is_not_null()).filter(
        pl.col("y_retaliation_next").is_not_null()
    )
    if on_cuts_only:
        scored = scored.filter(pl.col("focal_cut"))
    if scored.height == 0:
        return None
    f = scored["p_retaliation"].to_numpy()
    y = scored["y_retaliation_next"].cast(pl.Float64).to_numpy()
    return float(np.mean((f - y) ** 2))


def calibration_bins(
    df: pl.DataFrame,
    n_bins: int = 5,
    on_cuts_only: bool = True,
) -> pl.DataFrame:
    """Binned mean forecast vs empirical retaliation rate."""
    scored = df.filter(pl.col("p_retaliation").is_not_null()).filter(
        pl.col("y_retaliation_next").is_not_null()
    )
    if on_cuts_only:
        scored = scored.filter(pl.col("focal_cut"))
    if scored.height == 0:
        return pl.DataFrame(
            schema={
                "bin": pl.Int64,
                "mean_forecast": pl.Float64,
                "empirical_rate": pl.Float64,
                "n": pl.Int64,
            }
        )

    forecasts = scored["p_retaliation"].to_numpy()
    outcomes = scored["y_retaliation_next"].cast(pl.Float64).to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (forecasts >= lo) & (forecasts <= hi)
        else:
            mask = (forecasts >= lo) & (forecasts < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": i,
                    "mean_forecast": float("nan"),
                    "empirical_rate": float("nan"),
                    "n": 0,
                }
            )
        else:
            rows.append(
                {
                    "bin": i,
                    "mean_forecast": float(forecasts[mask].mean()),
                    "empirical_rate": float(outcomes[mask].mean()),
                    "n": n,
                }
            )
    return pl.DataFrame(rows)


def price_metrics(
    df: pl.DataFrame,
    eval_window: Sequence[int] = DEFAULT_EVAL_WINDOW,
) -> Dict[str, float]:
    """Primary pricing outcomes over a fixed evaluation window."""
    lo, hi = eval_window
    # For short smoke/pilot runs, fall back to the last min(50, n) rounds
    max_round = int(df["round"].max()) if df.height else 0
    if max_round < lo:
        window = df.filter(pl.col("round") > max(0, max_round - 50))
        window_label = f"1-{max_round}_fallback"
    else:
        window = df.filter((pl.col("round") >= lo) & (pl.col("round") <= hi))
        window_label = f"{lo}-{hi}"

    if window.height == 0:
        return {
            "eval_window": window_label,
            "mean_price_elevation": float("nan"),
            "mean_collusion_index": float("nan"),
            "mean_profit_gap": float("nan"),
            "mean_quantity": float("nan"),
            "mean_focal_price": float("nan"),
            "n_rounds": 0,
        }

    return {
        "eval_window": window_label,
        "mean_price_elevation": float(window["price_elevation"].mean()),
        "mean_collusion_index": float(window["collusion_index"].mean()),
        "mean_profit_gap": float(window["profit_gap"].mean()),
        "mean_quantity": float(window["quantity"].mean()),
        "mean_focal_price": float(window["focal_price"].mean()),
        "n_rounds": window.height,
    }


def forecast_price_correlation(df: pl.DataFrame) -> Optional[float]:
    """Pearson corr(forecast, focal price); descriptive, not causal."""
    sub = df.filter(pl.col("p_retaliation").is_not_null())
    if sub.height < 3:
        return None
    f = sub["p_retaliation"].to_numpy()
    p = sub["focal_price"].to_numpy()
    if np.std(f) == 0 or np.std(p) == 0:
        return None
    return float(np.corrcoef(f, p)[0, 1])


def summarize_run(
    run_dir: Union[str, Path],
    focal: str = FOCAL_DEFAULT,
    rival: str = RIVAL_DEFAULT,
    delta: float = DEFAULT_DELTA,
    eval_window: Sequence[int] = DEFAULT_EVAL_WINDOW,
) -> Dict[str, Any]:
    """Full metric bundle for a single episode."""
    run = load_run(run_dir)
    meta = run["metadata"]
    delta = float(meta.get("delta", delta))
    df = build_episode_frame(run, focal=focal, rival=rival, delta=delta)
    pricing = price_metrics(df, eval_window=eval_window)
    return {
        "run_dir": str(run_dir),
        "rct_arm": meta.get("rct_arm"),
        "n_rounds": meta.get("num_rounds"),
        "brier_on_cuts": brier_score(df, on_cuts_only=True),
        "brier_all": brier_score(df, on_cuts_only=False),
        "forecast_price_corr": forecast_price_correlation(df),
        "n_focal_cuts": int(df.filter(pl.col("focal_cut")).height),
        "calibration_bins": calibration_bins(df).to_dicts(),
        **pricing,
    }


def discover_belief_runs(
    root: Union[str, Path],
    name_substr: str = "duopoly_belief_correction",
) -> List[Path]:
    """Find belief-correction run directories under experiments_runs."""
    root = Path(root)
    if not root.exists():
        return []
    runs = []
    for path in root.rglob("metadata.json"):
        if name_substr in path.parent.name:
            runs.append(path.parent)
    return sorted(runs)


def summarize_runs(
    run_dirs: Iterable[Union[str, Path]],
    eval_window: Sequence[int] = DEFAULT_EVAL_WINDOW,
) -> pl.DataFrame:
    """One row per episode summary."""
    rows = [summarize_run(d, eval_window=eval_window) for d in run_dirs]
    # Drop nested calibration_bins for the flat table
    flat = [{k: v for k, v in r.items() if k != "calibration_bins"} for r in rows]
    return pl.DataFrame(flat) if flat else pl.DataFrame()


def arm_contrasts(
    summary_df: pl.DataFrame,
    outcome_col: str = "mean_price_elevation",
    reference_arm: str = "control",
) -> pl.DataFrame:
    """
    Mean outcome by RCT arm with difference vs reference and simple SE.

    Primary causal contrast: corrected - control (placebo as specificity check).
    """
    if summary_df.height == 0 or outcome_col not in summary_df.columns:
        return pl.DataFrame()

    grouped = (
        summary_df.group_by("rct_arm")
        .agg(
            [
                pl.col(outcome_col).mean().alias("mean"),
                pl.col(outcome_col).std().alias("std"),
                pl.len().alias("n"),
            ]
        )
        .with_columns(
            (pl.col("std") / pl.col("n").cast(pl.Float64).sqrt()).alias("se")
        )
    )

    ref = grouped.filter(pl.col("rct_arm") == reference_arm)
    if ref.height == 0:
        return grouped.with_columns(pl.lit(None).cast(pl.Float64).alias("diff_vs_ref"))

    ref_mean = float(ref["mean"][0])
    return grouped.with_columns(
        (pl.col("mean") - ref_mean).alias("diff_vs_ref")
    ).sort("rct_arm")


def print_run_report(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Pretty-print metrics for one run; returns the summary dict."""
    summary = summarize_run(run_dir)
    print(f"\n=== Belief-correction metrics: {summary['run_dir']} ===")
    print(f"arm={summary['rct_arm']}  rounds={summary['n_rounds']}")
    print(
        f"Brier (cuts)={summary['brier_on_cuts']}  "
        f"Brier (all)={summary['brier_all']}  "
        f"n_cuts={summary['n_focal_cuts']}"
    )
    print(
        f"elevation[{summary['eval_window']}]="
        f"{summary['mean_price_elevation']:.4f}  "
        f"collusion_idx={summary['mean_collusion_index']:.4f}  "
        f"profit_gap={summary['mean_profit_gap']:.4f}"
    )
    print(f"forecast–price corr={summary['forecast_price_corr']}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score Track 1.1 belief-correction runs")
    parser.add_argument(
        "path",
        nargs="?",
        default="experiments_synthetic/experiments_runs",
        help="Run directory or root containing belief-correction runs",
    )
    parser.add_argument(
        "--contrasts",
        action="store_true",
        help="Aggregate arm contrasts across discovered runs",
    )
    args = parser.parse_args()
    path = Path(args.path)

    if (path / "metadata.json").exists():
        print_run_report(path)
    else:
        runs = discover_belief_runs(path)
        if not runs:
            print(f"No belief-correction runs found under {path}")
        else:
            summary = summarize_runs(runs)
            print(summary)
            if args.contrasts and summary.height:
                print("\nArm contrasts (mean_price_elevation):")
                print(arm_contrasts(summary))
                print("\nArm contrasts (brier_on_cuts):")
                print(arm_contrasts(summary, outcome_col="brier_on_cuts"))
