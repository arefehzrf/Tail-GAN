"""
Generate regime-switching sliding-window datasets for Tail-GAN.

This script:
1. Simulates one long multivariate return series with two regimes:
   - normal
   - crisis
2. Uses a two-state Markov chain for regime persistence
3. Cuts the long series into sliding windows
4. Saves windows into:
   - gan_data/<data_name>__all
   - gan_data/<data_name>__normal
   - gan_data/<data_name>__crisis

It also saves:
- the long simulated series
- the regime path
- a labels file for all windows
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd


def make_corr_matrix(n_assets, rho):
    """Create an equicorrelation matrix."""
    corr = (1.0 - rho) * np.eye(n_assets) + rho * np.ones((n_assets, n_assets))
    np.fill_diagonal(corr, 1.0)
    return corr


def sample_multivariate_t(mean, cov, df, rng):
    """
    Draw one multivariate Student-t sample using the scale-mixture representation.

    Args:
        mean: Mean vector, shape (n_assets,)
        cov: Scale matrix, shape (n_assets, n_assets)
        df: Degrees of freedom
        rng: NumPy random generator

    Returns:
        np.ndarray of shape (n_assets,)
    """
    z = rng.multivariate_normal(mean=np.zeros(len(mean)), cov=cov)
    g = rng.chisquare(df) / df
    return mean + z / np.sqrt(g)


def simulate_regime_series(
    total_steps,
    tickers,
    p00,
    p11,
    normal_vol,
    crisis_vol,
    normal_rho,
    crisis_rho,
    normal_df,
    crisis_df,
    normal_phi,
    crisis_phi,
    normal_drift,
    crisis_drift,
    seed,
):
    """
    Simulate a long multivariate return series with two regimes.

    Regime 0 = normal
    Regime 1 = crisis
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    # Simulate regime path
    regimes = np.zeros(total_steps, dtype=int)
    for t in range(1, total_steps):
        if regimes[t - 1] == 0:
            regimes[t] = rng.choice([0, 1], p=[p00, 1.0 - p00])
        else:
            regimes[t] = rng.choice([1, 0], p=[p11, 1.0 - p11])

    returns = np.zeros((n_assets, total_steps), dtype=float)

    for t in range(total_steps):
        if regimes[t] == 0:
            vol_vec = np.full(n_assets, normal_vol)
            rho = normal_rho
            df = normal_df
            phi = normal_phi
            mean = np.full(n_assets, normal_drift)
        else:
            vol_vec = np.full(n_assets, crisis_vol)
            rho = crisis_rho
            df = crisis_df
            phi = crisis_phi
            mean = np.full(n_assets, crisis_drift)

        corr = make_corr_matrix(n_assets, rho)
        cov = np.outer(vol_vec, vol_vec) * corr
        eps_t = sample_multivariate_t(mean=mean, cov=cov, df=df, rng=rng)

        if t == 0:
            returns[:, t] = eps_t
        else:
            returns[:, t] = phi * returns[:, t - 1] + eps_t

    return returns, regimes


def save_long_series(returns, regimes, tickers, out_dir, data_name):
    """Save the full long simulated series and regime path."""
    os.makedirs(out_dir, exist_ok=True)

    returns_df = pd.DataFrame(returns.T, columns=tickers)
    regimes_df = pd.DataFrame(
        {
            "t": np.arange(len(regimes)),
            "regime": regimes,
        }
    )

    returns_df.to_csv(join(out_dir, f"{data_name}_long_returns.csv"), index=False)
    regimes_df.to_csv(join(out_dir, f"{data_name}_long_regimes.csv"), index=False)


def save_sliding_windows(
    returns,
    regimes,
    tickers,
    data_name,
    gan_data_path,
    labels_path,
    window,
    step,
    crisis_threshold,
):
    """
    Create and save sliding windows.

    A window is labeled crisis if the share of crisis observations
    in the window is >= crisis_threshold.
    """
    all_path = join(gan_data_path, f"{data_name}__all")
    normal_path = join(gan_data_path, f"{data_name}__normal")
    crisis_path = join(gan_data_path, f"{data_name}__crisis")

    os.makedirs(all_path, exist_ok=True)
    os.makedirs(normal_path, exist_ok=True)
    os.makedirs(crisis_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    label_rows = []
    idx_all = 1
    idx_normal = 1
    idx_crisis = 1

    for start in range(0, returns.shape[1] - window + 1, step):
        end = start + window

        window_returns = returns[:, start:end]
        window_regimes = regimes[start:end]
        crisis_share = float(window_regimes.mean())
        label = "crisis" if crisis_share >= crisis_threshold else "normal"

        df = pd.DataFrame(window_returns.T, columns=tickers)

        # Save to __all
        all_name = f"{idx_all:06d}.csv"
        df.to_csv(join(all_path, all_name), index=False)

        # Save to regime-specific folder
        if label == "normal":
            normal_name = f"{idx_normal:06d}.csv"
            df.to_csv(join(normal_path, normal_name), index=False)
            idx_normal += 1
        else:
            crisis_name = f"{idx_crisis:06d}.csv"
            df.to_csv(join(crisis_path, crisis_name), index=False)
            idx_crisis += 1

        label_rows.append(
            {
                "window_id_all": idx_all,
                "start": start,
                "end": end,
                "crisis_share": crisis_share,
                "label": label,
            }
        )
        idx_all += 1

    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(join(labels_path, f"{data_name}_labels.csv"), index=False)

    print(f"Saved {idx_all - 1} windows to {all_path}")
    print(f"Saved {idx_normal - 1} normal windows to {normal_path}")
    print(f"Saved {idx_crisis - 1} crisis windows to {crisis_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--your_path", type=str, required=True, help="Repo base path")
    parser.add_argument("--data_name", type=str, default="RegimeDemo", help="Base dataset name")

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["Gauss", "AR50", "AR-12", "GARCH-T5", "GARCH-T10"],
        help="Ticker names / column names",
    )

    parser.add_argument("--total_steps", type=int, default=600000, help="Length of long time series")
    parser.add_argument("--window", type=int, default=100, help="Sliding window length")
    parser.add_argument("--step", type=int, default=10, help="Sliding window step")
    parser.add_argument(
        "--crisis_threshold",
        type=float,
        default=0.30,
        help="Window is labeled crisis if crisis share >= this threshold",
    )

    # Markov transition probabilities
    parser.add_argument("--p00", type=float, default=0.995, help="P(normal -> normal)")
    parser.add_argument("--p11", type=float, default=0.96, help="P(crisis -> crisis)")

    # Normal regime parameters
    parser.add_argument("--normal_vol", type=float, default=0.002, help="Normal regime volatility")
    parser.add_argument("--normal_rho", type=float, default=0.15, help="Normal regime correlation")
    parser.add_argument("--normal_df", type=float, default=12.0, help="Normal regime t degrees of freedom")
    parser.add_argument("--normal_phi", type=float, default=0.05, help="Normal regime AR(1) coefficient")
    parser.add_argument("--normal_drift", type=float, default=0.0, help="Normal regime drift")

    # Crisis regime parameters
    parser.add_argument("--crisis_vol", type=float, default=0.008, help="Crisis regime volatility")
    parser.add_argument("--crisis_rho", type=float, default=0.85, help="Crisis regime correlation")
    parser.add_argument("--crisis_df", type=float, default=4.0, help="Crisis regime t degrees of freedom")
    parser.add_argument("--crisis_phi", type=float, default=0.20, help="Crisis regime AR(1) coefficient")
    parser.add_argument("--crisis_drift", type=float, default=-0.0005, help="Crisis regime drift")

    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    return parser.parse_args()


def main():
    """Run the full data-generation pipeline."""
    opt = parse_args()

    gan_data_path = join(opt.your_path, "gan_data")
    labels_path = join(opt.your_path, "regime_labels")
    long_series_path = join(opt.your_path, "long_series")

    os.makedirs(gan_data_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(long_series_path, exist_ok=True)

    returns, regimes = simulate_regime_series(
        total_steps=opt.total_steps,
        tickers=opt.tickers,
        p00=opt.p00,
        p11=opt.p11,
        normal_vol=opt.normal_vol,
        crisis_vol=opt.crisis_vol,
        normal_rho=opt.normal_rho,
        crisis_rho=opt.crisis_rho,
        normal_df=opt.normal_df,
        crisis_df=opt.crisis_df,
        normal_phi=opt.normal_phi,
        crisis_phi=opt.crisis_phi,
        normal_drift=opt.normal_drift,
        crisis_drift=opt.crisis_drift,
        seed=opt.seed,
    )

    save_long_series(
        returns=returns,
        regimes=regimes,
        tickers=opt.tickers,
        out_dir=long_series_path,
        data_name=opt.data_name,
    )

    save_sliding_windows(
        returns=returns,
        regimes=regimes,
        tickers=opt.tickers,
        data_name=opt.data_name,
        gan_data_path=gan_data_path,
        labels_path=labels_path,
        window=opt.window,
        step=opt.step,
        crisis_threshold=opt.crisis_threshold,
    )


if __name__ == "__main__":
    main()
