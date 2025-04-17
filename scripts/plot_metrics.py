#!/usr/bin/env python3
"""
Plot evaluation metrics and Q-value histograms for CarRacing agents.
Reads eval_*.json files for each algorithm and corresponding q_hist .npy files.
Produces bar charts and histogram figures in the specified output directory.
"""
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_eval_results(eval_dir):
    """Load JSON evaluation results from eval_<algo>.json files."""
    results = {}
    pattern = os.path.join(eval_dir, "eval_*.json")
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        algo = fname.replace("eval_", "").replace(".json", "")
        with open(path, 'r') as f:
            results[algo] = json.load(f)
    return results


def plot_bar_metric(results, mean_key, std_key, ylabel, out_path):
    """Create a bar chart with error bars for a given metric."""
    algos = list(results.keys())
    try:
        means = [results[a][mean_key] for a in algos]
        stds  = [results[a][std_key]  for a in algos]
    except KeyError as e:
        print(f"Missing metric {e} in results; skipping {ylabel}.")
        return
    x = np.arange(len(algos))
    plt.figure(figsize=(8, 6))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, algos, rotation=45)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by Algorithm")
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved {ylabel} plot to {out_path}")
    plt.close()


def plot_q_hist(q_hist_dir, algos, out_path):
    """Plot overlaid histograms of Q-values for specified algorithms."""
    plt.figure(figsize=(8, 6))
    found = False
    for algo in algos:
        q_path = os.path.join(q_hist_dir, f"{algo}_q_hist.npy")
        if os.path.exists(q_path):
            q_vals = np.load(q_path).flatten()
            plt.hist(q_vals, bins=50, alpha=0.5, label=algo)
            found = True
    if not found:
        print("No Q-histogram files found; skipping Q-value histogram.")
        return
    plt.xlabel("Q-value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Q-value Distributions (DQN Variants)")
    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved Q-value histogram plot to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics and Q-histograms for CarRacing agents"
    )
    parser.add_argument(
        '--eval_dir', type=str, default='.',
        help='Directory containing eval_<algo>.json files'
    )
    parser.add_argument(
        '--q_hist_dir', type=str, default='.',
        help='Directory containing <algo>_q_hist.npy files'
    )
    parser.add_argument(
        '--out_dir', type=str, default='figures',
        help='Output directory for plot images'
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load evaluation results
    results = load_eval_results(args.eval_dir)
    if not results:
        print(f"No evaluation JSON files found in {args.eval_dir}")
        return

    # Plot mean reward
    plot_bar_metric(
        results,
        mean_key='mean_reward',
        std_key='std_reward',
        ylabel='Mean Reward',
        out_path=os.path.join(args.out_dir, 'mean_reward.png')
    )

    # Plot mean episode length
    plot_bar_metric(
        results,
        mean_key='mean_length',
        std_key='std_length',
        ylabel='Mean Episode Length',
        out_path=os.path.join(args.out_dir, 'mean_length.png')
    )

    # Plot tiles visited if available
    # Compute mean and std for 'tiles' list
    if 'tiles' in next(iter(results.values())):
        # Add computed keys
        for algo, data in results.items():
            tiles = data.get('tiles', [])
            data['tiles_mean'] = float(np.mean(tiles))
            data['tiles_std'] = float(np.std(tiles))
        plot_bar_metric(
            results,
            mean_key='tiles_mean',
            std_key='tiles_std',
            ylabel='Tiles Visited (%)',
            out_path=os.path.join(args.out_dir, 'tiles_visited.png')
        )

    # Plot Q-value histograms for DQN variants
    dqn_algos = ['DQN', 'DoubleDQN', 'DuelingDQN', 'PERDQN']
    plot_q_hist(
        q_hist_dir=args.q_hist_dir,
        algos=dqn_algos,
        out_path=os.path.join(args.out_dir, 'q_hist.png')
    )

    print(f"Plots saved in {args.out_dir}")

if __name__ == '__main__':
    main()
