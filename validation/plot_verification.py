import os
import shutil
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HAS_LATEX = shutil.which('latex') is not None
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except ImportError:
    pass

plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": HAS_LATEX,
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.grid.which": "both",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 300
})
if HAS_LATEX:
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}"})


FULL_CONFIGS = [
    {
        'key': 'csl_vs_dadv', 'x_col': 'csl', 'y_col': 'd_adv',
        'color': '#D62728', 'shadow': '#f5b5b5',
        'title': r"\textbf{CSL vs.} $\bm{d_{adv}}$" if HAS_LATEX else r"CSL vs. $d_{adv}$",
        'xlabel': r"\textbf{CSL}" if HAS_LATEX else "CSL",
        'ylabel': r"$\bm{d_{adv}}$" if HAS_LATEX else r"$d_{adv}$"
    },
    {
        'key': 'csg_vs_dadv', 'x_col': 'csg', 'y_col': 'd_adv',
        'color': '#1F77B4', 'shadow': '#aec7e8',
        'title': r"\textbf{CSG vs.} $\bm{d_{adv}}$" if HAS_LATEX else r"CSG vs. $d_{adv}$",
        'xlabel': r"\textbf{CSG}" if HAS_LATEX else "CSG",
        'ylabel': r"$\bm{d_{adv}}$" if HAS_LATEX else r"$d_{adv}$"
    },
    {
        'key': 'csl_vs_csg', 'x_col': 'csl', 'y_col': 'csg',
        'color': '#E377C2', 'shadow': '#f7b6d2',
        'title': r"\textbf{CSL vs. CSG}" if HAS_LATEX else "CSL vs. CSG",
        'xlabel': r"\textbf{CSL}" if HAS_LATEX else "CSL",
        'ylabel': r"\textbf{CSG}" if HAS_LATEX else "CSG"
    },
    {
        'key': 'dadv_proxy_vs_dadv', 'x_col': 'd_adv_proxy', 'y_col': 'd_adv',
        'color': '#9467BD', 'shadow': '#c5b0d5',
        'title': r"$\bm{\bar{d}_{adv, T}}$ \textbf{vs.} $\bm{d_{adv}}$" if HAS_LATEX else r"$\bar{d}_{adv, T}$ vs. $d_{adv}$",
        'xlabel': r"$\bm{\bar{d}_{adv, T}}$" if HAS_LATEX else r"$\bar{d}_{adv, T}$",
        'ylabel': r"$\bm{d_{adv}}$" if HAS_LATEX else r"$d_{adv}$"
    }
]

MIA_CONFIGS = [
    {
        'key': 'csl_vs_yeom', 'x_col': 'csl', 'y_col': 'yeom_adv',
        'color': '#2CA02C', 'shadow': '#98df8a',
        'title': r"\textbf{CSL vs. Loss Threshold Adv.}" if HAS_LATEX else "CSL vs. Loss Threshold Adv.",
        'xlabel': r"\textbf{CSL}" if HAS_LATEX else "CSL",
        'ylabel': r"\textbf{Loss Threshold Adv.}" if HAS_LATEX else "Loss Threshold Adv."
    },
    {
        'key': 'csl_vs_lira', 'x_col': 'csl', 'y_col': 'lira_adv',
        'color': '#8C564B', 'shadow': '#c49c94',
        'title': r"\textbf{CSL vs. LiRA Advantage}" if HAS_LATEX else "CSL vs. LiRA Advantage",
        'xlabel': r"\textbf{CSL}" if HAS_LATEX else "CSL",
        'ylabel': r"\textbf{LiRA Advantage}" if HAS_LATEX else "LiRA Advantage"
    },
    {
        'key': 'yeom_vs_dadv', 'x_col': 'yeom_adv', 'y_col': 'd_adv',
        'color': '#FF7F0E', 'shadow': '#ffbb78',
        'title': r"\textbf{Loss Threshold Adv. vs.} $\bm{d_{adv}}$" if HAS_LATEX else r"Loss Threshold Adv. vs. $d_{adv}$",
        'xlabel': r"\textbf{Loss Threshold Adv.}" if HAS_LATEX else "Loss Threshold Adv.",
        'ylabel': r"$\bm{d_{adv}}$" if HAS_LATEX else r"$d_{adv}$"
    },
    {
        'key': 'lira_vs_dadv', 'x_col': 'lira_adv', 'y_col': 'd_adv',
        'color': '#17BECF', 'shadow': '#9edae5',
        'title': r"\textbf{LiRA Advantage vs.} $\bm{d_{adv}}$" if HAS_LATEX else r"LiRA Advantage vs. $d_{adv}$",
        'xlabel': r"\textbf{LiRA Advantage}" if HAS_LATEX else "LiRA Advantage",
        'ylabel': r"$\bm{d_{adv}}$" if HAS_LATEX else r"$d_{adv}$"
    }
]


def filter_valid_samples(df, x_col, y_col):
    sub = df.dropna(subset=[x_col, y_col]).copy()
    if ('d_adv' in (x_col, y_col)) and ('cw_success' in sub.columns):
        sub = sub[sub['cw_success'].astype(bool)]
    return sub


def compute_and_print_cw_stats(df, seeds, dataset, model, out_dir):
    rows = []
    for s in seeds:
        sub = df[df['seed'] == s].dropna(subset=['d_adv'])
        n_total = len(sub)
        if n_total == 0:
            continue
        succ = sub['cw_success'].astype(bool).values if 'cw_success' in sub.columns else np.ones(n_total, dtype=bool)
        init_mis = sub['init_misclassified'].astype(bool).values if 'init_misclassified' in sub.columns else (sub['d_adv'].values == 0.0)
        attacked = ~init_mis
        attacked_succ = attacked & succ
        failed = ~succ

        overall_asr = 100.0 * succ.mean()
        cond_asr = 100.0 * (attacked_succ.sum() / max(attacked.sum(), 1))
        valid_d = sub['d_adv'].values[succ]

        rows.append({
            'dataset': dataset,
            'model': model,
            'seed': str(s),
            'eval_samples': n_total,
            'init_misclassified': int(init_mis.sum()),
            'attacked_samples': int(attacked.sum()),
            'cw_flipped': int(attacked_succ.sum()),
            'cw_failed': int(failed.sum()),
            'conditional_asr_pct': cond_asr,
            'overall_valid_pct': overall_asr,
            'mean_d_adv': float(np.mean(valid_d)) if len(valid_d) > 0 else np.nan,
            'std_d_adv': float(np.std(valid_d)) if len(valid_d) > 0 else np.nan
        })

    df_asr = pd.DataFrame(rows)
    if len(df_asr) > 0:
        mean_cond_asr = df_asr['conditional_asr_pct'].mean()
        std_cond_asr = df_asr['conditional_asr_pct'].std() if len(df_asr) > 1 else 0.0
        mean_ov_asr = df_asr['overall_valid_pct'].mean()
        std_ov_asr = df_asr['overall_valid_pct'].std() if len(df_asr) > 1 else 0.0
        mean_d = df_asr['mean_d_adv'].mean()
        std_d = df_asr['mean_d_adv'].std() if len(df_asr) > 1 else 0.0

        print("\n==========================================================================================")
        print(f" C&W ATTACK SUCCESS RATE SUMMARY ({dataset.upper()} | {model.upper()})")
        print("==========================================================================================")
        print(df_asr[['seed', 'eval_samples', 'init_misclassified', 'cw_flipped', 'cw_failed',
                      'conditional_asr_pct', 'overall_valid_pct', 'mean_d_adv']].to_string(index=False))
        print("------------------------------------------------------------------------------------------")
        print(f" Across {len(seeds)} Seeds -> Conditional C&W ASR : {mean_cond_asr:.2f}% +/- {std_cond_asr:.2f}%")
        print(f" Across {len(seeds)} Seeds -> Overall Valid Rate  : {mean_ov_asr:.2f}% +/- {std_ov_asr:.2f}%")
        print(f" Across {len(seeds)} Seeds -> Mean Valid d_adv    : {mean_d:.4f} +/- {std_d:.4f}")
        print("==========================================================================================")

        asr_csv = os.path.join(out_dir, f"cw_asr_stats_{dataset}_{model}.csv")
        df_asr.to_csv(asr_csv, index=False)
        print(f"[+] Saved C&W ASR statistics to: {asr_csv}")


def bin_data(x, y, num_bins=10):
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]
    x_bins = np.array([np.mean(b) for b in np.array_split(x_sorted, num_bins) if len(b) > 0])
    y_bins = np.array([np.mean(b) for b in np.array_split(y_sorted, num_bins) if len(b) > 0])
    return x_bins, y_bins


def compute_decile_stats(df, x_col, y_col, seeds, num_bins=10):
    binned_x, binned_y = [], []
    for s in seeds:
        sub = filter_valid_samples(df[df['seed'] == s], x_col, y_col)
        if len(sub) == 0:
            continue
        bx, by = bin_data(sub[x_col].values, sub[y_col].values, num_bins)
        binned_x.append(bx)
        binned_y.append(by)

    min_len = min(len(bx) for bx in binned_x)
    b_x_arr = np.array([bx[:min_len] for bx in binned_x])
    b_y_arr = np.array([by[:min_len] for by in binned_y])

    mean_x = np.mean(b_x_arr, axis=0)
    mean_y = np.mean(b_y_arr, axis=0)
    std_y = np.std(b_y_arr, axis=0)
    return mean_x, mean_y, std_y


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def compute_alignment_metrics(df, x_col, y_col, seeds, rel_name, dataset, model):
    pearsons, spearmans, cosines, n_valids = [], [], [], []
    for s in seeds:
        sub = filter_valid_samples(df[df['seed'] == s], x_col, y_col)
        if len(sub) < 2:
            continue
        x = sub[x_col].values.astype(np.float64)
        y = sub[y_col].values.astype(np.float64)
        r_val, _ = stats.pearsonr(x, y)
        rho_val, _ = stats.spearmanr(x, y)
        cos_val = cosine_sim(x, y)
        pearsons.append(r_val)
        spearmans.append(rho_val)
        cosines.append(cos_val)
        n_valids.append(len(sub))

    return {
        'dataset': dataset,
        'model': model,
        'relationship': rel_name,
        'num_samples_per_seed': int(np.mean(n_valids)) if n_valids else 0,
        'pearson_mean': float(np.mean(pearsons)),
        'pearson_std': float(np.std(pearsons)),
        'spearman_mean': float(np.mean(spearmans)),
        'spearman_std': float(np.std(spearmans)),
        'cosine_mean': float(np.mean(cosines)),
        'cosine_std': float(np.std(cosines)),
        'pearson_str': f"{np.mean(pearsons):.3f} +/- {np.std(pearsons):.3f}",
        'spearman_str': f"{np.mean(spearmans):.3f} +/- {np.std(spearmans):.3f}",
        'cosine_str': f"{np.mean(cosines):.3f} +/- {np.std(cosines):.3f}"
    }


def render_axis(ax, x, mean_y, std_y, cfg):
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(mean_y - std_y), np.max(mean_y + std_y)

    x_span = max(x_max - x_min, 1e-5)
    y_span = max(y_max - y_min, 1e-5)

    ax.set_xlim(x_min - 0.08 * x_span, x_max + 0.08 * x_span)
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.12 * y_span)

    ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=cfg['shadow'], alpha=0.5, label=r'$\pm$ 1 Std. Dev.')
    ax.plot(x, mean_y, marker='o', markersize=4, color=cfg['color'], linewidth=1.2,
            markeredgecolor='black', markeredgewidth=0.5, label='Mean Trend')

    ax.set_title(cfg['title'], fontsize=12)
    ax.set_xlabel(cfg['xlabel'], fontsize=11)
    ax.set_ylabel(cfg['ylabel'], fontsize=11)
    ax.grid(True, which='major', color='#CCCCCC', linestyle='-', alpha=0.5)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=10)

    if HAS_LATEX:
        formatter = ticker.FuncFormatter(lambda val, pos: r'$\bm{' + f'{val:g}' + r'}$')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)


def main():
    parser = argparse.ArgumentParser(description="Plot Theoretical Verification Grid & Print ASR / Table 1 Scores")
    parser.add_argument('--full-csv', type=str, required=True, help='Full dataset results CSV')
    parser.add_argument('--mia-csv', type=str, default=None, help='Optional MIA half-dataset results CSV')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num-bins', type=int, default=10)
    parser.add_argument('--out-dir', type=str, default='./plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_full = pd.read_csv(args.full_csv)
    seeds_full = sorted(df_full['seed'].unique())

    has_mia = (args.mia_csv is not None) and os.path.exists(args.mia_csv)
    if has_mia:
        df_mia = pd.read_csv(args.mia_csv)
        seeds_mia = sorted(df_mia['seed'].unique())

    compute_and_print_cw_stats(df_full, seeds_full, args.dataset, args.model, args.out_dir)

    summary_rows = [
        compute_alignment_metrics(df_full, 'd_adv', 'd_adv_proxy', seeds_full, 'd_adv vs. d_adv_T (Proxy)', args.dataset, args.model),
        compute_alignment_metrics(df_full, 'csl', 'csg', seeds_full, 'CSL vs. CSG', args.dataset, args.model),
        compute_alignment_metrics(df_full, 'csl', 'd_adv', seeds_full, 'CSL vs. d_adv', args.dataset, args.model),
        compute_alignment_metrics(df_full, 'csg', 'd_adv', seeds_full, 'CSG vs. d_adv', args.dataset, args.model),
    ]
    if has_mia:
        summary_rows.extend([
            compute_alignment_metrics(df_mia, 'csl', 'lira_adv', seeds_mia, 'CSL vs. LiRA Adv.', args.dataset, args.model),
            compute_alignment_metrics(df_mia, 'csl', 'yeom_adv', seeds_mia, 'CSL vs. Yeom Loss Adv.', args.dataset, args.model),
        ])

    df_summary = pd.DataFrame(summary_rows)
    print("\n==========================================================================================")
    print(f" ALIGNMENT SCORES ({args.dataset.upper()} | {args.model.upper()} | N={summary_rows[0]['num_samples_per_seed']} valid samples/seed)")
    print("==========================================================================================")
    print(df_summary[['relationship', 'pearson_str', 'spearman_str', 'cosine_str']].to_string(index=False))
    print("==========================================================================================\n")

    scores_csv = os.path.join(args.out_dir, f"alignment_scores_{args.dataset}_{args.model}.csv")
    df_summary.to_csv(scores_csv, index=False)
    print(f"[+] Saved alignment scores table to: {scores_csv}")

    if has_mia:
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), dpi=300)
        for i, cfg in enumerate(FULL_CONFIGS):
            ax = axes[0, i]
            x, my, sy = compute_decile_stats(df_full, cfg['x_col'], cfg['y_col'], seeds_full, num_bins=args.num_bins)
            render_axis(ax, x, my, sy, cfg)
            if i == 0:
                ax.legend(frameon=True, fontsize=9, loc='upper right')

        for i, cfg in enumerate(MIA_CONFIGS):
            ax = axes[1, i]
            x, my, sy = compute_decile_stats(df_mia, cfg['x_col'], cfg['y_col'], seeds_mia, num_bins=args.num_bins)
            render_axis(ax, x, my, sy, cfg)
    else:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), dpi=300)
        for i, cfg in enumerate(FULL_CONFIGS):
            ax = axes[i]
            x, my, sy = compute_decile_stats(df_full, cfg['x_col'], cfg['y_col'], seeds_full, num_bins=args.num_bins)
            render_axis(ax, x, my, sy, cfg)
            if i == 0:
                ax.legend(frameon=True, fontsize=9, loc='upper right')

    fig.tight_layout(pad=0.6)
    out_base = os.path.join(args.out_dir, f"verification_grid_{args.dataset}_{args.model}_{args.num_bins}bins")
    fig.savefig(out_base + ".pdf", bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[+] Saved verification plot to: {out_base}.png and {out_base}.pdf")


if __name__ == '__main__':
    main()
