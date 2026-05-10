import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


NAME_TO_SCORE = {
    "Fully Supportive": 5,
    "Mostly Supportive": 4,
    "Moderately Supportive": 3,
    "Slightly Supportive": 2,
    "Not Supportive At All": 1,
}

BINARY_SCORE = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the lightweight annotation-study analysis."
    )
    parser.add_argument(
        "--csv",
        default="annotation_results/reasoning-chain-study.csv",
        help="Path to the annotation CSV shipped with the repo.",
    )
    parser.add_argument(
        "--outdir",
        default="reproduction/annotation",
        help="Directory for plots and summary files.",
    )
    return parser


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def correlation_dict(x: pd.Series, y: pd.Series) -> dict:
    corr = pearsonr(x, y)
    return {
        "statistic": float(corr.statistic),
        "pvalue": float(corr.pvalue),
    }


def scatter_plot(frame: pd.DataFrame, x_col: str, y_col: str, title: str, outpath: Path) -> None:
    ensure_parent(outpath)
    plt.figure(figsize=(4.5, 3.5))
    plt.scatter(frame[x_col], frame[y_col], alpha=0.8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["rating_score"] = df["rating"].map(NAME_TO_SCORE)
    df["binary_rating"] = df["rating_score"].map(BINARY_SCORE)
    df["dmass"] = df["dp"].astype(float)

    frame = df[["rating_score", "dmass"]].copy()
    binary_frame = df[["binary_rating", "dmass"]].copy()

    full_corr = correlation_dict(frame["rating_score"], frame["dmass"])
    binary_corr = correlation_dict(binary_frame["binary_rating"], binary_frame["dmass"])

    scatter_plot(
        frame,
        "rating_score",
        "dmass",
        f"Pearson r={full_corr['statistic']:.3f}, p={full_corr['pvalue']:.3g}",
        outdir / "rating_vs_dmass.png",
    )
    scatter_plot(
        binary_frame,
        "binary_rating",
        "dmass",
        f"Binary Pearson r={binary_corr['statistic']:.3f}, p={binary_corr['pvalue']:.3g}",
        outdir / "binary_rating_vs_dmass.png",
    )

    per_group = []
    grouped = df.groupby(["dataset", "model"], dropna=False)
    for (dataset, model), group in grouped:
        corr = correlation_dict(group["rating_score"], group["dmass"])
        per_group.append(
            {
                "dataset": dataset,
                "model": model,
                "n_rows": int(len(group)),
                "pearson_r": corr["statistic"],
                "pearson_p": corr["pvalue"],
                "flip_rate": float((group["flip"].astype(str) == "True").mean()),
            }
        )

    summary = {
        "n_rows": int(len(df)),
        "rating_distribution": df["rating"].value_counts().to_dict(),
        "flip_distribution": df["flip"].astype(str).value_counts().to_dict(),
        "overall_rating_vs_dmass": full_corr,
        "binary_rating_vs_dmass": binary_corr,
        "per_dataset_model": per_group,
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    pd.DataFrame(per_group).sort_values(["dataset", "model"]).to_csv(
        outdir / "per_dataset_model.csv", index=False
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
