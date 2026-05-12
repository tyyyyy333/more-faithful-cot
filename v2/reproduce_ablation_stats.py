import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ["arc-challenge", "openbook", "sports", "sqa"]
MODELS = ["Phi-3", "LLaMA-3", "LLaMA-3-3B", "Mistral-2"]

MODEL_TO_NICE = {
    "Phi-3": "Phi-3",
    "LLaMA-3": "LLaMA-3-8B",
    "LLaMA-3-3B": "LLaMA-3-3B",
    "Mistral-2": "Mistral-2",
}

DATASET_TO_NICE = {
    "arc-challenge": "ARC-Challenge",
    "openbook": "OpenBookQA",
    "sqa": "StrategyQA",
    "sports": "Sports",
}

MODEL_COLOR = {
    "Phi-3": "tab:blue",
    "LLaMA-3": "tab:red",
    "LLaMA-3-3B": "tab:orange",
    "Mistral-2": "tab:green",
}

METHOD_SHAPE = {
    "npo_grad_diff": "o",
    "npo_KL": "*",
}

BEST_LR = {
    "arc-challenge": {"Phi-3": 1e-04, "LLaMA-3": 1e-05, "LLaMA-3-3B": 3e-05, "Mistral-2": 5e-06},
    "openbook": {"Phi-3": 1e-04, "LLaMA-3": 1e-05, "LLaMA-3-3B": 3e-05, "Mistral-2": 5e-06},
    "sports": {"Phi-3": 5e-05, "LLaMA-3": 5e-06, "LLaMA-3-3B": 3e-05, "Mistral-2": 3e-06},
    "sqa": {"Phi-3": 5e-05, "LLaMA-3": 1e-05, "LLaMA-3-3B": 3e-05, "Mistral-2": 5e-06},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the lightweight ablation/full-results aggregation without loading models."
    )
    parser.add_argument("--ablation-root", default="ablation", help="Directory containing ablation outputs.")
    parser.add_argument("--results-root", default="results", help="Directory containing full results.")
    parser.add_argument("--outdir", default="reproduction/ablation", help="Output directory.")
    parser.add_argument("--method", default="npo_KL", help="Method name used in filenames.")
    parser.add_argument("--run-type", default="sentencize", help="Run type used in filenames.")
    parser.add_argument("--seed", type=int, default=1001, help="Run seed used in filenames.")
    parser.add_argument(
        "--pos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expect pos=True in filenames (use --no-pos for pos=False).",
    )
    parser.add_argument(
        "--ff2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expect ff2=True in filenames (use --no-ff2 for ff2=False).",
    )
    return parser


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open() as infile:
        for line in infile:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def list_learning_rates(root: Path) -> dict:
    dataset_model_lrs = {}
    for dataset in DATASETS:
        for model in MODELS:
            key = f"{dataset}_{model}"
            model_dir = root / dataset / model
            lrs = []
            if model_dir.exists():
                for file in model_dir.iterdir():
                    parts = file.name.split("_")
                    for part in parts:
                        if part.startswith("lr="):
                            lrs.append(float(part.replace("lr=", "").replace(".out", "")))
                            break
            dataset_model_lrs[key] = sorted(set(lrs))
    return dataset_model_lrs


def unique_instances(result_dict: list) -> int:
    return len({item["question"] for item in result_dict})


def instance_specificity(instance_outputs: dict) -> list:
    if "0" not in instance_outputs:
        return []

    baseline = instance_outputs["0"].get("specificity_preds") or []
    if not baseline:
        return []

    specificity = []
    initial_predictions = np.array(baseline)
    for key, value in sorted(instance_outputs.items(), key=lambda t: int(t[0])):
        if key == "0":
            continue
        preds = np.array(value.get("specificity_preds") or [])
        if len(preds) == 0:
            continue
        specificity.append(float((initial_predictions == preds).sum() / len(preds) * 100.0))
    return specificity


def compute_specificity(results: list) -> tuple[float, list]:
    spec_through_iters = {}
    for result in results:
        spec = instance_specificity(result["unlearning_results"])
        if not spec:
            continue
        for i, value in enumerate(spec):
            spec_through_iters.setdefault(i, []).append(value)
    if not spec_through_iters:
        return float("nan"), []
    avg_spec = [float(np.mean(values)) for _, values in sorted(spec_through_iters.items())]
    flat = np.concatenate([np.array(values) for values in spec_through_iters.values()])
    return float(np.mean(flat)), avg_spec


def average_efficacy(results: list, step: bool = True) -> tuple[float, list]:
    eff_through_iters = {}
    key = "cot_step_prob" if step else "cot_prob"
    for unlearned_step in results:
        unlearning_results = unlearned_step["unlearning_results"]
        if "0" not in unlearning_results:
            continue
        probabilities = []
        for _, r in sorted(unlearning_results.items(), key=lambda t: int(t[0])):
            if key not in r or not r[key]:
                probabilities = []
                break
            probabilities.append(np.exp(r[key][0]))
        if not probabilities:
            continue
        p0 = probabilities[0]
        for i, prob in enumerate(probabilities):
            eff_through_iters.setdefault(i, [])
            eff_through_iters[i].append(0.0 if i == 0 else (1 - prob / p0) * 100.0)
    if not eff_through_iters:
        return float("nan"), []
    flat = np.concatenate([np.array(values) for values in eff_through_iters.values()])
    first = list(eff_through_iters.values())[0]
    return float(np.mean(flat)), first


def instance_changed_prediction(epoch_results: dict) -> tuple[bool, list]:
    if "0" in epoch_results:
        preds = [int(np.argmax(r["probs"])) for _, r in sorted(epoch_results.items(), key=lambda t: int(t[0]))]
        flips = [pred != preds[0] for pred in preds]
        return any(flips), flips

    # Fallback for runs that only store the final evaluation.
    keys = sorted(epoch_results.keys(), key=lambda k: int(k))
    if not keys:
        return False, []
    final_pred = int(np.argmax(epoch_results[keys[-1]]["probs"]))
    initial_pred = int(epoch_results[keys[-1]].get("prediction", final_pred))
    return final_pred != initial_pred, [initial_pred != final_pred]


def changed_prediction(results: list) -> float:
    unique_questions = set()
    unique_flips = set()
    for result in results:
        unique_questions.add(result["question"])
        changed, _ = instance_changed_prediction(result["unlearning_results"])
        if changed:
            unique_flips.add(result["question"])
    if not unique_questions:
        return float("nan")
    return float(len(unique_flips) / len(unique_questions) * 100.0)


def make_stats(per_instance_results: list) -> dict:
    faithfulness = changed_prediction(per_instance_results)
    specificity, _ = compute_specificity(per_instance_results)
    efficacy, _ = average_efficacy(per_instance_results, step=True)
    return {
        "n_instances": unique_instances(per_instance_results),
        "faithfulness": faithfulness,
        "efficacy": efficacy,
        "specificity": specificity,
        "n_cot_steps": len(per_instance_results),
    }


def make_filename(method: str, run_type: str, lr: float, seed: int, pos: bool, ff2: bool) -> str:
    return f"{method}_{run_type}_s=True_lr={lr}_rs={seed}_pos={pos}_ff2={ff2}.out"


def match_result_files(root: Path, method: str, run_type: str, lr: float, seed: int, pos: bool, ff2: bool) -> list[Path]:
    prefix = f"{method}_{run_type}_s=True_lr={lr}_rs={seed}_pos={pos}_ff2={ff2}"
    return sorted(root.glob(f"{prefix}*.out"))


def traverse_stats(root: Path, method: str, run_type: str, seed: int, pos: bool, ff2: bool) -> tuple[dict, list]:
    extracted_lrs = list_learning_rates(root)
    nested = {}
    flat_rows = []
    for dataset in DATASETS:
        for model in MODELS:
            key = f"{dataset}_{model}"
            for lr in extracted_lrs[key]:
                paths = match_result_files(root / dataset / model, method, run_type, lr, seed, pos, ff2)
                if not paths:
                    continue
                for path in paths:
                    per_instance_results = load_jsonl(path)
                    if not per_instance_results:
                        continue
                    nested.setdefault(dataset, {}).setdefault(model, {}).setdefault(method, {})[lr] = make_stats(per_instance_results)
                    flat_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "method": method,
                            "lr": lr,
                            "file": path.name,
                            **nested[dataset][model][method][lr],
                        }
                    )
    return nested, flat_rows


def load_best_full_results(root: Path, method: str, run_type: str, seed: int, pos: bool, ff2: bool) -> list:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            lr = BEST_LR[dataset][model]
            paths = match_result_files(root / dataset / model, method, run_type, lr, seed, pos, ff2)
            for path in paths:
                results = load_jsonl(path)
                if not results:
                    continue
                stats = make_stats(results)
                rows.append({"dataset": dataset, "model": model, "lr": lr, "file": path.name, **stats})
    return rows


def scatter_results(dataset_results: dict, outpath: Path) -> None:
    if not dataset_results:
        return
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    major_ticks = np.arange(0, 101, 20)
    for idx, (dataset, model_results) in enumerate(sorted(dataset_results.items())):
        row = idx // 2
        col = idx % 2
        ax = axs[row][col]
        ax.set_ylim(-5, 105)
        ax.set_xlim(-5, 105)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid()
        for model, method_results in model_results.items():
            for method, lr_results in method_results.items():
                xs, ys, sizes = [], [], []
                for lr, res in sorted(lr_results.items()):
                    xs.append(res["efficacy"])
                    ys.append(res["specificity"])
                    sizes.append(50 + res["faithfulness"] / 100.0 * 150)
                ax.scatter(
                    xs,
                    ys,
                    marker=METHOD_SHAPE.get(method, "o"),
                    facecolors="none",
                    edgecolors=MODEL_COLOR[model],
                    s=sizes,
                    label=f"{MODEL_TO_NICE[model]}-{method}",
                )
        ax.set_title(DATASET_TO_NICE[dataset], fontweight="bold")
        if col == 0:
            ax.set_ylabel("Specificity")
        if row == 1:
            ax.set_xlabel("Efficacy")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=160)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ablation_root = Path(args.ablation_root)
    results_root = Path(args.results_root)

    ablation_nested, ablation_rows = traverse_stats(
        ablation_root, args.method, args.run_type, args.seed, args.pos, args.ff2
    )
    best_rows = load_best_full_results(
        results_root, args.method, args.run_type, args.seed, args.pos, args.ff2
    )

    summary = {
        "ablation_root": str(ablation_root),
        "results_root": str(results_root),
        "ablation_files_found": len(ablation_rows),
        "best_result_files_found": len(best_rows),
        "missing_ablation_root": not ablation_root.exists(),
        "missing_results_root": not results_root.exists(),
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if ablation_rows:
        pd.DataFrame(ablation_rows).sort_values(["dataset", "model", "lr"]).to_csv(
            outdir / "ablation_stats.csv", index=False
        )
        scatter_results(ablation_nested, outdir / "lr_ablation.png")
    if best_rows:
        pd.DataFrame(best_rows).sort_values(["dataset", "model"]).to_csv(
            outdir / "best_full_results.csv", index=False
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not ablation_rows:
        print("No ablation result files found yet.")
    if not best_rows:
        print("No full result files found yet.")
    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
