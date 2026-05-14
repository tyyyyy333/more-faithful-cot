from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


PROJECTION_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query_key_value",
    "c_attn",
    "Wqkv",
)


@dataclass
class DiagnosticPrompts:
    cot_prefix: str
    full_cot: str
    step_text: str
    answer_prompt: str
    removed_cot: str
    removed_answer_prompt: str


def _model_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _safe_detach_scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return _safe_detach_scalar(F.cosine_similarity(a.float(), b.float(), dim=0))


def _find_char_span(text: str, needle: str) -> Optional[tuple[int, int]]:
    start = text.find(needle)
    if start < 0:
        return None
    return start, start + len(needle)


def _tokenize_with_offsets(tokenizer, text: str, device: torch.device):
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded["offset_mapping"][0].tolist()
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        return input_ids, attention_mask, offsets, []
    except Exception as exc:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        return input_ids, attention_mask, None, [f"offset_mapping unavailable: {exc}"]


def _char_span_to_token_indices(offsets, char_span: tuple[int, int]) -> list[int]:
    if offsets is None:
        return []
    start_char, end_char = char_span
    indices = []
    for token_idx, (token_start, token_end) in enumerate(offsets):
        if token_end <= start_char:
            continue
        if token_start >= end_char:
            break
        indices.append(token_idx)
    return indices


def _forward_with_optional_captures(model, input_ids, attention_mask, capture_projection_modules):
    projection_outputs = {}
    hooks = []

    def make_hook(module_name):
        def hook(_, __, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 3:
                projection_outputs[module_name] = tensor.detach()
        return hook

    for module_name, module in capture_projection_modules:
        hooks.append(module.register_forward_hook(make_hook(module_name)))

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
    finally:
        for hook in hooks:
            hook.remove()

    return outputs, projection_outputs


def _iter_projection_modules(model, limit: int):
    matches = []
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(module_name.endswith(suffix) for suffix in PROJECTION_SUFFIXES):
            matches.append((module_name, module))
    if limit > 0:
        matches = matches[:limit]
    return matches


def _build_prompts(dh, target, step_idx: int) -> Optional[DiagnosticPrompts]:
    segmented = target.get("segmented_cot")
    if not segmented or step_idx >= len(segmented):
        return None
    cot_prefix = dh.make_cot_prompt(target["raw_instance"])
    step_text = segmented[step_idx]
    removed_steps = [step for idx, step in enumerate(segmented) if idx != step_idx]
    removed_cot = "\n".join(removed_steps)
    full_cot = target["cot"]
    return DiagnosticPrompts(
        cot_prefix=cot_prefix,
        full_cot=full_cot,
        step_text=step_text,
        answer_prompt=dh.make_answer_prompt(cot_prefix + full_cot),
        removed_cot=removed_cot,
        removed_answer_prompt=dh.make_answer_prompt(cot_prefix + removed_cot),
    )


def _representation_metrics(hidden_states_original, hidden_states_removed, step_indices, answer_index):
    result = {
        "step_answer_cosine_by_layer": [],
        "answer_removed_cosine_by_layer": [],
        "answer_norm_by_layer": [],
        "removed_answer_norm_by_layer": [],
    }
    if not step_indices:
        return result

    for layer_idx in range(len(hidden_states_original)):
        original_layer = hidden_states_original[layer_idx][0]
        removed_layer = hidden_states_removed[layer_idx][0]
        step_repr = original_layer[step_indices].mean(dim=0)
        answer_repr = original_layer[answer_index]
        removed_answer_repr = removed_layer[-1]
        result["step_answer_cosine_by_layer"].append(_cosine(step_repr, answer_repr))
        result["answer_removed_cosine_by_layer"].append(_cosine(answer_repr, removed_answer_repr))
        result["answer_norm_by_layer"].append(_safe_detach_scalar(answer_repr.norm()))
        result["removed_answer_norm_by_layer"].append(_safe_detach_scalar(removed_answer_repr.norm()))
    return result


def _attention_metrics(attentions_original, step_indices, answer_index):
    result = {
        "answer_to_step_mass_by_layer": [],
        "answer_to_prefix_mass_by_layer": [],
        "answer_to_post_step_mass_by_layer": [],
    }
    if not step_indices or not attentions_original:
        return result

    step_start = min(step_indices)
    step_end = max(step_indices) + 1
    for layer_attention in attentions_original:
        # [batch, heads, query, key]
        answer_attention = layer_attention[0, :, answer_index, :]
        step_mass = answer_attention[:, step_start:step_end].sum(dim=-1).mean()
        prefix_mass = answer_attention[:, :step_start].sum(dim=-1).mean() if step_start > 0 else torch.tensor(0.0)
        post_mass = answer_attention[:, step_end:answer_index + 1].sum(dim=-1).mean() if step_end < (answer_index + 1) else torch.tensor(0.0)
        result["answer_to_step_mass_by_layer"].append(_safe_detach_scalar(step_mass))
        result["answer_to_prefix_mass_by_layer"].append(_safe_detach_scalar(prefix_mass))
        result["answer_to_post_step_mass_by_layer"].append(_safe_detach_scalar(post_mass))
    return result


def _projection_metrics(original_outputs, removed_outputs, step_indices, answer_index):
    projections = []
    if not step_indices:
        return projections

    for module_name, tensor in original_outputs.items():
        if tensor.ndim < 3 or module_name not in removed_outputs:
            continue
        original_layer = tensor[0]
        removed_layer = removed_outputs[module_name][0]
        step_repr = original_layer[step_indices].mean(dim=0)
        answer_repr = original_layer[answer_index]
        removed_answer_repr = removed_layer[-1]
        projections.append({
            "module": module_name,
            "step_answer_cosine": _cosine(step_repr, answer_repr),
            "answer_removed_cosine": _cosine(answer_repr, removed_answer_repr),
            "answer_norm": _safe_detach_scalar(answer_repr.norm()),
            "removed_answer_norm": _safe_detach_scalar(removed_answer_repr.norm()),
        })
    return projections


def run_mechanistic_diagnostics(model, tokenizer, dh, target, step_idx: int, args=None) -> Optional[dict[str, Any]]:
    prompts = _build_prompts(dh, target, step_idx)
    if prompts is None:
        return None

    device = _model_input_device(model)
    warnings = []
    input_ids, attention_mask, offsets, token_warnings = _tokenize_with_offsets(
        tokenizer,
        prompts.answer_prompt,
        device,
    )
    warnings.extend(token_warnings)
    removed_input_ids, removed_attention_mask, _, removed_warnings = _tokenize_with_offsets(
        tokenizer,
        prompts.removed_answer_prompt,
        device,
    )
    warnings.extend(removed_warnings)

    step_span = _find_char_span(prompts.answer_prompt, prompts.step_text)
    if step_span is None:
        warnings.append("could not locate target step text inside answer prompt")
        return {"warnings": warnings}

    step_indices = _char_span_to_token_indices(offsets, step_span)
    if not step_indices:
        warnings.append("could not map target step span to token indices")
        return {"warnings": warnings}

    answer_index = input_ids.shape[-1] - 1
    projection_limit = getattr(args, "mechanistic_diag_proj_limit", 24) if args is not None else 24
    capture_projection_modules = _iter_projection_modules(model, projection_limit)

    outputs_original, projection_original = _forward_with_optional_captures(
        model,
        input_ids,
        attention_mask,
        capture_projection_modules,
    )
    outputs_removed, projection_removed = _forward_with_optional_captures(
        model,
        removed_input_ids,
        removed_attention_mask,
        capture_projection_modules,
    )

    representation = _representation_metrics(
        outputs_original.hidden_states,
        outputs_removed.hidden_states,
        step_indices,
        answer_index,
    )
    attention = _attention_metrics(
        outputs_original.attentions,
        step_indices,
        answer_index,
    )
    projections = _projection_metrics(
        projection_original,
        projection_removed,
        step_indices,
        answer_index,
    )

    return {
        "prompt_lengths": {
            "answer_prompt_tokens": int(input_ids.shape[-1]),
            "removed_answer_prompt_tokens": int(removed_input_ids.shape[-1]),
            "step_token_count": len(step_indices),
        },
        "representation": representation,
        "attention": attention,
        "projections": projections,
        "warnings": warnings,
    }


def maybe_run_mechanistic_diagnostics(model, tokenizer, dh, target, step_idx: int, args=None):
    enabled = bool(args and getattr(args, "mechanistic_diag", False))
    if not enabled:
        return None
    try:
        return run_mechanistic_diagnostics(model, tokenizer, dh, target, step_idx, args=args)
    except Exception as exc:
        return {"warnings": [f"mechanistic diagnostics failed: {exc}"]}


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _safe_ratio(after: float, before: float, eps: float = 1e-8) -> float:
    return float((after + eps) / (before + eps))


def _safe_log_ratio(after: float, before: float, eps: float = 1e-8) -> float:
    return float(np.log((after + eps) / (before + eps)))


def _series_change(before_values, after_values, mode: str = "diff") -> list[float]:
    before_values = _to_float_list(before_values)
    after_values = _to_float_list(after_values)
    if len(before_values) != len(after_values):
        raise ValueError("before/after series length mismatch")
    if mode == "diff":
        return [after - before for before, after in zip(before_values, after_values)]
    if mode == "ratio":
        return [_safe_ratio(after, before) for before, after in zip(before_values, after_values)]
    if mode == "log_ratio":
        return [_safe_log_ratio(after, before) for before, after in zip(before_values, after_values)]
    raise ValueError(f"Unsupported change mode: {mode}")


def compare_mechanistic_diagnostics(before_diag: dict[str, Any],
                                    after_diag: dict[str, Any],
                                    mode: str = "diff") -> dict[str, Any]:
    before_repr = before_diag.get("representation", {})
    after_repr = after_diag.get("representation", {})
    before_attn = before_diag.get("attention", {})
    after_attn = after_diag.get("attention", {})
    before_proj = before_diag.get("projections", [])
    after_proj = after_diag.get("projections", [])

    comparison = {
        "mode": mode,
        "representation": {},
        "attention": {},
        "projections": [],
        "warnings": list(before_diag.get("warnings", [])) + list(after_diag.get("warnings", [])),
    }

    for key in sorted(set(before_repr) & set(after_repr)):
        if isinstance(before_repr[key], list) and isinstance(after_repr[key], list):
            comparison["representation"][key] = _series_change(before_repr[key], after_repr[key], mode=mode)

    for key in sorted(set(before_attn) & set(after_attn)):
        if isinstance(before_attn[key], list) and isinstance(after_attn[key], list):
            comparison["attention"][key] = _series_change(before_attn[key], after_attn[key], mode=mode)

    before_proj_by_name = {item["module"]: item for item in before_proj if "module" in item}
    after_proj_by_name = {item["module"]: item for item in after_proj if "module" in item}
    for module_name in sorted(set(before_proj_by_name) & set(after_proj_by_name)):
        before_item = before_proj_by_name[module_name]
        after_item = after_proj_by_name[module_name]
        module_result = {"module": module_name}
        for key in ("step_answer_cosine", "answer_removed_cosine", "answer_norm", "removed_answer_norm"):
            if key in before_item and key in after_item:
                module_result[key] = _series_change([before_item[key]], [after_item[key]], mode=mode)[0]
        comparison["projections"].append(module_result)

    return comparison


def summarize_distribution_shift(before_diag: dict[str, Any],
                                 after_diag: dict[str, Any],
                                 mode: str = "diff") -> dict[str, Any]:
    comparison = compare_mechanistic_diagnostics(before_diag, after_diag, mode=mode)

    def summarize_section(section: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        summary = {}
        for key, values in section.items():
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                continue
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "max_abs": float(np.abs(arr).max()),
                "argmax_abs": int(np.abs(arr).argmax()),
            }
        return summary

    projection_summary = []
    for item in comparison["projections"]:
        numeric_items = {k: v for k, v in item.items() if k != "module"}
        if not numeric_items:
            continue
        projection_summary.append({
            "module": item["module"],
            "mean_abs_change": float(np.mean([abs(float(v)) for v in numeric_items.values()])),
            "max_abs_change": float(np.max([abs(float(v)) for v in numeric_items.values()])),
        })

    projection_summary.sort(key=lambda row: row["max_abs_change"], reverse=True)

    return {
        "mode": mode,
        "representation": summarize_section(comparison["representation"]),
        "attention": summarize_section(comparison["attention"]),
        "top_projection_changes": projection_summary[:20],
        "warnings": comparison.get("warnings", []),
    }


def _lazy_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_mechanistic_series(before_diag: dict[str, Any],
                            after_diag: dict[str, Any],
                            section: str,
                            key: str,
                            output_path: Optional[str] = None,
                            title: Optional[str] = None):
    plt = _lazy_import_matplotlib()
    before_values = before_diag.get(section, {}).get(key)
    after_values = after_diag.get(section, {}).get(key)
    if before_values is None or after_values is None:
        raise KeyError(f"Missing series {section}.{key}")

    layers = list(range(len(before_values)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, before_values, label="before", marker="o")
    ax.plot(layers, after_values, label="after", marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(key)
    ax.set_title(title or f"{section}.{key}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig


def plot_mechanistic_shift(before_diag: dict[str, Any],
                           after_diag: dict[str, Any],
                           section: str,
                           key: str,
                           mode: str = "diff",
                           output_path: Optional[str] = None,
                           title: Optional[str] = None):
    plt = _lazy_import_matplotlib()
    comparison = compare_mechanistic_diagnostics(before_diag, after_diag, mode=mode)
    values = comparison.get(section, {}).get(key)
    if values is None:
        raise KeyError(f"Missing comparison series {section}.{key}")

    layers = list(range(len(values)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(layers, values)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"{mode}({key})")
    ax.set_title(title or f"{mode}: {section}.{key}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig


def plot_projection_shift_heatmap(before_diag: dict[str, Any],
                                  after_diag: dict[str, Any],
                                  mode: str = "diff",
                                  output_path: Optional[str] = None,
                                  title: Optional[str] = None):
    plt = _lazy_import_matplotlib()
    comparison = compare_mechanistic_diagnostics(before_diag, after_diag, mode=mode)
    projection_rows = comparison.get("projections", [])
    if not projection_rows:
        raise ValueError("No projection diagnostics available")

    metric_keys = [key for key in projection_rows[0].keys() if key != "module"]
    matrix = np.asarray([[float(row[key]) for key in metric_keys] for row in projection_rows], dtype=float)
    labels = [row["module"] for row in projection_rows]

    fig_height = max(4, min(0.35 * len(labels), 18))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_keys, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title or f"Projection shift heatmap ({mode})")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig
