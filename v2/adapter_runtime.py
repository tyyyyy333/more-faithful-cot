from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

import evaluate as eval_mod
from lora_adapter import LoRAConfig, attach_lora_adapters
from models import load_model_and_tokenizer


@dataclass
class AdapterRecord:
    base_model_id: str
    adapter_id: str
    adapter_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model_id": self.base_model_id,
            "adapter_id": self.adapter_id,
            "adapter_path": self.adapter_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterRecord":
        return cls(
            base_model_id=data["base_model_id"],
            adapter_id=data["adapter_id"],
            adapter_path=data["adapter_path"],
            metadata=data.get("metadata", {}),
        )


def _config_from_payload(payload: dict[str, Any]) -> LoRAConfig:
    manifest = payload.get("manifest", {})
    config = manifest.get("config", payload.get("config", {}))
    return LoRAConfig(
        rank=int(config.get("rank", 8)),
        alpha=float(config.get("alpha", 16.0)),
        dropout=float(config.get("dropout", 0.0)),
        target_modules=tuple(config.get("target_modules", ())),
    )


def _load_adapter_payload(adapter_path: str | Path) -> dict[str, Any]:
    return torch.load(Path(adapter_path), map_location="cpu")


def _record_from_payload(base_model_id: str | None, adapter_path: str | Path, payload: dict[str, Any]) -> AdapterRecord:
    manifest = payload.get("manifest", {})
    metadata = dict(manifest.get("metadata", payload.get("metadata", {})))
    resolved_base_model_id = base_model_id or metadata.get("base_model_id")
    if resolved_base_model_id is None:
        raise ValueError("base_model_id is required to restore an adapter record")
    return AdapterRecord(
        base_model_id=resolved_base_model_id,
        adapter_id=manifest.get("adapter_name", payload.get("adapter_name")),
        adapter_path=str(adapter_path),
        metadata=metadata,
    )


def _resolve_record_adapter_path(record: AdapterRecord, record_source: Path | None = None) -> AdapterRecord:
    adapter_path = Path(record.adapter_path)
    if adapter_path.is_absolute() or adapter_path.exists() or record_source is None:
        return record
    resolved = record_source.parent / adapter_path
    return AdapterRecord(
        base_model_id=record.base_model_id,
        adapter_id=record.adapter_id,
        adapter_path=str(resolved),
        metadata=record.metadata,
    )


def load_adapter_record(record_or_path: str | Path | dict[str, Any] | AdapterRecord,
                        base_model_id: str | None = None) -> AdapterRecord:
    if isinstance(record_or_path, AdapterRecord):
        return record_or_path
    if isinstance(record_or_path, dict):
        return AdapterRecord.from_dict(record_or_path)

    path = Path(record_or_path)
    if path.suffix == ".json":
        with path.open("r") as infile:
            record = AdapterRecord.from_dict(json.load(infile))
        return _resolve_record_adapter_path(record, record_source=path)

    payload = _load_adapter_payload(path)
    return _record_from_payload(base_model_id, path, payload)


class AdapterRuntime:
    def __init__(self, model_id: str, model, tokenizer, adapter_manager, device_pref: str = "auto"):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_manager = adapter_manager
        self.device_pref = device_pref
        self.records: dict[str, AdapterRecord] = {}

    @classmethod
    def from_base_model(cls, model_id: str, device_pref: str = "auto",
                        lora_config: Optional[LoRAConfig] = None) -> "AdapterRuntime":
        model, tokenizer = load_model_and_tokenizer(model_id, device_pref=device_pref)
        manager = attach_lora_adapters(model, lora_config or LoRAConfig())
        return cls(model_id=model_id, model=model, tokenizer=tokenizer, adapter_manager=manager, device_pref=device_pref)

    @classmethod
    def from_adapter_record(cls, record_or_path: str | Path | dict[str, Any] | AdapterRecord,
                            device_pref: str = "auto") -> "AdapterRuntime":
        record = load_adapter_record(record_or_path)
        payload = _load_adapter_payload(record.adapter_path)
        model, tokenizer = load_model_and_tokenizer(record.base_model_id, device_pref=device_pref)
        manager = attach_lora_adapters(model, _config_from_payload(payload))
        manager.load_adapter(record.adapter_path, adapter_name=record.adapter_id)
        runtime = cls(
            model_id=record.base_model_id,
            model=model,
            tokenizer=tokenizer,
            adapter_manager=manager,
            device_pref=device_pref,
        )
        runtime.records[record.adapter_id] = record
        return runtime

    def register_record(self, record_or_path: str | Path | dict[str, Any] | AdapterRecord):
        record = load_adapter_record(record_or_path, base_model_id=self.model_id)
        self.records[record.adapter_id] = record
        return record

    def load_adapter(self, adapter_record_or_path: str | Path | dict[str, Any] | AdapterRecord) -> str:
        record = load_adapter_record(adapter_record_or_path, base_model_id=self.model_id)
        payload = _load_adapter_payload(record.adapter_path)
        payload_config = _config_from_payload(payload)
        if payload_config != self.adapter_manager.config:
            raise ValueError(
                "Adapter config mismatch between runtime manager and adapter payload. "
                "Create the runtime from the first adapter payload or ensure configs match."
            )
        adapter_id = self.adapter_manager.load_adapter(record.adapter_path, adapter_name=record.adapter_id)
        self.records[adapter_id] = record
        return adapter_id

    def save_record(self, adapter_id: str, output_path: str | Path):
        if adapter_id not in self.records:
            raise KeyError(f"Unknown adapter_id: {adapter_id}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as outfile:
            json.dump(self.records[adapter_id].to_dict(), outfile, ensure_ascii=False, indent=2)

    @contextmanager
    def activate(self, adapter_id: str):
        with self.adapter_manager.activate(adapter_id):
            yield

    def evaluate(self, adapter_id: str, DH, target, specificity_split, step_idx, args=None):
        with self.activate(adapter_id):
            return eval_mod.evaluate(
                self.model,
                self.tokenizer,
                DH,
                target,
                specificity_split,
                step_idx=step_idx,
                args=args,
            )

    def answer_probabilities(self, adapter_id: str, dh, instance):
        with self.activate(adapter_id):
            return eval_mod.answer_probabilities(self.model, self.tokenizer, dh, instance)

    def completion(self, adapter_id: str, prompt: str, **kwargs):
        with self.activate(adapter_id):
            return eval_mod.complete(self.model, self.tokenizer, prompt, **kwargs)

    def generate(self, adapter_id: str, instance):
        with self.activate(adapter_id):
            return eval_mod.generate(self.model, self.tokenizer, instance)

    def generation_fixed_cot(self, adapter_id: str, dh, instance, cot_text):
        with self.activate(adapter_id):
            return eval_mod.generation_fixed_cot(self.model, self.tokenizer, dh, instance, cot_text)

    def letter_completion(self, adapter_id: str, prompt: str, n_choices: int):
        with self.activate(adapter_id):
            return eval_mod.letter_completion(self.model, self.tokenizer, prompt, n_choices)

    def generate_cot(self, adapter_id: str, instance, **kwargs):
        with self.activate(adapter_id):
            return eval_mod.generate_cot(self.model, self.tokenizer, instance, **kwargs)

    def cot_generate(self, adapter_id: str, instance, **kwargs):
        with self.activate(adapter_id):
            return eval_mod.cot_generate(self.model, self.tokenizer, instance, **kwargs)

    def completion_probabilities(self, adapter_id: str, prefix: str, targets):
        with self.activate(adapter_id):
            return eval_mod.completion_probabilities(self.model, self.tokenizer, prefix, targets)


def evaluate_record(record_or_path: str | Path | dict[str, Any] | AdapterRecord,
                    DH, target, specificity_split, step_idx, args=None,
                    device_pref: str = "auto"):
    runtime = AdapterRuntime.from_adapter_record(record_or_path, device_pref=device_pref)
    record = load_adapter_record(record_or_path)
    return runtime.evaluate(record.adapter_id, DH, target, specificity_split, step_idx, args=args)


def load_runtime_for_records(records: Iterable[str | Path | dict[str, Any] | AdapterRecord],
                             device_pref: str = "auto") -> AdapterRuntime:
    records = list(records)
    if not records:
        raise ValueError("At least one adapter record is required")
    first = load_adapter_record(records[0])
    runtime = AdapterRuntime.from_adapter_record(first, device_pref=device_pref)
    for record in records[1:]:
        runtime.load_adapter(record)
    return runtime


def load_runtime_from_manifest(manifest_path: str | Path, device_pref: str = "auto") -> AdapterRuntime:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r") as infile:
        manifest = json.load(infile)
    records = []
    for entry in manifest.get("adapters", []):
        record_path = Path(entry["record_path"])
        if not record_path.is_absolute():
            candidate = manifest_path.parent / record_path
            if candidate.exists():
                record_path = candidate
        records.append(record_path)
    return load_runtime_for_records(records, device_pref=device_pref)
