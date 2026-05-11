from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import torch
from torch import nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = field(default_factory=tuple)


def _sanitize_adapter_name(name: str) -> str:
    return name.replace(".", "__")


class MultiAdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, layer_name: str, config: LoRAConfig):
        super().__init__()
        self.base_layer = base_layer
        self.layer_name = layer_name
        self.config = config
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scaling: Dict[str, float] = {}
        self.active_adapter: Optional[str | Sequence[Optional[str]]] = None

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def has_adapter(self, adapter_name: str) -> bool:
        key = _sanitize_adapter_name(adapter_name)
        return key in self.lora_A and key in self.lora_B

    def create_adapter(self, adapter_name: str):
        key = _sanitize_adapter_name(adapter_name)
        if self.has_adapter(adapter_name):
            return

        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        rank = self.config.rank
        a = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        b = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(a, a=5 ** 0.5)
        self.lora_A[key] = a
        self.lora_B[key] = b
        self.scaling[key] = self.config.alpha / max(1, rank)

    def delete_adapter(self, adapter_name: str):
        key = _sanitize_adapter_name(adapter_name)
        if key in self.lora_A:
            del self.lora_A[key]
        if key in self.lora_B:
            del self.lora_B[key]
        self.scaling.pop(key, None)

    def set_active_adapter(self, adapter_name: Optional[str | Sequence[Optional[str]]]):
        self.active_adapter = adapter_name

    def adapter_parameters(self, adapter_name: str) -> list[nn.Parameter]:
        key = _sanitize_adapter_name(adapter_name)
        return [self.lora_A[key], self.lora_B[key]]

    def adapter_state_dict(self, adapter_name: str) -> dict[str, torch.Tensor]:
        key = _sanitize_adapter_name(adapter_name)
        return {
            "lora_A": self.lora_A[key].detach().cpu(),
            "lora_B": self.lora_B[key].detach().cpu(),
        }

    def load_adapter_state(self, adapter_name: str, state: dict[str, torch.Tensor]):
        if not self.has_adapter(adapter_name):
            self.create_adapter(adapter_name)
        key = _sanitize_adapter_name(adapter_name)
        self.lora_A[key].data.copy_(state["lora_A"].to(device=self.lora_A[key].device, dtype=self.lora_A[key].dtype))
        self.lora_B[key].data.copy_(state["lora_B"].to(device=self.lora_B[key].device, dtype=self.lora_B[key].dtype))

    def _compute_delta(self, x: torch.Tensor, adapter_name: str) -> torch.Tensor:
        key = _sanitize_adapter_name(adapter_name)
        a = self.lora_A[key]
        b = self.lora_B[key]
        lora_hidden = self.dropout(x) @ a.transpose(0, 1)
        return (lora_hidden @ b.transpose(0, 1)) * self.scaling[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(x)
        if self.active_adapter is None:
            return output

        if isinstance(self.active_adapter, str):
            if not self.has_adapter(self.active_adapter):
                return output
            return output + self._compute_delta(x, self.active_adapter)

        adapter_routing = list(self.active_adapter)
        if x.shape[0] != len(adapter_routing):
            raise ValueError(
                f"Adapter routing length {len(adapter_routing)} does not match batch size {x.shape[0]}"
            )

        routed_output = output.clone()
        unique_adapters = {name for name in adapter_routing if name is not None}
        for adapter_name in unique_adapters:
            if adapter_name is None or not self.has_adapter(adapter_name):
                continue
            indices = [idx for idx, current in enumerate(adapter_routing) if current == adapter_name]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, device=x.device, dtype=torch.long)
            routed_output[index_tensor] = routed_output[index_tensor] + self._compute_delta(
                x.index_select(0, index_tensor),
                adapter_name,
            )
        return routed_output


class LoRAAdapterManager:
    def __init__(self, model: nn.Module, config: LoRAConfig):
        self.model = model
        self.config = config
        self.active_adapter: Optional[str | Sequence[Optional[str]]] = None
        self._wrapped_layers: Dict[str, MultiAdapterLinear] = {}
        self._adapter_names: set[str] = set()
        self._inject_lora_layers()
        self.freeze_non_lora_parameters()

    def _matches_target(self, module_name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if not self.config.target_modules:
            return True
        return any(module_name.endswith(target_name) for target_name in self.config.target_modules)

    def _iter_target_layers(self) -> Iterator[tuple[str, nn.Linear]]:
        for module_name, module in self.model.named_modules():
            if self._matches_target(module_name, module):
                yield module_name, module

    def _get_parent_module(self, module_name: str) -> tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _inject_lora_layers(self):
        for module_name, module in list(self._iter_target_layers()):
            parent, child_name = self._get_parent_module(module_name)
            wrapped = MultiAdapterLinear(module, module_name, self.config)
            setattr(parent, child_name, wrapped)
            self._wrapped_layers[module_name] = wrapped

    def freeze_non_lora_parameters(self):
        for module in self.model.modules():
            if isinstance(module, MultiAdapterLinear):
                for parameter in module.base_layer.parameters():
                    parameter.requires_grad = False
            else:
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad = False

    def create_adapter(self, adapter_name: str):
        self._adapter_names.add(adapter_name)
        for wrapped in self._wrapped_layers.values():
            wrapped.create_adapter(adapter_name)

    def delete_adapter(self, adapter_name: str):
        self._adapter_names.discard(adapter_name)
        for wrapped in self._wrapped_layers.values():
            wrapped.delete_adapter(adapter_name)

    def list_adapters(self) -> list[str]:
        return sorted(self._adapter_names)

    def has_adapter(self, adapter_name: str) -> bool:
        return adapter_name in self._adapter_names

    def set_active_adapter(self, adapter_name: Optional[str | Sequence[Optional[str]]]):
        self.active_adapter = adapter_name
        for wrapped in self._wrapped_layers.values():
            wrapped.set_active_adapter(adapter_name)

    @contextmanager
    def activate(self, adapter_name: Optional[str | Sequence[Optional[str]]]):
        previous = self.active_adapter
        self.set_active_adapter(adapter_name)
        try:
            yield
        finally:
            self.set_active_adapter(previous)

    def adapter_parameters(self, adapter_name: str) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for wrapped in self._wrapped_layers.values():
            params.extend(wrapped.adapter_parameters(adapter_name))
        return params

    def adapter_state_dict(self, adapter_name: str) -> dict[str, dict[str, torch.Tensor]]:
        return {
            layer_name: wrapped.adapter_state_dict(adapter_name)
            for layer_name, wrapped in self._wrapped_layers.items()
        }

    def build_adapter_manifest(self, adapter_name: str, metadata: Optional[dict] = None) -> dict:
        model_name = getattr(self.model, "name_or_path", None)
        if model_name is None and hasattr(self.model, "config"):
            model_name = getattr(self.model.config, "_name_or_path", None)
        return {
            "adapter_name": adapter_name,
            "model_name_or_path": model_name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "target_modules": list(self.config.target_modules),
            },
            "layers": list(self._wrapped_layers.keys()),
            "metadata": metadata or {},
        }

    def save_adapter(self, adapter_name: str, output_path: str | Path, metadata: Optional[dict] = None):
        payload = {
            "manifest": self.build_adapter_manifest(adapter_name, metadata=metadata),
            "layers": self.adapter_state_dict(adapter_name),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)

    def load_adapter(self, input_path: str | Path, adapter_name: Optional[str] = None) -> str:
        payload = torch.load(Path(input_path), map_location="cpu")
        manifest = payload["manifest"]
        resolved_name = adapter_name or manifest["adapter_name"]
        saved_config = manifest["config"]
        if (
            saved_config["rank"] != self.config.rank
            or saved_config["alpha"] != self.config.alpha
            or saved_config["dropout"] != self.config.dropout
            or list(saved_config["target_modules"]) != list(self.config.target_modules)
        ):
            raise ValueError(
                "Loaded adapter config does not match the current LoRA manager config"
            )
        self.create_adapter(resolved_name)
        for layer_name, layer_state in payload["layers"].items():
            self._wrapped_layers[layer_name].load_adapter_state(resolved_name, layer_state)
        return resolved_name


def attach_lora_adapters(model: nn.Module, config: LoRAConfig) -> LoRAAdapterManager:
    manager = LoRAAdapterManager(model, config)
    model._lora_adapter_manager = manager
    return manager
