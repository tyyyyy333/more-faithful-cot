from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


IGNORE_IDX = -100


def build_target_mask(labels: torch.Tensor, max_target_tokens: int = 0) -> torch.Tensor:
    mask = labels.ne(IGNORE_IDX)
    if max_target_tokens <= 0:
        return mask

    limited_mask = torch.zeros_like(mask, dtype=torch.bool)
    for batch_idx in range(mask.shape[0]):
        active = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        keep = active[:max_target_tokens]
        limited_mask[batch_idx, keep] = True
    return limited_mask


def build_shifted_target_mask(labels: torch.Tensor, max_target_tokens: int = 0) -> torch.Tensor:
    shifted_labels = labels[..., 1:].contiguous()
    return build_target_mask(shifted_labels, max_target_tokens=max_target_tokens)


def get_batch_loss_masked(output: torch.Tensor, labels: torch.Tensor, max_target_tokens: int = 0) -> torch.Tensor:
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    token_mask = build_target_mask(shifted_labels, max_target_tokens=max_target_tokens)

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_IDX, reduction='none')
    token_loss = loss_function(output.transpose(-1, -2), shifted_labels)
    token_loss = token_loss * token_mask.to(token_loss.dtype)
    return token_loss.sum(dim=-1)


def _selected_layer_indices(hidden_states: Iterable[torch.Tensor], last_n_layers: int) -> list[int]:
    total = len(tuple(hidden_states))
    if total <= 1:
        return []
    last_n_layers = max(1, min(last_n_layers, total - 1))
    return list(range(total - last_n_layers, total))


def _layer_weights(num_layers: int, gamma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if num_layers <= 0:
        return torch.zeros(0, device=device, dtype=dtype)
    exponents = torch.arange(num_layers - 1, -1, -1, device=device, dtype=dtype)
    weights = torch.pow(torch.full_like(exponents, gamma), exponents)
    return weights / weights.sum().clamp_min(1e-8)


def representation_similarity_loss(current_hidden_states,
                                   reference_hidden_states,
                                   labels: torch.Tensor,
                                   last_n_layers: int = 4,
                                   gamma: float = 0.9,
                                   max_target_tokens: int = 0,
                                   absolute_cosine: bool = True) -> torch.Tensor:
    layer_indices = _selected_layer_indices(current_hidden_states, last_n_layers)
    if not layer_indices:
        return torch.tensor(0.0, device=labels.device)

    target_mask = build_target_mask(labels, max_target_tokens=max_target_tokens)
    if not target_mask.any():
        return torch.tensor(0.0, device=labels.device)

    sample_losses = []
    for batch_idx in range(labels.shape[0]):
        token_indices = torch.nonzero(target_mask[batch_idx], as_tuple=False).flatten()
        if token_indices.numel() == 0:
            continue

        layer_losses = []
        for layer_idx in layer_indices:
            current_layer = current_hidden_states[layer_idx][batch_idx]
            reference_layer = reference_hidden_states[layer_idx][batch_idx]
            current_repr = current_layer.index_select(0, token_indices).mean(dim=0)
            reference_repr = reference_layer.index_select(0, token_indices).mean(dim=0)
            cosine = F.cosine_similarity(
                current_repr.float().unsqueeze(0),
                reference_repr.float().unsqueeze(0),
                dim=-1,
            ).squeeze(0)
            layer_losses.append(cosine.abs() if absolute_cosine else cosine)

        if not layer_losses:
            continue
        stacked = torch.stack(layer_losses)
        weights = _layer_weights(
            len(layer_losses),
            gamma=gamma,
            device=stacked.device,
            dtype=stacked.dtype,
        )
        sample_losses.append((stacked * weights).sum())

    if not sample_losses:
        return torch.tensor(0.0, device=labels.device)
    return torch.stack(sample_losses).mean()


def scale_auxiliary_loss(aux_loss: torch.Tensor,
                         reference_loss: torch.Tensor,
                         coeff: float = 1.0,
                         auto_scale: bool = False,
                         eps: float = 1e-8) -> torch.Tensor:
    scaled = aux_loss
    if auto_scale:
        ref_value = reference_loss.detach().abs().clamp_min(eps)
        aux_value = aux_loss.detach().abs().clamp_min(eps)
        scaled = aux_loss * (ref_value / aux_value)
    return coeff * scaled
