from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mechanistic_objectives import IGNORE_IDX, build_target_mask


@dataclass(frozen=True)
class CausalCoTLossBreakdown:
    loss: torch.Tensor
    ie_loss: torch.Tensor
    margin_loss: torch.Tensor
    full_logprob: torch.Tensor
    counterfactual_logprob: torch.Tensor


def sequence_logprob_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average target-token log-probability for causal-LM labeled completions."""
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    target_mask = build_target_mask(shifted_labels)

    safe_labels = shifted_labels.masked_fill(~target_mask, 0)
    token_logprobs = F.log_softmax(shifted_logits.float(), dim=-1)
    selected = token_logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    selected = selected * target_mask.to(selected.dtype)
    token_counts = target_mask.sum(dim=-1).clamp_min(1)
    return selected.sum(dim=-1) / token_counts


def causal_cot_frodo_loss(
    full_logits: torch.Tensor,
    full_labels: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    counterfactual_labels: torch.Tensor,
    *,
    margin: float = 1.0,
    ie_lambda: float = 1.0,
    margin_lambda: float = 1.0,
) -> CausalCoTLossBreakdown:
    """
    FRODO-style reasoning-module objective adapted to causal LM.

    full prompt: question + original CoT -> answer
    counterfactual prompt: question + ablated CoT -> same answer

    L_IE = -log p(answer | question, original CoT)
    L_MR = max(0, margin - (logp_full - logp_counterfactual))
    """
    full_logprob = sequence_logprob_from_logits(full_logits, full_labels)
    counterfactual_logprob = sequence_logprob_from_logits(
        counterfactual_logits,
        counterfactual_labels,
    )
    ie_loss = -full_logprob.mean()
    margin_loss = torch.clamp(
        margin - (full_logprob - counterfactual_logprob),
        min=0.0,
    ).mean()
    loss = ie_lambda * ie_loss + margin_lambda * margin_loss
    return CausalCoTLossBreakdown(
        loss=loss,
        ie_loss=ie_loss,
        margin_loss=margin_loss,
        full_logprob=full_logprob.detach(),
        counterfactual_logprob=counterfactual_logprob.detach(),
    )
