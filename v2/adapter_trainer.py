from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lora_adapter import LoRAAdapterManager


@dataclass
class AdapterTrainingJob:
    adapter_id: str
    dataset: torch.utils.data.Dataset
    collator: Callable
    epochs: int
    lr: float
    loss_type: str = "npo_grad_diff"
    batch_size: int = 1
    input_pad_value: Optional[int] = None
    label_pad_value: int = -100
    attention_pad_value: int = 0
    metadata: dict = field(default_factory=dict)


class AdapterTrainer:
    def __init__(self, model, oracle_model, adapter_manager: LoRAAdapterManager,
                 optimizer_cls=torch.optim.AdamW,
                 scheduler_builder: Optional[Callable] = None):
        self.model = model
        self.oracle_model = oracle_model
        self.adapter_manager = adapter_manager
        self.optimizer_cls = optimizer_cls
        self.scheduler_builder = scheduler_builder

    def train_job(self, job: AdapterTrainingJob, compute_loss_fn: Callable,
                  epoch_end_callback: Optional[Callable[[AdapterTrainingJob, int], dict]] = None):
        self.adapter_manager.create_adapter(job.adapter_id)
        dataloader = DataLoader(
            job.dataset,
            batch_size=job.batch_size,
            collate_fn=job.collator,
            shuffle=True,
        )
        optimizer = self.optimizer_cls(
            self.adapter_manager.adapter_parameters(job.adapter_id),
            lr=job.lr,
        )
        steps_per_epoch = len(dataloader)
        scheduler = None
        if self.scheduler_builder is not None:
            scheduler = self.scheduler_builder(optimizer, job.epochs * steps_per_epoch)
        history = []

        with self.adapter_manager.activate(job.adapter_id):
            for epoch in range(job.epochs):
                self.model.train()
                optimizer.zero_grad()
                epoch_loss = 0.0
                n_steps = 0
                for batch in dataloader:
                    loss = compute_loss_fn(
                        self.model,
                        self.oracle_model,
                        batch,
                        loss_type=job.loss_type,
                    )
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    epoch_loss += float(loss.detach().cpu())
                    n_steps += 1
                epoch_info = {
                    "epoch": epoch + 1,
                    "mean_loss": (epoch_loss / max(1, n_steps)),
                }
                if epoch_end_callback is not None:
                    epoch_info["evaluation"] = epoch_end_callback(job, epoch + 1)
                history.append(epoch_info)
        return {
            "adapter_id": job.adapter_id,
            "history": history,
            "metadata": job.metadata,
        }

    @staticmethod
    def _pad_and_concat(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(tensor.shape[-1] for tensor in tensors)
        padded = []
        for tensor in tensors:
            if tensor.shape[-1] == max_len:
                padded.append(tensor)
                continue
            pad_amount = max_len - tensor.shape[-1]
            padded.append(F.pad(tensor, (pad_amount, 0), value=pad_value))
        return torch.cat(padded, dim=0)

    def _merge_job_batches(self, jobs: list[AdapterTrainingJob], batches: list):
        forget_inputs = [batch[0] for batch in batches]
        retain_inputs = [batch[1] for batch in batches]

        input_pad_values = []
        for job in jobs:
            if job.input_pad_value is not None:
                input_pad_values.append(job.input_pad_value)
            elif hasattr(job.collator, "pad_token_id"):
                input_pad_values.append(job.collator.pad_token_id)
            else:
                raise ValueError(f"Job {job.adapter_id} is missing input_pad_value")

        if len(set(input_pad_values)) != 1:
            raise ValueError("All jobs in a merged batch must share the same input pad value")
        input_pad_value = input_pad_values[0]

        merged_forget = (
            self._pad_and_concat([forget[0] for forget in forget_inputs], input_pad_value),
            self._pad_and_concat([forget[1] for forget in forget_inputs], jobs[0].label_pad_value),
            self._pad_and_concat([forget[2] for forget in forget_inputs], jobs[0].attention_pad_value),
        )
        merged_retain = (
            self._pad_and_concat([retain[0] for retain in retain_inputs], input_pad_value),
            self._pad_and_concat([retain[1] for retain in retain_inputs], jobs[0].label_pad_value),
            self._pad_and_concat([retain[2] for retain in retain_inputs], jobs[0].attention_pad_value),
        )
        return merged_forget, merged_retain

    def train_job_group_batched(self, jobs: list[AdapterTrainingJob], compute_loss_fn: Callable,
                                epoch_end_callback: Optional[Callable[[AdapterTrainingJob, int], dict]] = None):
        if not jobs:
            return []

        reference = jobs[0]
        for job in jobs[1:]:
            if not (
                job.epochs == reference.epochs
                and job.lr == reference.lr
                and job.loss_type == reference.loss_type
                and job.batch_size == reference.batch_size
            ):
                raise ValueError("Batched adapter training requires matching epochs/lr/loss_type/batch_size")

        dataloaders = {}
        iterators = {}
        all_params = []
        for job in jobs:
            self.adapter_manager.create_adapter(job.adapter_id)
            dataloaders[job.adapter_id] = DataLoader(
                job.dataset,
                batch_size=job.batch_size,
                collate_fn=job.collator,
                shuffle=True,
            )
            all_params.extend(self.adapter_manager.adapter_parameters(job.adapter_id))

        optimizer = self.optimizer_cls(all_params, lr=reference.lr)
        max_steps_per_epoch = max(len(dataloader) for dataloader in dataloaders.values())
        scheduler = None
        if self.scheduler_builder is not None:
            scheduler = self.scheduler_builder(optimizer, reference.epochs * max_steps_per_epoch)
        history = {job.adapter_id: [] for job in jobs}

        for epoch in range(reference.epochs):
            iterators = {job.adapter_id: iter(dataloaders[job.adapter_id]) for job in jobs}
            self.model.train()
            optimizer.zero_grad()
            mean_losses = {job.adapter_id: [] for job in jobs}

            while True:
                current_jobs = []
                current_batches = []
                adapter_routing = []
                for job in jobs:
                    try:
                        batch = next(iterators[job.adapter_id])
                    except StopIteration:
                        continue
                    current_jobs.append(job)
                    current_batches.append(batch)
                    forget_batch_size = batch[0][0].shape[0]
                    adapter_routing.extend([job.adapter_id] * forget_batch_size)

                if not current_batches:
                    break

                merged_batch = self._merge_job_batches(current_jobs, current_batches)
                with self.adapter_manager.activate(adapter_routing):
                    loss = compute_loss_fn(
                        self.model,
                        self.oracle_model,
                        merged_batch,
                        loss_type=reference.loss_type,
                    )
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                step_loss = float(loss.detach().cpu())
                for job in current_jobs:
                    mean_losses[job.adapter_id].append(step_loss)

            for job in jobs:
                values = mean_losses[job.adapter_id]
                epoch_info = {
                    "epoch": epoch + 1,
                    "mean_loss": sum(values) / max(1, len(values)),
                }
                if epoch_end_callback is not None:
                    with self.adapter_manager.activate(job.adapter_id):
                        epoch_info["evaluation"] = epoch_end_callback(job, epoch + 1)
                history[job.adapter_id].append(epoch_info)

        return [
            {
                "adapter_id": job.adapter_id,
                "history": history[job.adapter_id],
                "metadata": job.metadata,
            }
            for job in jobs
        ]

    def train_jobs(self, jobs: list[AdapterTrainingJob], compute_loss_fn: Callable, mode: str = "sequential",
                   epoch_end_callback: Optional[Callable[[AdapterTrainingJob, int], dict]] = None):
        if mode == "batched":
            return self.train_job_group_batched(jobs, compute_loss_fn, epoch_end_callback=epoch_end_callback)
        results = []
        for job in jobs:
            results.append(self.train_job(job, compute_loss_fn, epoch_end_callback=epoch_end_callback))
        return results

    def save_trained_adapter(self, adapter_id: str, output_path: str | Path, metadata: Optional[dict] = None):
        self.adapter_manager.save_adapter(adapter_id, output_path, metadata=metadata)
