import os, sys, gc, json, copy, random, argparse, subprocess, shutil
from contextlib import nullcontext
from pprint import pprint
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from transformers import AutoTokenizer as TOK
from transformers import AutoModelForCausalLM as CLM

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from evaluate import completion_probabilities, answer_probabilities, complete, generation_fixed_cot
from data import FRCollator, cot_to_otfd, model_name_dict, load_or_generate_dataset_cots
from dataload import DATASETS
from adapter_trainer import AdapterTrainer, AdapterTrainingJob
from adapter_runtime import AdapterRecord
from lora_adapter import LoRAConfig, attach_lora_adapters
from util import set_random_seed
from models import resolve_device

_ACTIVE_ORACLE_CACHE = None
_SHARED_ORACLE_MODELS = {}

def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_reserved()/1024**2)


def run_lm_eval(model_path, log_path): 
  run_cmd = ["lm_eval","--model","hf",
    "--model_args", "pretrained={}",
    "--tasks", "mmlu",
    "--device","cuda:0",
    "--batch_size", "auto:4",
    "--num_fewshot=0",
    ]

  run_cmd[4] = run_cmd[4].format(model_path)

  result = subprocess.run(
      run_cmd,
      text=True,  # Return output as a string (not bytes)
      capture_output=True,  # Capture stdout and stderr
      check=True  # Raise CalledProcessError if the command fails
  )

  return result.stdout

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss


def _oracle_forward_context(model, oracle_model):
    if oracle_model is not model:
        return nullcontext()
    adapter_manager = getattr(model, "_lora_adapter_manager", None)
    if adapter_manager is None:
        return nullcontext()
    return adapter_manager.activate(None)

def compute_loss(model, oracle_model, inputs, loss_type='npo_grad_diff', ref_policy='fine_tuned', beta=0.1, npo_coeff=1.0, grad_diff_coeff=1.0, KL_coeff=1.0, return_outputs=False):
        oracle_cache = _ACTIVE_ORACLE_CACHE
        forget_inputs, retain_inputs = inputs
        input_ids, labels, attention_mask = forget_inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

        if ref_policy != 'fine_tuned':
            raise NotImplementedError

        forget_loss_current = get_batch_loss(outputs.logits, labels)
        if oracle_cache is not None and 'forget_loss' in oracle_cache:
            forget_loss_oracle = oracle_cache['forget_loss']
        else:
            with torch.no_grad(), _oracle_forward_context(model, oracle_model):
                forget_outputs_oracle = oracle_model(
                    input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            forget_loss_oracle = get_batch_loss(forget_outputs_oracle.logits, labels)

        neg_log_ratios = forget_loss_current - forget_loss_oracle
        forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

        if loss_type == 'npo':
            loss = forget_loss
        elif loss_type == 'npo_grad_diff':
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )
            retain_loss = retain_outputs.loss
            loss = npo_coeff * forget_loss + grad_diff_coeff * retain_loss
        elif loss_type == 'npo_KL':
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            if oracle_cache is not None and 'retain_log_probs' in oracle_cache:
                retain_probs = oracle_cache['retain_log_probs']
            else:
                with torch.no_grad(), _oracle_forward_context(model, oracle_model):
                    retain_outputs = oracle_model(
                        retain_input_ids,
                        labels=retain_labels,
                        attention_mask=retain_attention_mask,
                    )
                retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
                retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            retain_loss = nn.functional.kl_div(
                current_probs,
                retain_probs,
                reduction='batchmean',
                log_target=True,
            )
            loss = npo_coeff * forget_loss + KL_coeff * retain_loss
        else:
            raise NotImplementedError

        return (loss, outputs) if return_outputs else loss


def compute_specificity(model, tokenizer, DH, specificity_split):
  specificity = []
  specificity_probs = []
  if not specificity_split:
      return specificity, specificity_probs

  batch_size = 8
  device = model.device
  n_options = len(DH.get_answer_letters(specificity_split[0]['raw_instance']))
  answer_letters = ["A", "B", "C", "D", "E"][:n_options]
  answer_indices = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in answer_letters]

  for start in range(0, len(specificity_split), batch_size):
      batch_instances = specificity_split[start:start + batch_size]
      prompts = [DH.make_bowman_demonstration(instance['raw_instance']) for instance in batch_instances]
      old_padding_side = tokenizer.padding_side
      tokenizer.padding_side = "left"
      answer_inputs = tokenizer(
          prompts,
          padding=True,
          add_special_tokens=False,
          return_tensors='pt',
      )
      tokenizer.padding_side = old_padding_side

      input_ids = answer_inputs.input_ids.to(device)
      attention_mask = answer_inputs.attention_mask.to(device)
      with torch.no_grad():
          answer_output = model.generate(
              input_ids=input_ids,
              attention_mask=attention_mask,
              max_new_tokens=10,
              output_scores=True,
              temperature=0.0,
              do_sample=False,
              return_dict_in_generate=True,
              pad_token_id=tokenizer.pad_token_id,
          )

      first_token_probs = torch.softmax(answer_output['scores'][0], dim=-1)
      batch_letter_probs = first_token_probs[:, answer_indices].detach().cpu().float().numpy()
      batch_predictions = np.argmax(batch_letter_probs, axis=-1)
      for pred, probs in zip(batch_predictions, batch_letter_probs):
          specificity.append(int(pred))
          specificity_probs.append(probs.tolist())
  
  return specificity, specificity_probs

def evaluate(model, tokenizer, DH, target, specificity_split, step_idx, args=None):
  model.eval()
  skip_specificity = bool(args and args.skip_specificity)
  skip_new_cot = bool(args and args.skip_new_cot)
  # (0) efficacy: how does the probability of the initial CoT change after unlearning
  # model, tokenizer, prefix, target
  unlearned_cot = target['cot']
  cot_prefix = DH.make_cot_prompt(target['raw_instance'])
  cot_probability = completion_probabilities(model, tokenizer, cot_prefix, [unlearned_cot])

  # (0.1) how does the probability of the _unlearned step_ change after unlearning
  unlearned_step = target['segmented_cot'][step_idx]
  previous_steps = target['segmented_cot'][:step_idx]

  if previous_steps:
      unlearned_step_prefix = '\n'.join([cot_prefix]+previous_steps)
  else:
      unlearned_step_prefix = cot_prefix # First cot step

  # Measure the probability of the targeted step itself, not the whole CoT.
  step_probability = completion_probabilities(model, tokenizer, unlearned_step_prefix, [unlearned_step])

  # (1) faithfulness: how does the model perform wrt. unlearning target
  completion_after, probs_after, prediction_after = answer_probabilities(
  model, tokenizer, DH, target['raw_instance']) # question_prefix

  # (2) "specificity": currently, checks how the model performs on a heldout set of instances from the same dataset: (a) pred, (b) prob
  specificity_predictions, specificity_probabilities = [], []
  if not skip_specificity:
      specificity_predictions, specificity_probabilities = compute_specificity(
          model, tokenizer, DH, specificity_split
      )

  # (3) new CoT: check how the model generated CoT looks like after unlearning
  new_cot = ""
  new_cot_probs = []
  if not skip_new_cot:
      new_cot = complete(model, tokenizer, DH.make_cot_prompt(target['raw_instance']))

  # (4) probability under new CoT (agreement before/after unlearning)
  if not skip_new_cot:
      new_cot_probs, _  = generation_fixed_cot(model, tokenizer, DH, target['raw_instance'], new_cot)

  return_dict = {
      'completion': completion_after,
      'probs': probs_after.tolist(),
      'prediction': prediction_after,
      
      'target_cot_step': unlearned_step,
      'target_cot_step_prefix': unlearned_step_prefix,
      
      'specificity_preds': specificity_predictions,
      'specificity_probs': specificity_probabilities,
      
      'new_cot': new_cot,
      'new_cot_probs': new_cot_probs.tolist() if hasattr(new_cot_probs, "tolist") else new_cot_probs,

      'cot_prob': cot_probability.detach().cpu().float().numpy().tolist(),
      'cot_step_prob': step_probability.detach().cpu().float().numpy().tolist(),
  }

  return return_dict


def load_causal_lm(model_id, device_pref):
    device = resolve_device(device_pref)
    load_kwargs = {
        "trust_remote_code": True,
    }
    if device == "cuda":
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = CLM.from_pretrained(model_id, **load_kwargs)
    if device != "cuda":
        model = model.to(device)
    return model, torch.device(device)

def unlearn_single(model_id, tokenizer, args, target, step_idx, cots_train, cots_verify, dh, instance_idx):
    global _ACTIVE_ORACLE_CACHE, _SHARED_ORACLE_MODELS
    #stepwise 下， 只产生一个样本， retain这个样本里包含 step_idx 之前的步骤， forget这个样本里包含 step_idx 以及之后的步骤。 full_chain 下， retain样本不包含任何步骤， forget样本包含整个cot。
    dataset = cot_to_otfd(
        target,
        cots_train,
        tokenizer,
        n=args.retain_n,
        strategy=args.strategy,
        stepwise=args.stepwise,
        step_idx=step_idx,
        pos=args.pos,
    )

    NT = dataset.num_targets()
    print(f"Num targets: {NT}")
    print(target['segmented_cot'][step_idx])
    if NT <= 2:
         print("-"*20)
         print(f"Too few targets")
         print("-"*20)
         return {'unlearning_results': None, 'mmlu_results':None}

    # Load models only after verifying that this step has enough targets.
    model, device = load_causal_lm(model_id, args.device)
    oracle_cache_key = (model_id, args.device)
    if oracle_cache_key in _SHARED_ORACLE_MODELS:
      oracle_model = _SHARED_ORACLE_MODELS[oracle_cache_key]
    else:
      oracle_model, _ = load_causal_lm(model_id, args.device)
      _SHARED_ORACLE_MODELS[oracle_cache_key] = oracle_model
    collator = FRCollator(tokenizer, device=device)

    EPOCHS = args.epochs
    batch_size = args.batch_size

    # For loop for each of the steps in a Cot
    steps_per_epoch = len(dataset) # Unlearning only one statement
    max_steps = EPOCHS * steps_per_epoch

    print(f"Training, #E={EPOCHS}")
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    _ACTIVE_ORACLE_CACHE = None
    if len(dataset) == 1 and batch_size == 1:
      cached_batch = collator([dataset[0]])
      forget_inputs, retain_inputs = cached_batch
      forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
      oracle_cache = {}
      with torch.no_grad():
        forget_outputs_oracle = oracle_model(
            forget_input_ids,
            labels=forget_labels,
            attention_mask=forget_attention_mask,
        )
        oracle_cache['forget_loss'] = get_batch_loss(forget_outputs_oracle.logits, forget_labels)
        if args.method == 'npo_KL':
          retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
          retain_outputs_oracle = oracle_model(
              retain_input_ids,
              labels=retain_labels,
              attention_mask=retain_attention_mask,
          )
          retain_log_probs = F.log_softmax(retain_outputs_oracle.logits, dim=-1)
          oracle_cache['retain_log_probs'] = retain_log_probs.view(-1, retain_outputs_oracle.logits.shape[-1])
      _ACTIVE_ORACLE_CACHE = oracle_cache

    if args.ff2:
        print("Setting only FF2 parameters to be optimized")
        # model.layers.[num].mlp.down_proj
        # mlp.down_proj.weight is the key for all considered models
        param_key = 'mlp.down_proj.weight'
        for name, param in model.named_parameters():
          if param_key in name:
            param.requires_grad = True
          else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max_steps) # warmup_steps

    results_per_epoch = {}
    # Results before training, for comparison
    if not args.skip_initial_eval:
      results_per_epoch[0] = evaluate(model, tokenizer, dh, target, cots_verify, step_idx=step_idx, args=args)
    
    for epoch in range(EPOCHS):
      model.train()
      optimizer.zero_grad()

      for step, batch in enumerate(train_dataloader):
        loss = compute_loss(model, oracle_model, batch, loss_type=args.method) 

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

      # Eval step
      should_eval = (
          (epoch + 1) == EPOCHS or
          ((epoch + 1) % args.eval_interval == 0)
      )
      if should_eval:
        epoch_result = evaluate(model, tokenizer, dh, target, cots_verify, step_idx=step_idx, args=args)
        results_per_epoch[(epoch+1)] = epoch_result


    # So, this is very ugly but I had to do it quickly and it works ¯\_(ツ)_/¯
    # Calls lm_eval_harness on a saved model checkpoint, extracts the performance from stdout
    #  and then deletes the checkpoint.
    if args.mmlu or args.gsm:
      mod = model_id.split("/")[-1]
      short_model = model_name_dict[mod]
      name = f"{instance_idx}_{step_idx}"
      resdir = f"v2_outputs/chkp/{args.dataset}/{short_model}/"
      print(f"Instance and step idx: {name}")
      os.makedirs(resdir+name, exist_ok=True)
      model.save_pretrained(resdir+name, from_pt=True)
      tokenizer.save_pretrained(resdir+name)        

    # Delete model and clean cuda to free up space for lm eval
    _ACTIVE_ORACLE_CACHE = None
    del collator, train_dataloader, dataset, scheduler, optimizer, model
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    return_dict = {
        'unlearning_results': results_per_epoch,

    }

    if args.mmlu or args.gsm:
      logdir = resdir.replace("chkp", "gen_cap") + f"{args.lr}/"
      os.makedirs(logdir, exist_ok=True)

      print("Running evaluation from python")
      result = run_lm_eval(resdir + name, logdir + name)
      result_lines = result.split("\n")
      score_line = result_lines[-7]
      result_line_parts = score_line.split("|")
      assert result_line_parts[1].strip() == 'mmlu', "Error when retrieving scores"
      mmlu_acc, err = result_line_parts[-4], result_line_parts[-2]
      # print(f"Accuracy and error: {acc} +- {err}")
      key = 'mmlu_results' if args.mmlu else 'gsm8k_results'
      return_dict[key] = mmlu_acc

      print("Deleting model directory")
      shutil.rmtree(resdir + name, ignore_errors=False, onerror=None)

    return return_dict

def load_ids(fin, stepwise=False):
    ids = set()
    if os.path.exists(fin):
      with open(fin, 'r') as infile:
          for line in infile:
              jsonline = json.loads(line)
              id = jsonline.get('id', jsonline['question'])
              if stepwise:
                  id = f"{id}_{jsonline['step_idx']}"
              ids.add(id)
    return ids

def store(instance_info, fout):
    with open(fout, 'a') as outfile:
      outfile.write(json.dumps(instance_info)+"\n")


def load_adapter_index_from_results(fin):
    adapter_index = {}
    if not os.path.exists(fin):
      return []
    with open(fin, 'r') as infile:
      for line in infile:
        if not line.strip():
          continue
        row = json.loads(line)
        adapter_id = row.get('adapter_id') or row.get('adapter_record', {}).get('adapter_id')
        if not adapter_id:
          continue
        record_path = row.get('adapter_record_path')
        record = row.get('adapter_record', {})
        metadata = record.get('metadata', {})
        adapter_index[adapter_id] = {
            'adapter_id': adapter_id,
            'record_path': Path(record_path).name if record_path else f"{adapter_id}.json",
            'adapter_path': record.get('adapter_path', ''),
            'instance_id': row.get('id') or metadata.get('instance_id'),
            'step_idx': row.get('step_idx', metadata.get('step_idx')),
            'target_index': metadata.get('target_index'),
        }
    return sorted(
        adapter_index.values(),
        key=lambda item: (
            item['target_index'] if item['target_index'] is not None else 10**9,
            item['step_idx'] if item['step_idx'] is not None else 10**9,
            item['adapter_id'],
        ),
    )


def parse_lora_target_modules(raw_value: str) -> tuple[str, ...]:
    if not raw_value.strip():
        return tuple()
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def sanitize_adapter_id(raw_value: str) -> str:
    return (
        raw_value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def make_adapter_id(target, step_idx: int, stepwise: bool) -> str:
    base = str(target['id'])
    if stepwise:
        base = f"{base}_step_{step_idx}"
    return sanitize_adapter_id(base)


def chunk_list(items, chunk_size: int):
    if chunk_size <= 0:
        return [items]
    return [items[start:start + chunk_size] for start in range(0, len(items), chunk_size)]

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-3-mini-4k-instruct', help="Model name (hf) or local path")
    parser.add_argument('--dataset', type=str, default='sports', 
                        help="Which dataset to use")
    parser.add_argument('--method', type=str, default='npo_KL', 
                        help="Which unlearning method to use")
    parser.add_argument('--strategy', type=str, default='sentencize', 
                        help="Which unlearning strategy to use: full or sentencize.")
    parser.add_argument('--stepwise', dest='stepwise', action='store_true',
                        help="Unlearn one CoT step at a time (default).")
    parser.add_argument('--full_chain', dest='stepwise', action='store_false',
                        help="Unlearn the full CoT at once instead of stepwise.")
    parser.add_argument('--temperature', type=float, default=0.,
                        help="Sampling temperature for CoT generation")
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help="Device preference for local smoke tests or GPU runs.")
    parser.add_argument('--seed', type=int, default=1001,
                        help="Random seed for the experiments")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of unlearning epochs")
    parser.add_argument('--lr', type=float, default=5e-5,
                        help="Learning rate for NPO")
    parser.add_argument('--new_cot', action='store_true', help="Force generation of a fresh batch of CoTs.")
    parser.add_argument('--atomic', action='store_true', help="Use atomic-statement CoTs if available.")
    parser.add_argument('--pos', action='store_true', help="Filter out function tokens in unlearning.")
    parser.add_argument('--ff2', action='store_true', help="Optimize only the ff2 layers")
    parser.add_argument('--ablation', action='store_true', help="Run on subsample of instances, change logging dir.")
    parser.add_argument('--mmlu', type=int, default=0, help="Evaluate MMLU on a subsample of --mmlu model instances post-unlearning")
    parser.add_argument('--gsm', type=int, default=0, help="Evaluate GSM8K on a subsample of --gsm model instances post-unlearning [WIP]")
    parser.add_argument('--cot_limit', type=int, default=250,
                        help="Maximum number of CoTs to generate/load for the current run.")
    parser.add_argument('--verify_size', type=int, default=20,
                        help="Number of held-out CoT examples used for specificity checks.")
    parser.add_argument('--retain_n', type=int, default=4,
                        help="Number of retain CoTs sampled for the unlearning objective.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Training batch size for the unlearning dataloader.")
    parser.add_argument('--max_instances', type=int, default=0,
                        help="If > 0, only run this many training instances.")
    parser.add_argument('--max_steps_per_instance', type=int, default=0,
                        help="If > 0, only run up to this many CoT steps per instance.")
    parser.add_argument('--eval_interval', type=int, default=1,
                        help="Run evaluation every N epochs and always at the final epoch.")
    parser.add_argument('--skip_specificity', action='store_true',
                        help="Skip held-out specificity evaluation for faster runs.")
    parser.add_argument('--skip_new_cot', action='store_true',
                        help="Skip post-unlearning CoT generation for faster runs.")
    parser.add_argument('--skip_initial_eval', action='store_true',
                        help="Skip epoch-0 evaluation and only evaluate at configured intervals.")
    parser.add_argument('--lora_rank', type=int, default=8,
                        help="LoRA rank for each adapter.")
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help="LoRA alpha scaling.")
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help="LoRA dropout.")
    parser.add_argument('--lora_target_modules', type=str, default="",
                        help="Comma-separated linear module suffixes to wrap with LoRA. Empty means all linear layers.")
    parser.add_argument('--adapter_group_size', type=int, default=0,
                        help="How many adapter jobs to train together in one batched group. 0 means all jobs.")
    parser.set_defaults(stepwise=True)
    
    return parser

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = make_parser().parse_args()

    # Reproducibility
    seed = args.seed
    set_random_seed(seed)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception as exc:
            print(f"Warning: Hugging Face login failed, continuing without explicit login: {exc}")

    model_id = args.model_name
    tokenizer = TOK.from_pretrained(model_id)

    # Fix missing pad token if necessary
    if 'Phi' in model_id:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token

    base_model, device = load_causal_lm(model_id, args.device)

    # Data loading (needs tokenizer)
    # Question, CoT, Answer
    # Always unlearn either (a) one step of a cot or (b) entire cot, 
    #  then evaluate performance / probabilities on the same instance
    #  as well as a small held-out set
    DH = DATASETS[args.dataset]

    # Sentencize option, pos tag option
    cot_data = load_or_generate_dataset_cots(model_id=model_id, tokenizer=tokenizer,
                                              dataset_id=args.dataset,force_generate=args.new_cot, 
                                              sentencize=args.strategy == 'sentencize',
                                              temperature=args.temperature, seed=args.seed,
                                              atomic=args.atomic, max_instances=args.cot_limit,
                                              device_pref=args.device, model=base_model)

    # Shuffle data
    random.shuffle(cot_data)

    # "Specificity" split = same task, different instances
    N_verify = min(args.verify_size, max(1, len(cot_data) - 1))
    cots_train, cots_verify = cot_data[:-N_verify], cot_data[-N_verify:] #
    target_subset = cots_train[:args.max_instances] if args.max_instances > 0 else cots_train

    # Results / dataset / model_id
    mod = model_id.split("/")[-1]
    short_model = model_name_dict[mod]

    # Logging
    if args.mmlu:
      root_name = "v2_outputs/mmlu_results"
      N_unlearn = args.mmlu
    elif args.gsm:
      root_name = "v2_outputs/gsm8k_results"
      N_unlearn = args.gsm
    elif args.ablation:
      root_name = "v2_outputs/ablation"
      N_unlearn = 30
    else:
      root_name = "v2_outputs/final_results"
      N_unlearn = 250
    
    resdir = f"{root_name}/{args.dataset}/{short_model}/"
    os.makedirs(resdir, exist_ok=True)
    # No POS, no ff2, unlearn full
    logfile_name = f"{args.method}_{args.strategy}_s={args.stepwise}_lr={str(args.lr)}_rs={args.seed}_pos={args.pos}_ff2={args.ff2}.out"
    
    # Restore previous results 
    ids = load_ids(resdir + logfile_name, stepwise=args.stepwise)
    print(f"Ids so far: {len(ids)}")
    if args.ff2:
      raise ValueError("FF2-only optimization is incompatible with the multi-adapter LoRA training flow.")
    if not args.stepwise:
      raise NotImplementedError("The multi-adapter LoRA flow is defined for stepwise jobs only; full-chain mode is not supported here.")
    if args.mmlu or args.gsm:
      raise NotImplementedError("The multi-adapter LoRA flow does not yet support lm_eval export paths.")

    oracle_model = base_model

    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=parse_lora_target_modules(args.lora_target_modules),
    )
    adapter_manager = attach_lora_adapters(base_model, lora_config)
    collator = FRCollator(tokenizer, device=device)

    adapter_jobs = []
    n_targets = min(N_unlearn, len(target_subset))
    skipped_jobs = 0
    for idx, target in enumerate(target_subset[:n_targets]):
      n_steps = 1
      if args.stepwise:
        n_steps = len(target['segmented_cot'])
      if args.max_steps_per_instance > 0:
        n_steps = min(n_steps, args.max_steps_per_instance)

      for step_idx in range(n_steps):
        check_id = target['id']
        if args.stepwise:
          check_id = f"{check_id}_{step_idx}"
        if check_id in ids:
          skipped_jobs += 1
          continue

        dataset = cot_to_otfd(
            target,
            cots_train,
            tokenizer,
            n=args.retain_n,
            strategy=args.strategy,
            stepwise=args.stepwise,
            step_idx=step_idx,
            pos=args.pos,
        )
        if hasattr(dataset, "has_valid_retain") and not dataset.has_valid_retain():
          print("-" * 20)
          print(f"Skipping {check_id}: no retain samples; increase --cot_limit or lower --verify_size")
          print("-" * 20)
          continue

        target_count = dataset.num_targets()
        if target_count <= 2:
          print("-" * 20)
          print(f"Skipping {check_id}: too few targets")
          print("-" * 20)
          continue

        adapter_id = make_adapter_id(target, step_idx, args.stepwise)
        adapter_manager.create_adapter(adapter_id)
        sequence_length = len(dataset[0][0][0])
        base_instance_info = {
            'id': target['id'],
            'question': target['question'],
            'step_idx': step_idx,
            'options': target['options'],
            'correct': target['correct_letter'],
            'initial_cot': target['cot'],
            'initial_cot_probs': target['cot_probs'],
            'initial_probs': target['nocot_probs'],
            'prediction': int(np.argmax(target['nocot_probs'])),
            'cot_prediction': int(np.argmax(target['cot_probs'])),
            'adapter_id': adapter_id,
        }
        if args.stepwise:
          base_instance_info['cot_step'] = target['segmented_cot'][step_idx]
          base_instance_info['segmented_cot'] = target['segmented_cot']

        adapter_jobs.append({
            'adapter_id': adapter_id,
            'target': target,
            'step_idx': step_idx,
            'target_index': idx,
            'sequence_length': sequence_length,
            'base_instance_info': base_instance_info,
            'training_job': AdapterTrainingJob(
                adapter_id=adapter_id,
                dataset=dataset,
                collator=collator,
                epochs=args.epochs,
                lr=args.lr,
                loss_type=args.method,
                batch_size=args.batch_size,
                input_pad_value=collator.pad_token_id,
                metadata={
                    'target_id': target['id'],
                    'step_idx': step_idx,
                    'target_index': idx,
                    'question': target['question'],
                },
            ),
        })

    print(f"Pending adapter jobs: {len(adapter_jobs)} (skipped already logged jobs: {skipped_jobs})")

    adapter_root = Path(f"v2_outputs/adapters/{args.dataset}/{short_model}/{logfile_name[:-4]}")
    record_root = Path(f"v2_outputs/adapter_records/{args.dataset}/{short_model}/{logfile_name[:-4]}")
    adapter_root.mkdir(parents=True, exist_ok=True)
    record_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = record_root / "run_manifest.json"
    adapter_index = load_adapter_index_from_results(resdir + logfile_name)

    def write_run_manifest():
      with run_manifest_path.open('w') as outfile:
        json.dump({
            'base_model_id': model_id,
            'dataset': args.dataset,
            'strategy': args.strategy,
            'stepwise': args.stepwise,
            'method': args.method,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'adapter_group_size': args.adapter_group_size,
            'lora': {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'target_modules': list(parse_lora_target_modules(args.lora_target_modules)),
            },
            'adapters': adapter_index,
        }, outfile, ensure_ascii=False, indent=2)

    if not adapter_jobs:
      print("No adapter jobs to run.")
      write_run_manifest()
      return

    adapter_jobs.sort(key=lambda job: job['sequence_length'])
    job_group_size = args.adapter_group_size if args.adapter_group_size > 0 else len(adapter_jobs)
    job_groups = chunk_list(adapter_jobs, job_group_size)

    def scheduler_builder(optimizer, total_steps):
      return get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=0,
          num_training_steps=max(1, total_steps),
      )

    trainer = AdapterTrainer(
        base_model,
        oracle_model,
        adapter_manager,
        scheduler_builder=scheduler_builder,
    )

    eval_results_by_adapter = {job['adapter_id']: {} for job in adapter_jobs}
    if not args.skip_initial_eval:
      for job in adapter_jobs:
        with adapter_manager.activate(job['adapter_id']):
          eval_results_by_adapter[job['adapter_id']][0] = evaluate(
              base_model,
              tokenizer,
              DH,
              job['target'],
              cots_verify,
              step_idx=job['step_idx'],
              args=args,
          )

    def epoch_end_callback(training_job: AdapterTrainingJob, epoch: int):
      should_eval = (epoch == args.epochs) or ((epoch % args.eval_interval) == 0)
      if not should_eval:
        return None
      job_spec = next(job for job in adapter_jobs if job['adapter_id'] == training_job.adapter_id)
      result = evaluate(
          base_model,
          tokenizer,
          DH,
          job_spec['target'],
          cots_verify,
          step_idx=job_spec['step_idx'],
          args=args,
      )
      eval_results_by_adapter[training_job.adapter_id][epoch] = result
      return result

    for group_index, job_group in enumerate(job_groups):
      print(f"Training adapter group {group_index + 1}/{len(job_groups)} with {len(job_group)} jobs")
      group_results = trainer.train_jobs(
          [job['training_job'] for job in job_group],
          compute_loss,
          mode="batched",
          epoch_end_callback=epoch_end_callback,
      )
      result_by_adapter = {result['adapter_id']: result for result in group_results}

      for job in job_group:
        adapter_id = job['adapter_id']
        adapter_path = adapter_root / f"{adapter_id}.pt"
        record_path = record_root / f"{adapter_id}.json"
        relative_adapter_path = os.path.relpath(adapter_path, record_root)
        metadata = {
            'base_model_id': model_id,
            'dataset': args.dataset,
            'strategy': args.strategy,
            'stepwise': args.stepwise,
            'method': args.method,
            'instance_id': job['target']['id'],
            'step_idx': job['step_idx'],
            'target_index': job['target_index'],
            'question': job['target']['question'],
        }
        trainer.save_trained_adapter(adapter_id, adapter_path, metadata=metadata)
        record = AdapterRecord(
            base_model_id=model_id,
            adapter_id=adapter_id,
            adapter_path=relative_adapter_path,
            metadata=metadata,
        )
        with record_path.open('w') as outfile:
          json.dump(record.to_dict(), outfile, ensure_ascii=False, indent=2)
        adapter_index.append({
            'adapter_id': adapter_id,
            'record_path': record_path.name,
            'adapter_path': relative_adapter_path,
            'instance_id': job['target']['id'],
            'step_idx': job['step_idx'],
            'target_index': job['target_index'],
        })

        instance_info = dict(job['base_instance_info'])
        instance_info['adapter_record'] = record.to_dict()
        instance_info['adapter_record_path'] = str(record_path)
        instance_info['adapter_training_history'] = result_by_adapter[adapter_id]['history']
        instance_info['unlearning_results'] = eval_results_by_adapter[adapter_id]
        store(instance_info, resdir + logfile_name)

    write_run_manifest()

    del collator, trainer, base_model
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
