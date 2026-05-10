import os, json, copy, random
from pathlib import Path

import random
import spacy
import datasets
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from evaluate import generate_dataset_cots
from segment import align_cot_to_pos


IGNORE_IDX = -100
_SPACY_NLP = None

model_name_dict = {
    'Phi-3-mini-4k-instruct': 'Phi-3',
    'Meta-Llama-3-8B-Instruct': 'LLaMA-3',
    'Llama-3.2-3B-Instruct': 'LLaMA-3-3B',
    'Qwen2.5-0.5B-Instruct': 'Qwen-0.5B',
    'Meta-Llama-3-70B-Instruct': 'LLaMA-3-70B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-2',
    'phi-2': 'Phi-2',
    'llama-2-hf': 'LLaMA-2',
    'Llama-2-7b-chat-hf': 'LLaMA-2',
    'Mistral-7B-Instruct-v0.1': 'Mistral-1',
}

def cache_cots(dataset_cots, root, model_id, dataset_id, seed, temp, max_instances=250):
  limit_tag = "" if max_instances == 250 else f"_n={max_instances}"
  floc = f"{root}/{dataset_id}/{model_id}_s={seed}_t={temp}{limit_tag}_cots.jsonl"
  dir = f"{root}/{dataset_id}"
  os.makedirs(dir, exist_ok=True)
  with open(floc, 'w') as outfile:
    for line in dataset_cots:
      outfile.write(json.dumps(line) + "\n")

def load_or_generate_dataset_cots(model_id, tokenizer, dataset_id, seed, temperature,
                                  force_generate=False, sentencize=True, atomic=False,
                                  max_instances=250, device_pref='auto', model=None):
    root = 'final_cot' if not atomic else 'atomic_cot'
    temp = f"{temperature:.{2}}"
    short_model_id = model_id.split("/")[-1]
    limit_tag = "" if max_instances == 250 else f"_n={max_instances}"
    floc = f"{root}/{dataset_id}/{short_model_id}_s={seed}_t={temp}{limit_tag}_cots.jsonl"
    if not os.path.exists(floc) or force_generate:
        dataset_cots = generate_dataset_cots(
            model_id,
            tokenizer,
            dataset_id,
            temperature=temperature,
            sentencize=sentencize,
            max_instances=max_instances,
            device_pref=device_pref,
            model=model,
        )
        cache_cots(dataset_cots, root, short_model_id, dataset_id, seed, temp, max_instances=max_instances)
        # Store dependent on seed/temperature
        return dataset_cots
    else:
        return load_jsonl(floc)

def left_pad_sequence(vector_list, padding_value):
    # print(vector_list)
    N = len(vector_list)
    T = max([len(f) for f in vector_list])
    # print(N, T)

    ret = torch.full((N,T), fill_value=padding_value, dtype=vector_list[0].dtype)
    for i, v_i in enumerate(vector_list):
        L = len(v_i)
        ret[i, T-L:] = v_i # Leave padding on left
    return ret

# Single-sample fallback kept for POS-filtered paths; the main non-POS path is batched in SegmentOTFDataset._preencode_samples.
def qcot_encoder(tokenizer, question, cot, pos_filter=False, nlp=None):
    question += "\n\n"
    question_tokens = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt')[0]

    # input = question + cot # newlines are already prepended
  
    if pos_filter:
      cot_tokens, word_to_span = align_cot_to_pos(cot, tokenizer, tokenizer.name_or_path, nlp=nlp)
    else:
      cot_tokens = tokenizer.encode(cot, add_special_tokens=False, return_tensors='pt').squeeze()

    encoded_input = torch.cat((question_tokens, cot_tokens), dim=0)

    labels = encoded_input.clone() # IMPORTANT: Shift by one when computing loss
    attention_mask = torch.ones_like(encoded_input)

    # Do not unlearn the question, only the CoT
    QL = len(question_tokens)
    labels[:QL] = IGNORE_IDX
    
    if pos_filter:
      for w in word_to_span:
        if not w.is_content():
          # Mask out function words from loss
          labels[QL + w.span_start: QL + w.span_end] = IGNORE_IDX

    L = (labels != IGNORE_IDX).sum()
    return encoded_input, labels, attention_mask, L

#On-the-Fly
class OTFDataset(Dataset):
    def __init__(self, forget, retain):
        self.forget = forget # Either a single sentence or a set of atomic statements
        self.retain = retain

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # 1. Take a forget sample at the given index
        forget_sample = self.forget[idx]

        # 2. Take a (random?) retain sample 
        retain_sample = self.retain[idx]

        # Tokenization, padding etc done in collator
        return [forget_sample, retain_sample]

class SegmentOTFDataset(Dataset):
    def __init__(self, forget, retain, tokenizer, stepwise=False, pos_filter=False, step_idx=0):
        self.forget = forget # Either a single sentence or a set of atomic statements
        self.retain = retain
        self.tokenizer = tokenizer
        self.stepwise = stepwise
        self.step = step_idx
        self.retain_idx = 0
        self.min_targets = 2
        self.pos_filter = pos_filter
        self.NLP = None
        if pos_filter:
            global _SPACY_NLP
            if _SPACY_NLP is None:
                _SPACY_NLP = spacy.load("en_core_web_sm", disable=['ner'])
            self.NLP = _SPACY_NLP

        self._forget_sample = None
        self._retain_sample = None
        self._selected_retain_idx = None
        self._encoded_forget = self._preencode_samples(self.forget)
        self._encoded_retain = self._preencode_samples(self.retain)
        self._selected_retain_idx = self._find_first_valid_retain_idx()

    def __len__(self):
        # If stepwise, we unlearn only one step for each dataset instantiation
        return len(self.forget) if not self.stepwise else 1

    def num_targets(self):
        print(f"L = {len(self)}")
        total_targets = 0
        for idx in range(len(self)):
            cur_idx = self.step if self.stepwise else idx
            total_targets += self._encoded_forget[cur_idx]['target_count']
            print(total_targets)
        return total_targets

    @staticmethod
    def targets(aten):
        return (aten != IGNORE_IDX).sum()

    def _build_prompt(self, sample):
        if 'prefix' in sample:
            return '\n'.join([sample['prompt'], sample['prefix']])
        return sample['prompt']

    def _preencode_samples(self, samples):
        encoded_samples = []
        prompts = [self._build_prompt(sample) for sample in samples]
        completions = [sample['completion'] for sample in samples]

        if not self.pos_filter:
            question_tokens_batch = self.tokenizer(
                [prompt + "\n\n" for prompt in prompts],
                add_special_tokens=False,
            )["input_ids"]
            cot_tokens_batch = self.tokenizer(
                completions,
                add_special_tokens=False,
            )["input_ids"]

            for prompt, completion, question_tokens, cot_tokens in zip(
                prompts,
                completions,
                question_tokens_batch,
                cot_tokens_batch,
            ):
                question_tensor = torch.tensor(question_tokens, dtype=torch.long)
                cot_tensor = torch.tensor(cot_tokens, dtype=torch.long)
                encoded_input = torch.cat((question_tensor, cot_tensor), dim=0)
                labels = encoded_input.clone()
                labels[:len(question_tensor)] = IGNORE_IDX
                attention_mask = torch.ones_like(encoded_input)
                target_count = int(len(cot_tensor))
                encoded_samples.append({
                    'bookkeeping': prompt + "\nTarget:" + completion,
                    'encoded': (encoded_input, labels, attention_mask),
                    'target_count': target_count,
                })
            return encoded_samples

        for prompt, completion in zip(prompts, completions):
            encoded_input, labels, attention_mask, target_count = qcot_encoder(
                self.tokenizer,
                prompt,
                completion,
                pos_filter=self.pos_filter,
                nlp=self.NLP,
            )
            encoded_samples.append({
                'bookkeeping': prompt + "\nTarget:" + completion,
                'encoded': (encoded_input, labels, attention_mask),
                'target_count': int(target_count.item() if torch.is_tensor(target_count) else target_count),
            })
        return encoded_samples

    def _find_first_valid_retain_idx(self):
        for an_idx in range(len(self._encoded_retain)):
            cur_retain_idx = (self.retain_idx + an_idx) % len(self._encoded_retain)
            if self._encoded_retain[cur_retain_idx]['target_count'] > self.min_targets:
                return cur_retain_idx
        return None

    def __getitem__(self, idx):
        idx = self.step if self.stepwise else idx
        forget_sample = self._encoded_forget[idx]
        self._forget_sample = forget_sample['bookkeeping']

        if self._selected_retain_idx is None:
            raise ValueError("No long enough retain samples")
        retain_sample = self._encoded_retain[self._selected_retain_idx]
        self._retain_sample = retain_sample['bookkeeping']

        # Padding etc done in collator
        return forget_sample['encoded'], retain_sample['encoded']

class FRCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
        # print(self.pad_token_id)

    def __call__(self, samples):
        # bsz, [F, R], (3)

        F, R = zip(*samples)
        
        # Alt: turn this into a matrix and then slice rows
        # Alt: transpose somehow? but it's 2x2x3
        E_fb = [f[0] for f in F]
        L_fb = [f[1] for f in F]
        A_fb = [f[2] for f in F]
        
        E_rb = [r[0] for r in R]
        L_rb = [r[1] for r in R]
        A_rb = [r[2] for r in R]

        # print(E_rb)
        # print(L_rb)
        # print(A_rb)
        
        E_fib = left_pad_sequence(E_fb, padding_value=self.pad_token_id)
        L_fib = left_pad_sequence(L_fb, padding_value=IGNORE_IDX)
        A_fib = left_pad_sequence(A_fb, padding_value=0) # 0 > ignore for attention

        E_rib = left_pad_sequence(E_rb, padding_value=self.pad_token_id)
        L_rib = left_pad_sequence(L_rb, padding_value=IGNORE_IDX)
        A_rib = left_pad_sequence(A_rb, padding_value=0) # 0 > ignore for attention

        E_fib = E_fib.to(self.device)
        L_fib = L_fib.to(self.device)
        A_fib = A_fib.to(self.device)

        E_rib = E_rib.to(self.device)
        L_rib = L_rib.to(self.device)
        A_rib = A_rib.to(self.device)

        return (E_fib, L_fib, A_fib), (E_rib, L_rib, A_rib)

# Forget-Retain
class DualCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        forgets, retain = zip(*samples)
        # Tokenize, check max length, stack together
        forget_batch = self.tokenizer.batch_encode_plus(forgets, pad_to_max_length=True)
        retain_batch = self.tokenizer.batch_encode_plus(retain, pad_to_max_length=True)
        return forget_batch, retain_batch

COT_ROOT = Path('cots/full')
SEGMENT_COT_ROOT = Path('cots/sentencized')

def load_jsonl(floc):
    data = []
    with open(floc, 'r') as infile:
        for line in infile:
            data.append(json.loads(line))
    return data

# sports_Phi-3-mini-4k-instruct_cots.json
def load_cotfiles(model='Phi-3-mini-4k-instruct', dataset='sports', root=COT_ROOT):
    short_model = model_name_dict[model]
    cot_data = []
    with open(root / dataset / f"{short_model}_cots.jsonl") as infile:
        for line in infile:
            cot_data.append(json.loads(line))
    return cot_data

def make_targets(cot_dict, segment=lambda d: [(d['cot'], None)]):
    DELIM = '\n\n'
    # We want to have prompt & completion fields
    prompt = cot_dict['cot_prompt']
    ret = []

    # Segment into components to be unlearned
    completions = segment(cot_dict)
    for completion, prefix in completions:
        # So far, only full
        if prefix is not None:
            ret.append(
                {'prompt': prompt, 
                  'completion': completion,
                  'prefix': '\n'.join(prefix)
                })
        else:
            ret.append(
                {'prompt': prompt, 
                 'completion': completion
                })
    return ret

# strategies = full, newline, sentencize, atomic statements
def cot_to_otfd(target, all, tokenizer, n=4, strategy='full', stepwise=True, step_idx=0, pos=False):
    # Target cot, other cots, generate a dataset
    if strategy == 'full':
        all = copy.deepcopy(all)
        all.remove(target)
        
        target = make_targets(target)

        retain = random.sample(all, min(n, len(all)))
        # Format into dict for segment dataset by transforming content into prompt & completion
        retain = [rr for r in retain for rr in make_targets(r)]

        return SegmentOTFDataset(target, retain, tokenizer, stepwise, pos_filter=pos)
    
    elif strategy == 'sentencize':
        all = copy.deepcopy(all)
        all.remove(target)

        def segment(d):
            cot_segments = d['segmented_cot']
            outs = []
            prefixes = []
            for s in cot_segments:
                outs.append((s, list(prefixes)))
                prefixes.append(s)
            return outs

        targets = make_targets(target, segment=segment)

        retain = random.sample(all, min(n, len(all)))
        # Format into dict for segment dataset by transforming content into prompt & completion
        retain = [rr for r in retain for rr in make_targets(r, segment=segment)]

        return SegmentOTFDataset(targets, retain, tokenizer, stepwise, step_idx=step_idx, pos_filter=pos)

    return None
