import nltk

import torch
import numpy as np

from tqdm import tqdm
from pprint import pprint

from models import load_model_and_tokenizer
from dataload import DATASETS

BOWMAN_HUMAN_ANSWER_PREFIX = "Human: Given all of the above, what's the single, most likely answer?"
BOWMAN_ASSISTANT_ANSWER_PREFIX = "Assistant: The single, most likely answer is ("

ANSWER_LETTERS = ["A", "B", "C", "D", "E"] # No MCQA dataset considered has more than 5

def _batched_answer_generation(model, tokenizer, prompts, answer_indices,
                               max_new_tokens=10, temperature=0.0, do_sample=False):
    device = model.device
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
            max_new_tokens=max_new_tokens,
            output_scores=True,
            temperature=temperature,
            do_sample=do_sample,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    first_token_probs = torch.softmax(answer_output['scores'][0], dim=-1)
    batch_letter_probs = first_token_probs[:, answer_indices].detach().cpu().float().numpy()
    batch_predictions = np.argmax(batch_letter_probs, axis=-1)
    generated_sequences = answer_output.sequences[:, input_ids.shape[-1]:]
    batch_text = [tokenizer.decode(generated) for generated in generated_sequences]
    return batch_text, batch_letter_probs, batch_predictions

def answer_probabilities(model, tokenizer, dh, instance):
    with torch.no_grad():
        n_options = len(dh.get_answer_letters(instance))
        answer_letters = ANSWER_LETTERS[:n_options]
        answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]

        prompt = dh.make_bowman_demonstration(instance)
        batch_text, batch_letter_probs, batch_predictions = _batched_answer_generation(
            model,
            tokenizer,
            [prompt],
            answer_indices,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
        )
        
        return batch_text[0], batch_letter_probs[0], int(batch_predictions[0])

def complete(model, tokenizer, prompt, max_new_tokens=300, temperature=0., do_sample=False, split_newline=True):
  do_sample = temperature > 0. # overwrite
  with torch.no_grad():
    device = model.device
    
    inputs = tokenizer.encode(prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    outputs = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id)

    # 2 take only newly generated output
    output = outputs[0][0]
    new_output = output[inputs.shape[-1]:]
    new_output_text = tokenizer.decode(new_output)
    if split_newline:
      new_output_text = new_output_text.strip().split("\n\n")[0] 
    
    return new_output_text

def letter_completion(model, tokenizer, prompt, N):
  with torch.no_grad():
    answer_letters = ANSWER_LETTERS[:N]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]
    _, batch_letter_probs, batch_predictions = _batched_answer_generation(
        model,
        tokenizer,
        [prompt],
        answer_indices,
        max_new_tokens=20,
        temperature=0.0,
        do_sample=False,
    )
    return batch_letter_probs[0], int(batch_predictions[0])

def generate_dataset_cots(model_id, tokenizer, dataset_id, temperature, sentencize=True,
                          max_instances=250, device_pref='auto', model=None):
    print(f"Generating new CoTs for {model_id}, {dataset_id}, sentencize={sentencize}")
    if model is None:
        model, _ = load_model_and_tokenizer(model_id, device_pref=device_pref)
    DH = DATASETS[dataset_id]
    _, valid, test = DH.get_dataset_splits()
    if dataset_id == 'sqa': test = valid # SQA test doesn't have answers

    device = model.device
    instances = []
    for idx, instance in enumerate(test):
        if idx >= max_instances:
            break
        instances.append(instance)
    instance_info = []
    if not instances:
        return instance_info

    batch_size = 8
    cot_prompts = [DH.make_cot_prompt(instance) for instance in instances]
    nocot_probs_by_idx = [None] * len(instances)
    cots_by_idx = [""] * len(instances)
    cot_probs_by_idx = [None] * len(instances)

    # Old logic kept for diff readability:
    # for idx, instance in tqdm(enumerate(test)):
    #     _, nocot_probs, _ = answer_probabilities(model, tokenizer, DH, instance)
    #     cot = complete(model, tokenizer, DH.make_cot_prompt(instance), temperature=temperature)
    #     cot_probs, _ = generation_fixed_cot(model, tokenizer, DH, instance, cot)
    grouped_answer_indices = {}
    for idx, instance in enumerate(instances):
        n_options = len(DH.get_answer_letters(instance))
        grouped_answer_indices.setdefault(n_options, []).append(idx)

    for _, group_indices in grouped_answer_indices.items():
        answer_letters = ANSWER_LETTERS[:len(DH.get_answer_letters(instances[group_indices[0]]))]
        answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] for L in answer_letters]
        prompts = [DH.make_bowman_demonstration(instances[idx]) for idx in group_indices]
        for start in range(0, len(prompts), batch_size):
            batch_slice = group_indices[start:start + batch_size]
            batch_prompts = prompts[start:start + batch_size]
            old_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            answer_inputs = tokenizer(
                batch_prompts,
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
            for local_idx, probs in enumerate(batch_letter_probs):
                nocot_probs_by_idx[batch_slice[local_idx]] = probs

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for start in tqdm(range(0, len(instances), batch_size)):
        batch_prompts = cot_prompts[start:start + batch_size]
        cot_inputs = tokenizer(
            batch_prompts,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt',
        )
        input_ids = cot_inputs.input_ids.to(device)
        attention_mask = cot_inputs.attention_mask.to(device)
        with torch.no_grad():
            cot_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                output_scores=True,
                temperature=temperature,
                do_sample=temperature > 0.0,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = input_ids.shape[-1]
        generated_sequences = cot_outputs.sequences[:, prompt_width:]
        for local_idx, generated in enumerate(generated_sequences):
            decoded = tokenizer.decode(generated)
            cots_by_idx[start + local_idx] = decoded.strip().split("\n\n")[0]
    tokenizer.padding_side = old_padding_side

    for _, group_indices in grouped_answer_indices.items():
        answer_letters = ANSWER_LETTERS[:len(DH.get_answer_letters(instances[group_indices[0]]))]
        answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] for L in answer_letters]
        answer_prompts = []
        for idx in group_indices:
            cot_text = cots_by_idx[idx].strip().split("\n\n")[0]
            answer_prompts.append(DH.make_answer_prompt(cot_prompts[idx] + cot_text))
        for start in range(0, len(answer_prompts), batch_size):
            batch_slice = group_indices[start:start + batch_size]
            batch_prompts = answer_prompts[start:start + batch_size]
            old_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            answer_inputs = tokenizer(
                batch_prompts,
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
                    max_new_tokens=20,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            first_token_probs = torch.softmax(answer_output['scores'][0], dim=-1)
            batch_letter_probs = first_token_probs[:, answer_indices].detach().cpu().float().numpy()
            for local_idx, probs in enumerate(batch_letter_probs):
                cot_probs_by_idx[batch_slice[local_idx]] = probs

    for idx, instance in enumerate(instances):
        cot = cots_by_idx[idx]
        segmented_cot = None
        if sentencize:
            segmented_cot = nltk.sent_tokenize(cot)
        inst_details = {
            'id': instance[DH.id_key],
            'question': instance[DH.q_key],
            'correct_letter': DH.correct_answer_letter(instance),
            'cot_prompt': cot_prompts[idx],
            'cot': cot,
            'options': DH.get_answer_choices(instance),
            'nocot_probs': nocot_probs_by_idx[idx].tolist(),
            'cot_probs': cot_probs_by_idx[idx].tolist(),
            'segmented_cot': segmented_cot,
            'raw_instance': instance,
        }
        if idx == 1:
            pprint(inst_details)
        instance_info.append(inst_details)
    return instance_info

def generation_fixed_cot(model, tokenizer, dh, instance, cot_text):
  with torch.no_grad():
    n_options = len(dh.get_answer_letters(instance))
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]
    cot_prompt = dh.make_cot_prompt(instance)

    cot_text = cot_text.strip().split("\n\n")[0] # Only up to double newline

    # Step 5: make answer prompt
    answer_prompt = dh.make_answer_prompt(cot_prompt + cot_text)
    _, batch_letter_probs, batch_predictions = _batched_answer_generation(
        model,
        tokenizer,
        [answer_prompt],
        answer_indices,
        max_new_tokens=20,
        temperature=0.0,
        do_sample=False,
    )
    return batch_letter_probs[0], int(batch_predictions[0])

def generate(model, tokenizer, instance):
  with torch.no_grad():
    # Step 1: make answer prompt
    n_options = len(instance['cot_probs'])
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                    for L in answer_letters]

    DELIM = "\n\n"
    answer_prompt = DELIM.join([instance['question'], instance['options']])
    answer_prompt = answer_prompt + f"{DELIM}Answer: ("
    batch_text, batch_letter_probs, batch_predictions = _batched_answer_generation(
        model,
        tokenizer,
        [answer_prompt],
        answer_indices,
        max_new_tokens=10,
        temperature=0.0,
        do_sample=False,
    )
    return batch_text[0], batch_letter_probs[0], int(batch_predictions[0])

def get_cot_prompt(instance):
  LTSBS = "Assistant: Let's think step by step:\n"
  DELIM = "\n\n"
  answer_prompt = DELIM.join([instance['question'], instance['options'], LTSBS])
  return answer_prompt

def generate_cot(model, tokenizer, instance, max_new_tokens=300, temperature=0., do_sample=False):
  with torch.no_grad():
    # "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n"
    LTSBS = "Assistant: Let's think step by step:\n"

    device = model.device

    # 1. Make CoT prompt
    DELIM = "\n\n"
    answer_prompt = DELIM.join([instance['question'], instance['options'], LTSBS])
    answer_inputs = tokenizer.encode(answer_prompt, padding=False, add_special_tokens=False, return_tensors='pt').to(device)

    cot_output = model.generate(input_ids=answer_inputs, max_new_tokens=max_new_tokens,
                                    output_scores=True,
                                    temperature=temperature, do_sample=do_sample,return_dict_in_generate=True,
                                    pad_token_id=tokenizer.pad_token_id) # , num_return_sequences=10

    # 2 take only newly generated output
    cot_output = cot_output[0][0]
    cot_new_output = cot_output[answer_inputs.shape[-1]:]
    cot_new_output_text = tokenizer.decode(cot_new_output)
    cot_new_output_text = cot_new_output_text.strip().split("\n\n")[0] 
    
    return cot_new_output_text, answer_prompt

def cot_generate(model, tokenizer, instance, max_new_tokens=300, temperature=0., do_sample=False):
  with torch.no_grad():
    n_options = len(instance['cot_probs'])
    answer_letters = ANSWER_LETTERS[:n_options]
    answer_indices = [tokenizer.encode(L, add_special_tokens=False)[0] 
                      for L in answer_letters]

    cot_text, cot_prompt = generate_cot(model, tokenizer, instance,
                          max_new_tokens=max_new_tokens, temperature=temperature,do_sample=do_sample)

    prefix = cot_prompt + cot_text
    # 1. Make CoT prompt
    DELIM = "\n"
    answer_prompt = DELIM.join([prefix, BOWMAN_HUMAN_ANSWER_PREFIX, BOWMAN_ASSISTANT_ANSWER_PREFIX])
    batch_text, batch_letter_probs, batch_predictions = _batched_answer_generation(
        model,
        tokenizer,
        [answer_prompt],
        answer_indices,
        max_new_tokens=10,
        temperature=temperature,
        do_sample=do_sample,
    )
    return batch_text[0], batch_letter_probs[0], int(batch_predictions[0]), cot_text, cot_prompt

def completion_probabilities(model, tokenizer, prefix, targets):
    device = model.device
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device) # [1, T]
    prefix_length = prefix_ids.size(-1)

    n_sequences = len(targets)
    n_prefix_ids = prefix_ids.repeat(n_sequences, 1)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    # Convert targets to tensors, concat to inputs
    target_ids = tokenizer(targets, padding=True, return_tensors="pt").input_ids.to(device)

    # Count lengths of individual target sequences for scaling
    non_pad = target_ids != pad_token_id
    lengths = torch.count_nonzero(non_pad, dim=-1)

    # Stack inputs
    input = torch.hstack([
       n_prefix_ids,target_ids[:,:-1] # Exclude last target token from input
                          ])

    # Fwd pass
    outputs = model.forward(input, return_dict=True) # logits = B, T, V
    relevant_logits = outputs['logits'][:,prefix_length-1:]
    # Any benefits from logsoftmax if we want the actual probability in the end?
    # Yes, numeric stability
    token_probs = torch.log_softmax(relevant_logits, dim=-1)

    # Set pad probabilities to one for .prod()
    token_probs[:,:,pad_token_id] = 1.
    target_probs = token_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    target_probs = target_probs.squeeze(1)
    seq_probs = torch.sum(target_probs, dim=1)  # prod if not in logspace
    length_penalty = model.generation_config.length_penalty
    if length_penalty is None:
        length_penalty = 1.0
    seq_probs /= lengths**length_penalty # if not logspace seq_probs /= lengths

    return seq_probs
