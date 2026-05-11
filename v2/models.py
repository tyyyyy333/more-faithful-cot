import random
import torch
import transformers
import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device_pref="auto"):
  if device_pref != "auto":
    return device_pref
  if torch.cuda.is_available():
    return "cuda"
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def model_input_device(model, fallback_device=None):
  try:
    return model.get_input_embeddings().weight.device
  except Exception:
    try:
      return next(model.parameters()).device
    except StopIteration:
      return torch.device(fallback_device or "cpu")


def load_model_and_tokenizer(model_name, half=True, device_pref="auto", device_map="auto"):
  trust_remote_code = True
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
  if tokenizer.pad_token is None:
    if 'Phi' in model_name:
      tokenizer.pad_token = tokenizer.unk_token
    else:
      tokenizer.pad_token = tokenizer.eos_token
  device = resolve_device(device_pref)

  load_kwargs = {
      "trust_remote_code": trust_remote_code,
  }
  using_device_map = device == "cuda" and device_map and device_map != "none"
  if device == "cuda":
    load_kwargs["torch_dtype"] = torch.bfloat16
    if using_device_map:
      load_kwargs["device_map"] = device_map
  else:
    load_kwargs["torch_dtype"] = torch.float32

  model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
  if not using_device_map:
    model = model.to(device)
  return model, tokenizer
