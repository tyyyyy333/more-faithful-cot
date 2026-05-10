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


def load_model_and_tokenizer(model_name, half=True, device_pref="auto"):
  trust_remote_code = True
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
  device = resolve_device(device_pref)

  load_kwargs = {
      "trust_remote_code": trust_remote_code,
  }
  if device == "cuda":
    load_kwargs["torch_dtype"] = torch.bfloat16
    load_kwargs["device_map"] = "auto"
  else:
    load_kwargs["torch_dtype"] = torch.float32

  model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
  if device != "cuda":
    model = model.to(device)
  return model, tokenizer
