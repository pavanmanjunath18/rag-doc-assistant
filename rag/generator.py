"""Small local LLM for answer generation (no API key needed)."""

import logging
from functools import lru_cache

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from rag.config import GEN_MODEL, MAX_NEW_TOKENS
from rag.prompts import Message

logger = logging.getLogger(__name__)


def pick_device() -> tuple[str, torch.dtype]:
    """Choose where to run the model: NVIDIA GPU, then Apple GPU, then CPU.

    GPUs get float16 (half the memory of float32). CPUs get float32, since float16 is slow
    or unsupported there.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _load() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    device, dtype = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    # Placed explicitly rather than with device_map="auto", which silently offloaded
    # weights to disk on a 16 GB Mac and later crashed (docs/decisions.md, D12).
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=dtype).to(device)
    logger.info("Loaded %s on %s (%s)", GEN_MODEL, model.device, dtype)
    if device == "cpu":
        logger.warning("No GPU found: generation will be much slower than on a GPU")
    return tokenizer, model


def generate(messages: list[Message], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate a reply to chat `messages` with greedy decoding (same input, same answer)."""
    tokenizer, model = _load()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # Qwen's generation config ships sampling settings; clear them since we decode greedily.
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
