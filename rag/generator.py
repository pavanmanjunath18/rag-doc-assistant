"""Small local LLM for answer generation (no API key needed)."""

import logging
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


class HuggingFaceGenerator:
    """Generator backed by a Hugging Face chat model, loaded once when constructed."""

    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        device, dtype = pick_device()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Placed explicitly rather than with device_map="auto", which silently offloaded
        # weights to disk on a 16 GB Mac and later crashed (docs/decisions.md, D12).
        self._model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self._max_new_tokens = max_new_tokens
        # One model instance serves every request, so run one generation at a time.
        self._lock = threading.Lock()
        self.description = f"{model_name} on {self._model.device} ({dtype})"
        logger.info("Loaded %s", self.description)
        if device == "cpu":
            logger.warning("No GPU found: generation will be much slower than on a GPU")

    def generate(self, messages: list[Message]) -> str:
        """Reply to chat `messages` with greedy decoding (same input, same answer)."""
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._lock, torch.no_grad():
            # Qwen's generation config ships sampling settings; clear them since we decode greedily.
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
