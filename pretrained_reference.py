from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_IDS = {
    "tiny-gpt2": "sshleifer/tiny-gpt2",
    "tiny-random-gpt2": "hf-internal-testing/tiny-random-gpt2",
}


@functools.lru_cache(maxsize=2)
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def generate_reference(
    prompt: str,
    model_name: str = "tiny-gpt2",
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
) -> dict[str, Any]:
    model_id = MODEL_IDS[model_name]
    tokenizer, model = load_model(model_id)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return {
        "model": model_id,
        "source": "Hugging Face Hub",
        "prompt": prompt,
        "text": generated[len(prompt):].strip() or generated,
    }