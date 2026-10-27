"""Existing guardrails to compare against. Every baseline is a binary BLOCK / ALLOW filter.

FREE and UNGATED (work with no account):
  keyword      simple keyword filter (weak reference point)
  protectai    protectai/deberta-v3-base-prompt-injection-v2  (Apache-2.0)  injection/jailbreak detector
  granite      ibm-granite/granite-guardian-3.2-3b-a800m      (Apache-2.0)  general harm detector

FREE but GATED (accept the licence on Hugging Face, then `huggingface-cli login`):
  llamaguard   meta-llama/Llama-Guard-3-1B
  promptguard  meta-llama/Llama-Prompt-Guard-2-86M

Heavy imports (torch/transformers) happen lazily, only for the baselines you ask for.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import List

KEYWORDS = ["kill", "poison", "bomb", "hack", "lethal", "overdose", "weapon", "suicide",
            "illegal", "steal", "forge", "explosive", "malware", "untraceable"]


def _device():
    import torch
    return 0 if torch.cuda.is_available() else -1


class KeywordBaseline:
    name = "keyword"

    def predict_many(self, texts: List[str]) -> List[str]:
        return ["BLOCK" if any(k in t.lower() for k in KEYWORDS) else "ALLOW" for t in texts]


class ProtectAIBaseline:
    name = "protectai"

    def __init__(self):
        from transformers import pipeline
        self.clf = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2",
                            truncation=True, max_length=512, device=_device())

    def predict_many(self, texts):
        out = self.clf(list(texts), batch_size=16)
        return ["BLOCK" if o["label"].upper() == "INJECTION" else "ALLOW" for o in out]


class GraniteGuardianBaseline:
    name = "granite"

    def __init__(self, model_id="ibm-granite/granite-guardian-3.2-3b-a800m"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev).eval()

    def _one(self, text):
        msgs = [{"role": "user", "content": text[:6000]}]
        ids = self.tok.apply_chat_template(msgs, guardian_config={"risk_name": "harm"},
                                           add_generation_prompt=True, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=5, do_sample=False)
        gen = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip().lower()
        return "BLOCK" if gen.startswith("yes") else "ALLOW"

    def predict_many(self, texts):
        return [self._one(t) for t in texts]


class LlamaGuard3Baseline:
    name = "llamaguard"

    def __init__(self, model_id="meta-llama/Llama-Guard-3-1B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev).eval()

    def _one(self, text):
        conv = [{"role": "user", "content": [{"type": "text", "text": text[:6000]}]}]
        ids = self.tok.apply_chat_template(conv, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=0)
        gen = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip().lower()
        return "BLOCK" if gen.startswith("unsafe") else "ALLOW"

    def predict_many(self, texts):
        return [self._one(t) for t in texts]


class PromptGuard2Baseline:
    name = "promptguard"

    def __init__(self, model_id="meta-llama/Llama-Prompt-Guard-2-86M"):
        from transformers import pipeline
        self.clf = pipeline("text-classification", model=model_id, truncation=True,
                            max_length=512, device=_device())

    def predict_many(self, texts):
        out = self.clf(list(texts), batch_size=16)
        return ["BLOCK" if (o["label"].endswith("1") or "malicious" in o["label"].lower()) else "ALLOW"
                for o in out]


REGISTRY = {"keyword": KeywordBaseline, "protectai": ProtectAIBaseline, "granite": GraniteGuardianBaseline,
            "llamaguard": LlamaGuard3Baseline, "promptguard": PromptGuard2Baseline}


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def run_cached(name: str, texts: List[str], cache_dir: Path, chunk=64, log=print):
    """Predict with disk caching, so an interrupted run resumes where it stopped."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"cache_{name}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [t for t in dict.fromkeys(texts) if sha(t) not in cache]
    if todo:
        log(f"[{name}] predicting {len(todo)} new prompts ({len(cache)} cached)")
        model = REGISTRY[name]()
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            for t, a in zip(part, model.predict_many(part)):
                cache[sha(t)] = a
            path.write_text(json.dumps(cache))
            log(f"  {min(i + chunk, len(todo))}/{len(todo)}")
    return [cache[sha(t)] for t in texts]
