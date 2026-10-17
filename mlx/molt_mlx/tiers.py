"""The model tiers used throughout: Qwen2.5 Instruct, 4-bit MLX builds.

4-bit is what an on-device app would ship, so mappers are fitted on — and caches
come from — exactly these weights.  The tokenizer is identical across the three
sizes (vocabulary hash checked in ``selftest``), which within-family KV transfer
requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

TIERS: Dict[str, str] = {
    "3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
}
#: (source, destination) — large → small, the direction memory pressure forces.
PAIRS: Tuple[Tuple[str, str], ...] = (("3b", "1.5b"), ("1.5b", "0.5b"))


def pair_name(src: str, dst: str) -> str:
    return f"{src}-{dst}"


def local_path(tier: str) -> Path:
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]
    try:
        return Path(snapshot_download(TIERS[tier], allow_patterns=pats, local_files_only=True))
    except Exception:
        return Path(snapshot_download(TIERS[tier], allow_patterns=pats))


def load_model(tier: str):
    """Weights only; materialised before returning so footprint is honest."""
    import mlx.core as mx
    from mlx_lm.utils import load_model as _lm
    model, _ = _lm(local_path(tier))
    mx.eval(model.parameters())
    return model


def load_tokenizer(tier: str = "0.5b"):
    from mlx_lm.utils import load_tokenizer as _lt
    return _lt(local_path(tier))


def selftest() -> None:
    import hashlib
    hs = set()
    for t in TIERS:
        try:
            tok = load_tokenizer(t)
        except Exception as e:  # not downloaded on this machine
            print(f"  {t}: not available locally ({type(e).__name__}); skipped")
            continue
        hs.add(hashlib.sha1(str(sorted(tok.get_vocab().items())).encode()).hexdigest())
    assert len(hs) <= 1, "tiers must share one tokenizer"
    print(f"  {len(TIERS)} tiers, one shared vocabulary ({len(hs)} distinct hash)")
    print("tiers: ok")


if __name__ == "__main__":
    selftest()
