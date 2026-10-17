"""The prompt suite.  Three kinds, each probing a different failure:

``gsm8k``   multi-step arithmetic (GSM8K test, first N problems in file order).
            Migration lands mid-reasoning; scored by exact final answer.  Tests
            whether the continuation keeps the *line of reasoning*.
``needle``  a list of similar records with distractors, then a question about
            one.  Migration lands right after the answer's lead-in, before the
            value — the value survives only if the transferred cache still
            carries the prompt.  Seeded generator; exact-match scored.
``open``    open-ended explanations; scored by a fixed judge model (agreement,
            NLL, boundary jump) and by human raters (protocol in
            ``bench/human_eval``).

``python -m bench.prompts`` writes ``bench/prompts.json`` (committed, so every
run and every reader sees the same items).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
OUT = HERE / "prompts.json"

GSM8K_INSTR = ("Solve the problem step by step. Keep it short. On the last line write "
               "'#### ' followed by the final number only.\n\n")

OPEN = [
    "Explain how a refrigerator moves heat out of its interior.",
    "Describe how vaccines train the immune system, for a high-school student.",
    "Why does the Moon always show the same face to the Earth?",
    "Explain what a hash table is and why lookups are fast on average.",
    "How do noise-cancelling headphones work?",
    "Explain the difference between weather and climate with examples.",
    "Describe how a bill becomes a law in a parliamentary system, step by step.",
    "What causes the seasons on Earth? Correct the common misconception.",
    "Explain how public-key cryptography lets strangers communicate securely.",
    "Describe the water cycle and the role of evaporation and condensation.",
    "How does a transistor act as a switch in a digital circuit?",
    "Explain why the sky is blue and sunsets are red.",
    "Describe how GPS determines your position.",
    "Explain compound interest with a short numerical example.",
    "How do plants turn sunlight into chemical energy?",
    "Explain what inflation is and two common causes.",
    "Describe how an electric motor converts electricity into motion.",
    "Explain how the heart pumps blood through the body.",
    "What is the greenhouse effect and how does it warm the planet?",
    "Explain how a search engine ranks web pages, at a high level.",
    "Describe how earthquakes happen and how their size is measured.",
    "Explain how a compiler turns source code into a running program.",
    "How do bees communicate the location of food?",
    "Explain why ice floats on water and why that matters for lakes.",
]

_NAMES = ["Aster", "Birch", "Cedar", "Dahlia", "Elm", "Fern", "Grove", "Hazel", "Iris",
          "Juniper", "Kestrel", "Linden", "Maple", "Nettle", "Oak", "Poplar", "Quince",
          "Rowan", "Sorrel", "Thistle", "Umber", "Violet", "Willow", "Yarrow"]


def needle_items(n: int, seed: int = 7, n_records: int = 14) -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        names = rng.sample(_NAMES, n_records)
        codes = rng.sample(range(1000, 10000), n_records)
        lines = [f"- The access code for the {nm} room is {c}." for nm, c in zip(names, codes)]
        q = rng.randrange(n_records)
        text = ("Here is a list of rooms and their access codes.\n" + "\n".join(lines) +
                f"\n\nWhat is the access code for the {names[q]} room? Answer with one "
                f"sentence in exactly this form: 'The access code for the {names[q]} room is "
                f"<code>.'")
        items.append(dict(id=f"needle-{i:02d}", kind="needle", text=text, answer=str(codes[q]),
                          key=names[q], max_new=32, pressure_at=8))
    return items


def gsm8k_items(n: int) -> List[Dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    p = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    tb = pq.read_table(p).to_pydict()
    out = []
    for i in range(n):
        ans = tb["answer"][i].split("####")[-1].strip().replace(",", "")
        out.append(dict(id=f"gsm8k-{i:03d}", kind="gsm8k", text=GSM8K_INSTR + tb["question"][i],
                        answer=ans, max_new=256, pressure_at=32))
    return out


def open_items() -> List[Dict]:
    return [dict(id=f"open-{i:02d}", kind="open", text=t, answer=None, max_new=160, pressure_at=48)
            for i, t in enumerate(OPEN)]


def build(n_gsm8k: int = 40, n_needle: int = 24) -> List[Dict]:
    return gsm8k_items(n_gsm8k) + needle_items(n_needle) + open_items()


def load(kinds=None) -> List[Dict]:
    items = json.loads(OUT.read_text())
    return [p for p in items if kinds is None or p["kind"] in kinds]


def selftest() -> None:
    a, b = needle_items(3), needle_items(3)
    assert a == b, "needle generator must be deterministic"
    for it in a:
        assert it["answer"] in it["text"] and it["text"].count(it["key"]) >= 2
    print(f"  needle items deterministic; e.g. key={a[0]['key']} answer={a[0]['answer']}")
    print("prompts: ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        items = build()
        OUT.write_text(json.dumps(items, indent=1, ensure_ascii=False))
        print(f"wrote {len(items)} prompts -> {OUT}")
