"""Calibration and held-out text for fitting KV mappers.

Matches the data recipe of Heo et al. (arXiv:2608.03893): FineWeb-Edu
(``sample-10BT``), 500 sequences of 1 024 tokens.  Their text does not say how a
sequence is cut from a document; we take the **first 1 024 tokens of documents
that have at least that many** (no packing across documents), in file order,
starting at the first row group of ``sample/10BT/000_00000.parquet``.

Only the Parquet row groups actually needed are fetched (HTTP range reads via
``HfFileSystem``), so the full 2 GB shard is never downloaded.  The token ids are
cached under ``~/.cache/molt`` and a manifest (document ids + SHA-256 of the
token stream) is written next to the fitted mappers so a fit can be reproduced
exactly.

Splits are disjoint by document: ``calib`` (500), ``heldout`` (k-selection and
held-out R²), ``eval`` (benchmark continuations).  If the Hub is unreachable the
loader falls back to the locally cached WikiText-103 and says so in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

FWE_PATH = "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/000_00000.parquet"
CACHE_DIR = os.path.expanduser("~/.cache/molt")
SEQ_LEN = 1024
SPLITS = (("calib", 500), ("heldout", 64), ("eval", 64))


@dataclass
class Corpus:
    source: str
    seq_len: int
    splits: Dict[str, List[List[int]]]
    doc_ids: Dict[str, List[str]]

    def manifest(self) -> dict:
        h = hashlib.sha256()
        for name in sorted(self.splits):
            h.update(np.asarray(self.splits[name], dtype=np.uint32).tobytes())
        return dict(source=self.source, seq_len=self.seq_len,
                    sizes={k: len(v) for k, v in self.splits.items()},
                    sha256=h.hexdigest(), doc_ids=self.doc_ids)


def select_sequences(docs: Iterable[Tuple[str, str]], encode: Callable[[str], List[int]],
                     sizes: Sequence[Tuple[str, int]], seq_len: int = SEQ_LEN):
    """First ``seq_len`` tokens of each document that has at least that many,
    assigned to splits in order.  Pure function of its inputs (selftested)."""
    want = list(sizes)
    out: Dict[str, List[List[int]]] = {n: [] for n, _ in want}
    ids: Dict[str, List[str]] = {n: [] for n, _ in want}
    si = 0
    for doc_id, text in docs:
        if si >= len(want):
            break
        toks = encode(text)
        if len(toks) < seq_len:
            continue
        name, size = want[si]
        out[name].append(list(toks[:seq_len]))
        ids[name].append(doc_id)
        if len(out[name]) == size:
            si += 1
    missing = {n: s - len(out[n]) for n, s in want if len(out[n]) < s}
    if missing:
        raise RuntimeError(f"ran out of documents: {missing}")
    return out, ids


def _fineweb_docs() -> Iterator[Tuple[str, str]]:
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    f = HfFileSystem().open(FWE_PATH, block_size=8 * 2 ** 20)
    pf = pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        tb = pf.read_row_group(g, columns=["id", "text"]).to_pydict()
        yield from zip(tb["id"], tb["text"])


def _wikitext_docs() -> Iterator[Tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    doc, start = [], 0
    for i, line in enumerate(ds["text"]):
        if line.startswith(" = ") and not line.startswith(" = = ") and doc:
            yield f"wt103:{start}", "".join(doc)
            doc, start = [], i
        doc.append(line)


def load(encode: Callable[[str], List[int]], tokenizer_tag: str,
         sizes: Sequence[Tuple[str, int]] = SPLITS, seq_len: int = SEQ_LEN,
         allow_fallback: bool = True) -> Corpus:
    """Return the corpus, building and caching it on first use."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = hashlib.sha1(json.dumps([tokenizer_tag, list(sizes), seq_len]).encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"calib_{tag}.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return Corpus(d["source"], d["seq_len"], d["splits"], d["doc_ids"])
    try:
        splits, ids = select_sequences(_fineweb_docs(), encode, sizes, seq_len)
        source = "HuggingFaceFW/fineweb-edu sample-10BT " + FWE_PATH.rsplit("/", 1)[-1]
    except Exception as e:  # network or schema failure
        if not allow_fallback:
            raise
        splits, ids = select_sequences(_wikitext_docs(), encode, sizes, seq_len)
        source = f"FALLBACK wikitext-103-raw-v1 (FineWeb-Edu unavailable: {type(e).__name__})"
    c = Corpus(source, seq_len, splits, ids)
    with open(path, "w") as f:
        json.dump(dict(source=source, seq_len=seq_len, splits=splits, doc_ids=ids), f)
    return c


def selftest() -> None:
    docs = [(f"d{i}", "x" * (i % 7) * 10) for i in range(200)]
    enc = lambda s: [ord(c) for c in s]
    out, ids = select_sequences(docs, enc, [("a", 5), ("b", 3)], seq_len=40)
    assert [len(out["a"]), len(out["b"])] == [5, 3]
    assert all(len(s) == 40 for s in out["a"] + out["b"])
    assert not set(ids["a"]) & set(ids["b"]), "splits must be disjoint by document"
    assert ids["a"][0] == "d4", ids["a"][:2]  # first doc with ≥40 chars
    out2, _ = select_sequences(docs, enc, [("a", 5), ("b", 3)], seq_len=40)
    assert out2 == out
    try:
        select_sequences(docs[:10], enc, [("a", 50)], seq_len=40)
        raise AssertionError("expected shortage error")
    except RuntimeError:
        pass
    m = Corpus("t", 40, out, ids).manifest()
    assert m["sizes"] == {"a": 5, "b": 3} and len(m["sha256"]) == 64
    print("  deterministic, document-disjoint splits; shortage is an error, not a silent shrink")
    print("calib_data: ok")


if __name__ == "__main__":
    selftest()
