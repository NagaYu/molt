"""Quality measurement for generations produced under memory pressure.

Four scores, because the obvious one is a trap.

**Judge perplexity** — teacher-forced NLL of the generated continuation under
the top-tier model.  Reported, but **never read alone**: degenerate, repetitive
text is *highly* predictable and scores beautifully.  A first version of this
benchmark ranked a repetitive arm best on perplexity, which is how that lesson
was learned.

**Judge agreement** — fraction of positions where the top-tier model's own
argmax equals the token that was actually emitted, given the same prefix.  This
is the sharper form of "would the good model have said this", and repetition
does not rescue it.

**Needle recall** — distractor-laden fact retrieval.  Several similar facts are
planted and the question selects one, so a model that copies the nearest number
is wrong.  This is the sharpest probe of a KV transplant: the fact survives only
if the migrated cache still carries the prompt's content.

**Task accuracy** — deterministic short answers, normalised exact match.

A **killed** generation scores 0 on every reference task rather than being
excluded from the average — otherwise the reclaimed condition quietly drops out
of the denominator and looks better for having died.

Claims supported by this module
-------------------------------
* **continuity**: the quality delta between Molt (D) and always-small (B) is the
  "higher average quality than the small model" half of the headline claim; the
  delta against always-large (A, where it survives) bounds what elasticity costs.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------
# prompt suite
# --------------------------------------------------------------------------


@dataclass
class TaskPrompt:
    """One benchmark prompt, optionally with a checkable answer."""

    id: str
    prompt: str
    answer: Optional[str] = None
    kind: str = "open"          # "open" | "needle" | "task"
    max_new_tokens: int = 96


def _filler(n: int) -> str:
    """Neutral filler that lengthens a prompt without adding answerable facts."""
    base = (
        "The device runs several applications at once. Memory is shared between them. "
        "Background work is expected to yield when the foreground needs room. "
        "Schedulers make trade-offs between throughput, latency and footprint. "
        "Caches trade memory for recomputation. Quantisation trades precision for space. "
    )
    out = []
    while len(" ".join(out)) < n:
        out.append(base)
    return " ".join(out)[:n]


def default_suite(long_prompt_chars: int = 1400) -> List[TaskPrompt]:
    """The benchmark's prompt set.

    Two design constraints, both learned the hard way:

    * Prompts are deliberately **long**.  A short prompt makes a re-prefill cheap
      and would understate condition C's cost, so an honest comparison uses the
      regime the technique is actually for — a long conversation interrupted
      mid-answer.
    * Tasks must **discriminate**.  A first version used a single needle
      ("the code is 7413 … what is the code?") and *every* condition scored
      1.00, including the ones that were killed — the answer sits in the prompt,
      so any model that copies wins and the metric measured nothing.  The
      needles below are therefore *distractor-laden*: several similar facts are
      planted and the question selects one of them, so copying the nearest number
      is wrong.
    """
    pad = _filler(long_prompt_chars)
    half = _filler(long_prompt_chars // 2)
    noise = ("Reference codes seen earlier in the log were 2201, 8874 and 3190. "
             "Teams in Sapporo, Nagoya and Fukuoka filed unrelated reports. ")
    return [
        TaskPrompt(
            id="needle_select", kind="needle", answer="7413", max_new_tokens=24,
            prompt=(f"{pad}\n\n{noise}\n"
                    f"The activation code for the maintenance console is 7413.\n"
                    f"The activation code for the diagnostic console is 5628.\n"
                    f"{half}\n\nQuestion: What is the activation code for the maintenance "
                    f"console? Answer with the four digits only.\nAnswer:")),
        TaskPrompt(
            id="needle_second_hop", kind="needle", answer="Hokkaido", max_new_tokens=24,
            prompt=(f"{pad}\n\n{noise}\n"
                    f"Team B is based in Hokkaido. Team A is based in Okinawa.\n"
                    f"The maintenance work was assigned to Team B.\n{half}\n\n"
                    f"Question: In which region is the team that was assigned the "
                    f"maintenance work based? Answer with the region name only.\nAnswer:")),
        TaskPrompt(
            id="task_order", kind="task", answer="kettle", max_new_tokens=16,
            prompt=(f"{pad}\n\nThe crates were unloaded in this order: lantern, kettle, "
                    f"anvil, ledger, rope.\n{half}\n\nQuestion: Which item was unloaded "
                    f"second? Answer with the single word.\nAnswer:")),
        TaskPrompt(
            id="open_explain", kind="open", max_new_tokens=128,
            prompt=(f"{pad}\n\nExplain, in a few sentences, why an operating system might "
                    f"reclaim memory from a background process, and what a well-behaved "
                    f"application should do about it.\nAnswer:")),
        TaskPrompt(
            id="open_story", kind="open", max_new_tokens=128,
            prompt=(f"{pad}\n\nWrite a short, coherent paragraph describing a workshop where "
                    f"an old machine is repaired.\nAnswer:")),
    ]


# --------------------------------------------------------------------------
# judge perplexity
# --------------------------------------------------------------------------


@torch.no_grad()
def judge_nll(judge_model, tokenizer, prompt: str, continuation_ids: Sequence[int],
              device: torch.device, max_ctx: int = 3072) -> Dict[str, float]:
    """Mean NLL (nats/token) of ``continuation_ids`` given ``prompt``, under the judge.

    Only the continuation's positions are scored — the prompt is context, not
    prediction — so a condition is never rewarded for having a different prompt.
    """
    if not continuation_ids:
        return dict(nll=float("nan"), ppl=float("nan"), n=0)
    p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    ids = p_ids + list(continuation_ids)
    if len(ids) > max_ctx:
        drop = len(ids) - max_ctx
        p_ids = p_ids[drop:]
        ids = p_ids + list(continuation_ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = judge_model(x)
    logits = out.logits[0, :-1].float()
    targets = x[0, 1:]
    start = len(p_ids) - 1
    lp = torch.log_softmax(logits[start:], dim=-1)
    tgt = targets[start:]
    nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    m = float(nll.mean())
    # Agreement is reported alongside perplexity because perplexity alone is a
    # trap: degenerate, repetitive text is *highly* predictable and therefore
    # scores well.  Agreement asks the sharper question — at each position,
    # would the good model have chosen this very token?
    agree = float((logits[start:].argmax(-1) == tgt).float().mean())
    return dict(nll=m, ppl=float(math.exp(min(20.0, m))), n=int(tgt.numel()),
                judge_agreement=agree)


# --------------------------------------------------------------------------
# reference scoring
# --------------------------------------------------------------------------


def _normalise(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s)


def exact_match(prediction: str, answer: str) -> float:
    """1.0 when the answer appears in the first line-ish of the prediction.

    Deliberately lenient about surrounding chatter (small models pad), strict
    about the fact itself.
    """
    p, a = _normalise(prediction), _normalise(answer)
    if not a:
        return float("nan")
    head = " ".join(p.split()[:32])
    return 1.0 if a in head else 0.0


def score_generation(task: TaskPrompt, text: str, killed: bool = False) -> Dict[str, Any]:
    """Score one generation.

    A **killed** generation scores 0, not "excluded".  From a user's point of
    view an answer that was cut off by the OS is a failure, and letting the
    reclaimed condition quietly drop out of the average would flatter exactly
    the strategy this project argues against.
    """
    out: Dict[str, Any] = dict(id=task.id, kind=task.kind, killed=bool(killed))
    if task.answer:
        out["correct"] = 0.0 if killed else exact_match(text, task.answer)
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


@dataclass
class QualityReport:
    per_prompt: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, rec: Dict[str, Any]) -> None:
        self.per_prompt.append(rec)

    def _mean(self, key: str, kinds: Optional[Sequence[str]] = None) -> float:
        vals = [r[key] for r in self.per_prompt
                if key in r and r[key] == r[key]
                and (kinds is None or r.get("kind") in kinds)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def summary(self) -> Dict[str, Any]:
        nll = self._mean("nll")
        # NaN must propagate: a condition that produced no scorable text has an
        # *unknown* perplexity, not an astronomically bad one.  ``min(20, nan)``
        # silently returns 20 in Python, which would have printed exp(20) as if
        # it were a measurement.
        ppl = float("nan") if nll != nll else math.exp(min(20.0, nll))
        return dict(
            # NOTE: perplexity is *not* a quality ranking on its own — degenerate
            # repetition is highly predictable and scores well.  Read it together
            # with judge_agreement, repetition_rate and distinct_2.
            judge_nll=nll,
            judge_ppl=ppl,
            judge_agreement=self._mean("judge_agreement"),
            accuracy=self._mean("correct"),
            needle_recall=self._mean("correct", kinds=("needle",)),
            task_accuracy=self._mean("correct", kinds=("task",)),
            repetition_rate=self._mean("repetition_rate"),
            distinct_2=self._mean("distinct_2"),
            n_prompts=len(self.per_prompt),
            n_killed=sum(1 for r in self.per_prompt if r.get("killed")),
        )
