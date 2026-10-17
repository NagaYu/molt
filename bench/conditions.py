"""The benchmark conditions, as data.

Primary (the four the paper compares, plus the strongest simple baseline):

  A    restart        the generation is discarded; the small model starts over
                      from the prompt (what an app does after a jetsam kill or a
                      cautious reload).  Non-co-resident by nature.
  A2   re-prefill     the big model is released, the small one re-reads prompt +
                      everything emitted so far and continues.  No mapper.  Added
                      because it is the obvious alternative a reviewer would ask
                      for; it is *not* one of the four the brief names.
  B    always-small   never migrates.
  C    ridge mid-decode   Heo et al.'s closed-form mapper (selected k*), applied
                      at the token boundary in the prefill-handoff shape: load the
                      receiver and mapper while the sender is resident, map, drop
                      the sender.  No anchors.
  D    molt           same mapper, memory-ordered (sender released and footprint
                      confirmed down before anything is loaded), plus native
                      sink/tail anchors selected on held-out text.

Ablations (one factor each):
  C_ordered        C's computation with D's memory order  (isolates the schedule)
  D_colocated      D's computation with C's memory order  (isolates the anchors)
  D_del_only       D with "drop references, then load"   (MLX allocator cache)
  D_clear_no_wait  D with "drop + clear_cache, load now" (asynchronous reclaim)
  D_depth          D with molt v0.1's depth-linear mapper instead of Heo top-k
  big              the big model with no pressure (quality upper bound only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

PRIMARY = ("A", "A2", "B", "C", "D")
ABLATIONS = ("C_ordered", "D_colocated", "D_del_only", "D_clear_no_wait", "D_depth")


@dataclass(frozen=True)
class Condition:
    name: str
    start_tier: str = "src"          # "src" or "dst"
    migrates: bool = True
    schedule: str = "ordered"
    mapper: Optional[str] = "heo"    # "heo" | "depth" | None
    anchors: bool = False            # use molt's selected (sink, tail)
    restart: bool = False
    reprefill: bool = False
    label: str = ""


CONDITIONS = {c.name: c for c in [
    Condition("big", migrates=False, mapper=None, label="big, no pressure"),
    Condition("static_big", migrates=False, mapper=None, label="big, ignores pressure"),
    Condition("A", restart=True, mapper=None, label="(A) restart"),
    Condition("A2", reprefill=True, mapper=None, label="(A′) re-prefill"),
    Condition("B", start_tier="dst", migrates=False, mapper=None, label="(B) always small"),
    Condition("C", schedule="colocated", label="(C) ridge mid-decode"),
    Condition("D", anchors=True, label="(D) molt"),
    Condition("C_ordered", label="C + ordered"),
    Condition("D_colocated", schedule="colocated", anchors=True, label="D, colocated"),
    Condition("D_del_only", schedule="del_only", anchors=True, label="D, del only"),
    Condition("D_clear_no_wait", schedule="clear_no_wait", anchors=True, label="D, clear no wait"),
    Condition("D_depth", mapper="depth", anchors=True, label="D, depth-linear mapper"),
]}


def selftest() -> None:
    for n in PRIMARY + ABLATIONS:
        assert n in CONDITIONS
    assert CONDITIONS["C"].schedule == "colocated" and not CONDITIONS["C"].anchors
    assert CONDITIONS["C_ordered"].schedule == CONDITIONS["D"].schedule
    print("conditions: ok")


if __name__ == "__main__":
    selftest()
