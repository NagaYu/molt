**At a glance (Qwen2.5-1.5B → 0.5B, 4-bit, M2, jetsam emulation; [all numbers](RESULTS.md)):**

- **Load before release** is the prefill hand-off shape (condition C, Heo et al.'s
  mapper). It is killed whenever spare memory is below the {{m.15b-05b.need}} MiB it
  needs (h = 345 MiB: {{m.15b-05b.h345.C.surv}} survived). **Release first** survives
  every cell ({{m.15b-05b.h345.D.surv}}).
- **But the transferred KV does not carry the prompt's content.** Needle
  accuracy is {{q.15b-05b.D.needle}} with molt's anchors and {{q.15b-05b.C.needle}}
  without, against **{{q.15b-05b.A2.needle}}** when the small model simply re-reads
  the transcript. Re-reading also resumed faster ({{m.15b-05b.h345.A2.resume}} s vs
  {{m.15b-05b.h345.D.resume}} s).
- So the recommendation from our own data: **switch release-first, then
  re-prefill**. KV transfer is cheaper only in raw compute, and only once the
  context is long enough to pay for its extra release. Our switches, at
  {{q.15b-05b.ctx.open}}–{{q.15b-05b.ctx.needle}} tokens, were never that long, and
  its quality gap grows with context.
