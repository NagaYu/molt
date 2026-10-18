**At a glance (Qwen2.5 4-bit, 3B → 1.5B and 1.5B → 0.5B, M2, jetsam emulation; [every number](RESULTS.md)):**

- **Load before release** is the prefill hand-off shape (condition C, Heo et al.'s
  mapper). It is killed whenever spare memory is below what it needs
  ({{m.3b-15b.need}} MiB for 3B → 1.5B: {{m.3b-15b.h1112.C.surv}} survived at
  h = 1112). **Release first** survives every cell of both pairs.
- **But the transferred KV does not carry the prompt's content.** On a needle
  task with a 3B → 1.5B switch right before the answer, molt answers
  {{q.3b-15b.D.needle}}, the plain map {{q.3b-15b.C.needle}}, and **re-reading the
  transcript {{q.3b-15b.A2.needle}}**. At ~2 100 tokens the transfer resumes
  {{qp.3b-15b.D.A2.longneedle.resume}} s sooner but answers {{q.3b-15b.D.longneedle}},
  against {{q.3b-15b.A2.longneedle}}.
- **The recommendation from our own data: switch release-first, then
  re-prefill.** molt's careful memory release matters only under MLX's
  default cache. With the 20 MiB cache limit mlx-swift uses on iOS, plain
  "drop then load" also survives.
