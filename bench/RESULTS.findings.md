## Summary (Qwen2.5-1.5B → 0.5B, 4-bit; 3B → 1.5B still running)

| | C ridge mid-decode (Heo et al., hand-off shape) | D molt | A′ re-prefill | B always small |
|---|---:|---:|---:|---:|
| needle exact | {{q.15b-05b.C.needle}} | {{q.15b-05b.D.needle}} | **{{q.15b-05b.A2.needle}}** | {{q.15b-05b.B.needle}} |
| GSM8K exact (big alone: {{q.15b-05b.big.gsm8k}}) | {{q.15b-05b.C.gsm8k}} | {{q.15b-05b.D.gsm8k}} | **{{q.15b-05b.A2.gsm8k}}** | {{q.15b-05b.B.gsm8k}} |
| judge agreement after the switch | {{q.15b-05b.C.agree}} | {{q.15b-05b.D.agree}} | **{{q.15b-05b.A2.agree}}** | {{q.15b-05b.B.agree}} |
| seam jump (nats) | {{q.15b-05b.C.jump}} | {{q.15b-05b.D.jump}} | **{{q.15b-05b.A2.jump}}** | — |
| survives h = 345 MiB (co-resident needs {{m.15b-05b.need}}) | {{m.15b-05b.h345.C.surv}} | {{m.15b-05b.h345.D.surv}} | {{m.15b-05b.h345.A2.surv}} | {{m.15b-05b.h345.B.surv}} |
| time to resume at h = 345 (s) | killed | {{m.15b-05b.h345.D.resume}} | **{{m.15b-05b.h345.A2.resume}}** | — |

## Where the hypothesis fails, stated plainly

**The transferred cache does not carry the content.** The design hypothesis was
that carrying the KV cache across the switch would let the small model continue
the same sentence better than re-reading the text, and that molt's additions
would make the prior closed-form map sufficient. It does not. With a mapped
cache, the receiver keeps the topic and the voice but loses the specifics:

- needle accuracy is {{q.15b-05b.C.needle}} (C) and {{q.15b-05b.D.needle}} (D), against
  {{q.15b-05b.A2.needle}} for re-prefill;
- GSM8K is {{q.15b-05b.C.gsm8k}} and {{q.15b-05b.D.gsm8k}}, against {{q.15b-05b.A2.gsm8k}}.

molt's native anchors make the seam smaller than C's: paired agreement
+{{qp.15b-05b.D.C.agree_post}} {{qp.15b-05b.D.C.agree_post.ci}}, seam
{{qp.15b-05b.D.C.jump}} {{qp.15b-05b.D.C.jump.ci}} nats. Task score does not move
({{qp.15b-05b.D.C.task}} {{qp.15b-05b.D.C.task.ci}}). "Better than C" and "good
enough" are different claims, and only the first holds here.

**Re-prefill wins in exactly the setting we claim.** A′ switches mid-decode, is
triggered by headroom, and releases the sender before loading the receiver, so
it satisfies all three of molt's conditions. It survives everywhere C dies.
It resumes faster ({{m.15b-05b.h345.A2.resume}} s vs {{m.15b-05b.h345.D.resume}} s)
with less CPU energy ({{m.15b-05b.h345.A2.ecpu}} J vs {{m.15b-05b.h345.D.ecpu}} J).
The mapping is cheaper only in compute, and only at long context
({{r.15b-05b.D.P4096}} s vs {{r.15b-05b.A2.P4096}} s at 4 096 tokens). Our switches
happened at {{q.15b-05b.ctx.open}}–{{q.15b-05b.ctx.needle}} tokens, and the
intrinsic quality gap grows with context ({{int.15b-05b.D.P768}} nats at 768).

**The careful release is a fix for one allocator setting.** Under MLX's default
buffer cache:

- drop-then-load is killed in every cell (h = 818: {{m.15b-05b.h818.Ddelonly.surv}});
- clear-without-sync is erratic ({{m.15b-05b.h128.Dclearnowait.surv}},
  {{m.15b-05b.h205.Dclearnowait.surv}}, {{m.15b-05b.h345.Dclearnowait.surv}},
  {{m.15b-05b.h569.Dclearnowait.surv}}, {{m.15b-05b.h818.Dclearnowait.surv}}).

Under the 20 MiB cache limit that mlx-swift's examples use on iOS, both survive
(h = 345: {{m.c20.15b-05b.h345.Ddelonly.surv}}, {{m.c20.15b-05b.h345.Dclearnowait.surv}}).
Only the order is robust: never load the receiver before releasing the sender.

**molt v0.1's mapper does not survive.** Its depth-linear layer pairing scores
held-out NLL {{fit.15b-05b.depth_nll}}, against {{fit.15b-05b.kstar_nll}} for Heo et
al.'s top-k={{fit.15b-05b.kstar}}.

**What holds.** Below the headroom a co-resident switch needs
({{m.15b-05b.need}} MiB), the prefill-hand-off shape is killed and a release-first
switch survives with no retracted token. At the context lengths we measured,
the receiver should be given the transcript, not the transferred KV.
