## Summary

| | C ridge mid-decode (Heo et al., hand-off shape) | D molt | A′ re-prefill | B always small |
|---|---:|---:|---:|---:|
| needle, 3B → 1.5B | {{q.3b-15b.C.needle}} | {{q.3b-15b.D.needle}} | **{{q.3b-15b.A2.needle}}** | {{q.3b-15b.B.needle}} |
| needle at ~2 100 tokens, 3B → 1.5B | {{q.3b-15b.C.longneedle}} | {{q.3b-15b.D.longneedle}} | **{{q.3b-15b.A2.longneedle}}** | {{q.3b-15b.B.longneedle}} |
| needle, 1.5B → 0.5B | {{q.15b-05b.C.needle}} | {{q.15b-05b.D.needle}} | **{{q.15b-05b.A2.needle}}** | {{q.15b-05b.B.needle}} |
| GSM8K, 3B → 1.5B (big alone {{q.3b-15b.big.gsm8k}}) | {{q.3b-15b.C.gsm8k}} | {{q.3b-15b.D.gsm8k}} | **{{q.3b-15b.A2.gsm8k}}** | {{q.3b-15b.B.gsm8k}} |
| judge agreement after switch, 3B → 1.5B | {{q.3b-15b.C.agree}} | {{q.3b-15b.D.agree}} | **{{q.3b-15b.A2.agree}}** | {{q.3b-15b.B.agree}} |
| survives h = 1112 MiB, 3B → 1.5B (co-resident needs {{m.3b-15b.need}}) | {{m.3b-15b.h1112.C.surv}} | {{m.3b-15b.h1112.D.surv}} | {{m.3b-15b.h1112.A2.surv}} | {{m.3b-15b.h1112.B.surv}} |
| time to resume, short prompts (s) | killed | {{m.3b-15b.h1112.D.resume}} | **{{m.3b-15b.h1112.A2.resume}}** | — |
| time to resume, ~2 100-token prompts (s) | killed | **{{m.long.3b-15b.h1112.D.resume}}** | {{m.long.3b-15b.h1112.A2.resume}} | — |

## Where the hypothesis fails, stated plainly

**The transferred cache does not carry the content.** The design hypothesis was
that carrying the KV cache across the switch would let the small model continue
the same sentence better than re-reading the text, and that molt's additions
would make the prior closed-form map sufficient. For both pairs we measured, it
does not. With a mapped cache, the receiver keeps the topic and the voice but
loses the specifics (needle accuracy in the table above).

molt's native anchors make the seam smaller than C's: paired agreement
+{{qp.3b-15b.D.C.agree_post}} {{qp.3b-15b.D.C.agree_post.ci}} for 3B → 1.5B.
Task score is not distinguishable ({{qp.3b-15b.D.C.task}} {{qp.3b-15b.D.C.task.ci}}).
"Better than C" and "good enough" are different claims, and only the first holds
here. It is not an implementation fault: the same code path with an identity
map reproduces native decoding on {{id.same}} of {{id.items}} needle items.

**Where the transfer wins, it wins at the wrong thing.** At short context
re-prefill also resumes faster ({{m.3b-15b.h1112.A2.resume}} s vs
{{m.3b-15b.h1112.D.resume}} s for 3B → 1.5B), because molt pays a second release.
At ~2 100 tokens the transfer is faster ({{m.long.3b-15b.h1112.D.resume}} s vs
{{m.long.3b-15b.h1112.A2.resume}} s). On the same long context, though, it answers
the needle {{q.3b-15b.D.longneedle}} where re-prefill answers
{{q.3b-15b.A2.longneedle}}. The seconds it saves are the seconds needed to get
the answer right. Re-prefill meets all three of molt's conditions: mid-decode,
headroom-triggered, release-first.

**The careful release is a fix for one allocator setting.** Under MLX's default
buffer cache:

- drop-then-load is killed in every cell of both pairs (3B → 1.5B at
  h = 2353: {{m.3b-15b.h2353.Ddelonly.surv}});
- a plain clear-without-sync is erratic for 1.5B → 0.5B and always killed for
  3B → 1.5B.

Under the 20 MiB cache limit that mlx-swift's examples use on iOS, both survive
every cell (3B → 1.5B, h = 1112: {{m.c20.3b-15b.h1112.Ddelonly.surv}}). Only the
order is robust: never load the receiver before releasing the sender.

**molt v0.1's mapper does not survive.** Its depth-linear layer pairing scores
held-out NLL {{fit.3b-15b.depth_nll}} and {{fit.15b-05b.depth_nll}}, against
{{fit.3b-15b.kstar_nll}} and {{fit.15b-05b.kstar_nll}} for Heo et al.'s top-k.

**What holds.** Below the headroom a co-resident switch needs
({{m.15b-05b.need}} and {{m.3b-15b.need}} MiB), the prefill-hand-off shape is
killed. A release-first switch survives with no retracted token, under both
allocator settings and at both context lengths. Every measurement favours
giving the receiver the transcript, not the transferred KV.
