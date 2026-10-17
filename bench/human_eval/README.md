# Human evaluation of the visible seam: protocol

**Status: PENDING.** No ratings have been collected. The paper and `RESULTS.md`
report the judge-model seam metric only, and label the human column PENDING.

## What raters see

Each item is an open-ended prompt and two continuations of **the same opening
text**. In both, the first part (up to the switch token) was produced by the big
model. After that point, one continuation comes from the condition under test
(C, D or B) and the other from A′, where the small model re-reads the
transcript. Raters are not told where the switch is, or that there is one.
Sides are randomised with a fixed seed. The key is kept in `key.json`, which
raters never see.

Raters answer three questions per item:

1. **Seam.** "Is there a point where the text stops making sense or changes
   style abruptly?" Answer for each side: *no* / *slight* / *clear*. If not
   *no*, click the first word where it happens.
2. **Preference.** "Which continuation is better overall?" Left / right / tie.
3. **Fluency.** Rate each side 1–5.

## Design

- Items are the 24 `open-*` prompts × 3 comparisons (C vs A′, D vs A′,
  B vs A′) = 72 pairs per model pair.
- At least 3 raters per pair. Raters must be fluent English readers. Authors do
  not rate.
- Agreement is reported as Krippendorff's α on the seam scale.
- The primary outcome is the rate of *clear* seams on the condition side minus
  the rate on the A′ side, paired by item, with a 95% bootstrap CI.
- Seam clicks within ±8 tokens of the true switch count as *localised*. This
  checks that raters are reacting to the switch and not to some other defect.

## Files

    python -m bench.human_eval.export --pair 1.5b-0.5b     # -> pairs.csv + key.json
    # ... raters fill ratings.csv (columns: rater,pair_id,seam_left,seam_right,click_left,click_right,pref,flu_left,flu_right)
    python -m bench.human_eval.score --pair 1.5b-0.5b      # -> bench/results/human/<pair>.json
