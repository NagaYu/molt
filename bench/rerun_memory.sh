#!/bin/sh
# Re-run every memory sweep and resume-cost measurement after the 2026-10-18
# fixes (see bench/results/superseded/README.md).  Sequential on purpose.
set -u
for pair in 1.5b-0.5b 3b-1.5b; do
  python3 -m bench.run_memory --pair "$pair"
  python3 -m bench.run_memory --pair "$pair" --cache-limit-mib 20 --conds A,A2,C,D,C_ordered,D_del_only,D_clear_no_wait
  python3 -m bench.resume_cost --pair "$pair"
done
python3 -m bench.analyze && python3 figures/make_paper_figures.py && python3 -m bench.make_results_md
echo rerun-done
