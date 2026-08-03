#!/usr/bin/env python3
"""Splice the generated results table into README.md.

The README's results section is a *marker*, not prose: this script replaces
whatever sits between the markers with the table that
``figures/make_figures.py`` produced. Keeping the numbers generated rather than
transcribed is the only way a long-lived README stays true after a re-run.

    python scripts/update_readme.py --table figures/results_table.md
"""

from __future__ import annotations

import argparse
import os
import re
import sys

START = "<!-- RESULTS_TABLE -->"
END = "<!-- /RESULTS_TABLE -->"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--readme", default="README.md")
    p.add_argument("--table", default="figures/results_table.md")
    args = p.parse_args(argv)

    if not os.path.exists(args.table):
        print(f"no table at {args.table}; run figures/make_figures.py first",
              file=sys.stderr)
        return 1
    with open(args.table) as fh:
        table = fh.read().strip()
    with open(args.readme) as fh:
        readme = fh.read()

    block = f"{START}\n\n{table}\n\n{END}"
    if START in readme and END in readme:
        readme = re.sub(re.escape(START) + r".*?" + re.escape(END), block,
                        readme, flags=re.S)
    elif START in readme:
        readme = readme.replace(START, block)
    else:
        print(f"{args.readme} has no {START} marker", file=sys.stderr)
        return 1

    with open(args.readme, "w") as fh:
        fh.write(readme)
    print(f"spliced {len(table.splitlines())} table lines into {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
