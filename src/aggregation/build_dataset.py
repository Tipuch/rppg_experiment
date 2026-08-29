"""Build the unified table: facial images + HR + SBP + DBP.

    uv run python -m src.aggregation.build_dataset

Scope is the strict intersection of the datasets that carry blood pressure AND
filmed faces: CLBP-300 and MCD-rPPG. See schema.py for what was excluded and why.

MCD videos arrive via git-lfs. This is re-runnable: recordings whose .avi is
still a pointer are skipped now and picked up once the pull delivers them.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .extractors import clbp300, mcd
from .manifest import assert_disjoint_subjects, build, coverage, validate
from .schema import TARGETS
from .splits import assign, summarise

DATASETS = Path("datasets")
BUILD = Path("build")
STORE = BUILD / "frames"
OUT = BUILD / "dataset.parquet"

SOURCES = [
    ("clbp300", clbp300.extract_all, DATASETS / "clbp-300-sample/ClBP-300_samples"),
    ("mcd", mcd.extract_all, DATASETS / "mcd_rppg"),
]


def main() -> int:
    STORE.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name, fn, root in SOURCES:
        if not root.exists():
            print(f"  {name:9} SKIP  (missing {root})")
            continue
        t = time.time()
        try:
            got = fn(root, STORE)
        except Exception as exc:  # noqa: BLE001 - one illegible record must not end the pass  # one bad source must not sink the build
            print(f"  {name:9} ERROR {type(exc).__name__}: {exc}")
            continue
        rows.extend(got)
        print(f"  {name:9} {len(got):6d} rows  ({time.time() - t:.0f}s)")

    df = build(rows)
    if df.height == 0:
        print("\nno rows extracted")
        return 1

    df = assign(df)
    print(f"\ntotal: {df.height} rows, {df['subject_id'].n_unique()} subjects\n")
    print(coverage(df))
    print()
    print(summarise(df))

    complete = df.drop_nulls(subset=TARGETS).height
    print(f"\nrows with all {len(TARGETS)} targets: {complete} / {df.height}")

    problems = validate(df) + assert_disjoint_subjects(df)
    if problems:
        print("\nVALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nvalidation passed")

    if problems:
        # Never leave a table that failed the gate where a loader will find it.
        failed = OUT.with_suffix(".failed.parquet")
        df.write_parquet(failed)
        print(f"wrote {failed} (NOT promoted: validation failed)")
        return 1

    df.write_parquet(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
