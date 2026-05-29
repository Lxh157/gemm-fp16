#!/usr/bin/env python3
#默认用法：python3 scripts/raw_to_csv.py
#指定输入/输出：python3 scripts/raw_to_csv.py results/raw/bench_xxx.txt
import argparse
import csv
import re
from pathlib import Path


RAW_DIR = Path("results/raw")
OUT_DIR = Path("results/table")

PAT_CASE = re.compile(r"===== impl=([\w_]+), M=N=K=(\d+) =====")
PAT_PERF = re.compile(r"\[perf\]\s+median=([0-9.]+)\s+GFLOP/s")


def pick_latest_raw(raw_dir: Path) -> Path:
    candidates = sorted(
        raw_dir.glob("*.txt"),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No raw result .txt files found under {raw_dir}")
    return candidates[-1]


def parse_raw(path: Path):
    rows = []
    cur_impl = None
    cur_size = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_case = PAT_CASE.search(line)
            if m_case:
                cur_impl = m_case.group(1)
                cur_size = int(m_case.group(2))
                continue

            m_perf = PAT_PERF.search(line)
            if m_perf and cur_impl is not None and cur_size is not None:
                rows.append(
                    {
                        "impl": cur_impl,
                        "size": cur_size,
                        "median_gflops": m_perf.group(1),
                    }
                )
                cur_impl = None
                cur_size = None

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract impl, matrix size, and median GFLOP/s from bench raw output."
    )
    parser.add_argument(
        "raw_file",
        nargs="?",
        type=Path,
        help="Raw result file to parse. Defaults to latest results/raw/*.txt by mtime.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. Defaults to results/table/<raw_file_stem>.csv.",
    )
    args = parser.parse_args()

    raw_path = args.raw_file or pick_latest_raw(RAW_DIR)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw result file not found: {raw_path}")

    out_path = args.output or (OUT_DIR / f"{raw_path.stem}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_raw(raw_path)
    if not rows:
        raise ValueError(f"No benchmark rows parsed from {raw_path}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["impl", "size", "median_gflops"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Parsed: {raw_path}")
    print(f"[OK] Rows: {len(rows)}")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()

