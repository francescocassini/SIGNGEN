#!/usr/bin/env python3
import argparse
import csv
import gzip
import os
import pickle
from pathlib import Path


def has_min_frames(path: Path, min_frames: int = 4) -> bool:
    if not path.is_dir():
        return False
    try:
        return sum(1 for _ in path.iterdir()) >= min_frames
    except OSError:
        return False


def read_h2s_ids(csv_path: Path):
    ids = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["SENTENCE_NAME"])
    return ids


def read_split_names(gz_path: Path):
    with gzip.open(gz_path, "rb") as f:
        ann = pickle.load(f)
    return [x["name"] for x in ann]


def coverage(total, ok):
    pct = 0.0 if total == 0 else ok * 100.0 / total
    return f"{ok}/{total} ({pct:.2f}%)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    if args.data_root:
        data = Path(args.data_root).resolve()
    else:
        data_root = os.environ.get("SOKE_DATA_ROOT", "../data")
        data = (repo / data_root).resolve() if not os.path.isabs(data_root) else Path(data_root).resolve()

    h2s = data / "How2Sign"
    csl = data / "CSL-Daily"
    pho = data / "Phoenix_2014T"

    print(f"repo: {repo}")
    print(f"data: {data}")
    print("")

    # How2Sign
    h2s_stats = []
    for split in ["train", "val", "test"]:
        csv_path = h2s / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"
        if not csv_path.exists():
            h2s_stats.append((split, 0, 0, "missing csv"))
            continue
        ids = read_h2s_ids(csv_path)
        ok = sum(1 for sid in ids if has_min_frames(h2s / split / "poses" / sid))
        h2s_stats.append((split, len(ids), ok, ""))

    print("How2Sign coverage")
    for split, total, ok, note in h2s_stats:
        suffix = f" [{note}]" if note else ""
        print(f"  {split}: {coverage(total, ok)}{suffix}")
    print("")

    # CSL
    csl_stats = []
    for split in ["train", "val", "test"]:
        sp = csl / f"csl_clean.{split}"
        if not sp.exists():
            csl_stats.append((split, 0, 0, "missing split"))
            continue
        names = read_split_names(sp)
        ok = sum(1 for name in names if has_min_frames(csl / "poses" / name))
        csl_stats.append((split, len(names), ok, ""))

    print("CSL coverage")
    for split, total, ok, note in csl_stats:
        suffix = f" [{note}]" if note else ""
        print(f"  {split}: {coverage(total, ok)}{suffix}")
    print("")

    # Phoenix
    pho_stats = []
    for split in ["train", "dev", "test"]:
        sp = pho / f"phoenix14t.{split}"
        if not sp.exists():
            pho_stats.append((split, 0, 0, "missing split"))
            continue
        names = read_split_names(sp)
        ok = sum(1 for name in names if has_min_frames(pho / name))
        pho_stats.append((split, len(names), ok, ""))

    print("Phoenix coverage")
    for split, total, ok, note in pho_stats:
        suffix = f" [{note}]" if note else ""
        print(f"  {split}: {coverage(total, ok)}{suffix}")


if __name__ == "__main__":
    main()
