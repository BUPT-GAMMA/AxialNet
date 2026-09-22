#!/usr/bin/env python3
"""Create a deterministic 10/10/80 seen split plus an unseen-design test set."""

import argparse
import random
from pathlib import Path


SEEN_DESIGNS = ("RISCY-a", "RISCY-b", "RISCY-FPU-a", "RISCY-FPU-b")
UNSEEN_DESIGNS = ("zero-riscy-a", "zero-riscy-b")


def classify(file_name):
    for design in UNSEEN_DESIGNS:
        if design in file_name:
            return design
    for design in SEEN_DESIGNS:
        if design in file_name:
            return design
    return None


def write_ids(path, values):
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory containing one file per sample")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups = {name: [] for name in (*SEEN_DESIGNS, *UNSEEN_DESIGNS)}
    for path in sorted(Path(args.data_dir).iterdir()):
        design = classify(path.name)
        if design:
            sample_id = path.name.removesuffix("_pixel.pt").removesuffix("_net.pt")
            groups[design].append(sample_id)

    rng = random.Random(args.seed)
    train, val, test_seen, test_unseen = [], [], [], []
    for design in SEEN_DESIGNS:
        values = groups[design]
        rng.shuffle(values)
        train_end = int(len(values) * 0.1)
        val_end = train_end + int(len(values) * 0.1)
        train.extend(values[:train_end])
        val.extend(values[train_end:val_end])
        test_seen.extend(values[val_end:])
    for design in UNSEEN_DESIGNS:
        test_unseen.extend(groups[design])

    for values in (train, val, test_seen, test_unseen):
        rng.shuffle(values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ids(output_dir / "train.txt", train)
    write_ids(output_dir / "val.txt", val)
    write_ids(output_dir / "test_seen.txt", test_seen)
    write_ids(output_dir / "test.txt", test_unseen)
    print({"train": len(train), "val": len(val), "test_seen": len(test_seen), "test": len(test_unseen)})


if __name__ == "__main__":
    main()

