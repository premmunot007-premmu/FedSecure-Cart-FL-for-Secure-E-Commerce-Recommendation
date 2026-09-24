#!/usr/bin/env python3
"""Convert an Amazon review JSONL file into the ratings CSV schema used by the FL project.

Expected input keys (either Amazon's review schema or the project schema):
- user_id or reviewerID or userId
- asin or movieId
- rating or overall
- timestamp or unixReviewTime

The script writes a CSV with columns:
    userId,movieId,rating,timestamp

Usage examples:
    python3 scripts/convert_amazon_reviews.py \
        --input /Users/mac/Downloads/Movies_and_TV.jsonl \
        --output /Users/mac/Downloads/amazon_movies_tv_ratings.csv \
        --max-rows 200000

    python3 scripts/convert_amazon_reviews.py --input /Users/mac/Downloads/Movies_and_TV.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _get_first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_row(obj: dict[str, Any]) -> dict[str, Any] | None:
    user = _get_first(obj, "userId", "user_id", "reviewerID")
    item = _get_first(obj, "movieId", "asin")
    rating = _get_first(obj, "rating", "overall")
    ts = _get_first(obj, "timestamp", "unixReviewTime")

    if user is None or item is None or rating is None or ts is None:
        return None

    try:
        user_id = str(user)
        movie_id = str(item)
        rating_value = float(rating)
        timestamp_value = int(float(ts))
    except (TypeError, ValueError):
        return None

    return {
        "userId": user_id,
        "movieId": movie_id,
        "rating": rating_value,
        "timestamp": timestamp_value,
    }


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            normalized = normalize_row(obj)
            if normalized is not None:
                yield normalized


def convert_amazon_reviews(input_path: str | Path, output_path: str | Path, max_rows: int | None = None) -> int:
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["userId", "movieId", "rating", "timestamp"])
        writer.writeheader()
        for row in iter_rows(source):
            writer.writerow(row)
            written += 1
            if max_rows is not None and written >= max_rows:
                break
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Amazon review JSONL / JSONGZ to project ratings CSV")
    parser.add_argument("--input", type=str, default="/Users/mac/Downloads/Movies_and_TV.jsonl", help="Path to the source Amazon review file")
    parser.add_argument("--output", type=str, default="/Users/mac/Downloads/amazon_movies_tv_ratings.csv", help="Output CSV path")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows to convert (useful for quick runs)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = convert_amazon_reviews(args.input, args.output, max_rows=args.max_rows)
    print(f"Converted {count} rows to {args.output}")
