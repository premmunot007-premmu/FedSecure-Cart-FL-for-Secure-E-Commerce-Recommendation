#!/usr/bin/env python3
"""Extract listed Amazon reviewer-product pairs as implicit positive interactions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_catalog(input_path: str | Path, output_path: str | Path, category: str = "Electronics") -> int:
    source = Path(input_path)
    target = Path(output_path)
    products = pd.read_csv(source, usecols=["product_id", "category", "user_id"])
    products = products[products["category"].fillna("").str.startswith(category)].dropna(subset=["product_id", "user_id"])

    interactions = [
        {"userId": user_id.strip(), "movieId": str(product_id).strip(), "rating": 1.0}
        for product_id, users in zip(products["product_id"], products["user_id"])
        for user_id in str(users).split(",")
        if user_id.strip()
    ]
    result = pd.DataFrame(interactions, columns=["userId", "movieId", "rating"]).drop_duplicates()
    if result.empty:
        raise ValueError(f"No reviewer-product pairs found for category prefix {category!r}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    print(f"Wrote {len(result)} implicit interactions for {result['userId'].nunique()} users and {result['movieId'].nunique()} products to {target}")
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/Users/mac/Desktop/TS_Material/amazon.csv", help="Amazon product catalog CSV")
    parser.add_argument("--output", default="/Users/mac/Desktop/TS_Material/amazon_electronics_implicit.csv", help="Output interaction CSV")
    parser.add_argument("--category", default="Electronics", help="Category prefix to include")
    args = parser.parse_args()
    convert_catalog(args.input, args.output, args.category)


if __name__ == "__main__":
    main()