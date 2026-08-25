from __future__ import annotations

import argparse
import gzip
import os
from collections import Counter
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


CHUNK_SIZE = 250_000
REQUIRED_COLUMNS = ["CHR", "POS", "REF", "ALT", "BETA", "P"]
OUTPUT_COLUMNS = [
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "p_value",
]
MISSING_ALLELE_VALUES = {"", "NA", "NAN", "NULL", "NONE", ".", "N/A"}


def positive_integer(value: str) -> int:
    """Argparse type requiring a positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--chunksize must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned NG00182 GWAS summary statistics for FUMA SNP2GENE (GRCh38)."
    )
    parser.add_argument("input_file", type=Path, help="QC-cleaned NG00182 TSV or TSV.GZ file")
    parser.add_argument("output_file", type=Path, help="FUMA output file (.tsv or .tsv.gz)")
    parser.add_argument(
        "--chunksize",
        type=positive_integer,
        default=CHUNK_SIZE,
        help=f"Rows read per chunk (default: {CHUNK_SIZE:,})",
    )
    return parser.parse_args()


def normalized_path(path: Path) -> str:
    """Return a normalized absolute path suitable for Windows comparisons."""
    return os.path.normcase(str(path.resolve()))


def validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if normalized_path(input_path) == normalized_path(output_path):
        raise ValueError("Input and output paths are the same; refusing to overwrite the input file.")


def verify_header(input_path: Path) -> list[str]:
    available = pd.read_csv(
        input_path,
        sep="\t",
        nrows=0,
        compression="infer",
    ).columns.tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in available]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {available}"
        )
    return available


def open_output(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def validate_and_transform(chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    chromosome = pd.to_numeric(chunk["CHR"], errors="coerce")
    position = pd.to_numeric(chunk["POS"], errors="coerce")
    p_value = pd.to_numeric(chunk["P"], errors="coerce")
    beta_value = pd.to_numeric(chunk["BETA"], errors="coerce")

    chromosome_finite = chromosome.notna() & np.isfinite(chromosome)
    position_finite = position.notna() & np.isfinite(position)
    p_finite = p_value.notna() & np.isfinite(p_value)
    beta_finite = beta_value.notna() & np.isfinite(beta_value)

    chromosome_integer = chromosome_finite & chromosome.eq(np.floor(chromosome))
    position_integer = position_finite & position.eq(np.floor(position))
    valid_chromosome = chromosome_integer & chromosome.between(1, 22)
    valid_position = position_integer & position.gt(0)
    valid_p = p_finite & p_value.gt(0) & p_value.le(1)
    valid_beta = beta_finite

    ref = chunk["REF"].astype(str).str.strip().str.upper()
    alt = chunk["ALT"].astype(str).str.strip().str.upper()
    ref_missing = ref.isin(MISSING_ALLELE_VALUES)
    alt_missing = alt.isin(MISSING_ALLELE_VALUES)
    valid_ref = (~ref_missing) & ref.str.fullmatch(r"[ACGT]+", na=False)
    valid_alt = (~alt_missing) & alt.str.fullmatch(r"[ACGT]+", na=False)
    ref_equals_alt = valid_ref & valid_alt & ref.eq(alt)

    valid = (
        valid_chromosome
        & valid_position
        & valid_p
        & valid_beta
        & valid_ref
        & valid_alt
        & (~ref_equals_alt)
    )

    reason_counts = {
        "invalid_chromosome": int((~valid_chromosome).sum()),
        "invalid_position": int((~valid_position).sum()),
        "invalid_p_value": int((~valid_p).sum()),
        "invalid_beta": int((~valid_beta).sum()),
        "invalid_ref_allele": int((~valid_ref).sum()),
        "invalid_alt_allele": int((~valid_alt).sum()),
        "ref_equals_alt": int(ref_equals_alt.sum()),
    }

    # Keep BETA and P as their stripped original strings after numeric
    # validation. This avoids recalculation and preserves beta signs exactly.
    output = pd.DataFrame(
        {
            "chromosome": chromosome.loc[valid].astype(np.int64),
            "base_pair_location": position.loc[valid].astype(np.int64),
            "effect_allele": alt.loc[valid],
            "other_allele": ref.loc[valid],
            "beta": chunk.loc[valid, "BETA"].astype(str).str.strip(),
            "p_value": chunk.loc[valid, "P"].astype(str).str.strip(),
        },
        columns=OUTPUT_COLUMNS,
    )
    return output, reason_counts


def main() -> None:
    args = parse_args()
    input_path = args.input_file.resolve()
    output_path = args.output_file.resolve()

    validate_paths(input_path, output_path)
    verify_header(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written_rows = 0
    removal_reasons: Counter[str] = Counter()
    first_chunk = True

    with open_output(output_path) as outfile:
        for chunk in pd.read_csv(
            input_path,
            sep="\t",
            usecols=REQUIRED_COLUMNS,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            chunksize=args.chunksize,
            compression="infer",
            on_bad_lines="error",
        ):
            total_rows += len(chunk)
            output, reason_counts = validate_and_transform(chunk)
            removal_reasons.update(reason_counts)

            output.to_csv(
                outfile,
                sep="\t",
                index=False,
                header=first_chunk,
                lineterminator="\n",
            )
            first_chunk = False
            written_rows += len(output)

    removed_rows = total_rows - written_rows
    retained_percent = 100 * written_rows / total_rows if total_rows else 0.0

    print("\nFUMA preparation complete")
    print("-------------------------")
    print(f"Input:             {input_path}")
    print(f"Output:            {output_path}")
    print(f"Rows read:         {total_rows:,}")
    print(f"Rows written:      {written_rows:,}")
    print(f"Rows removed:      {removed_rows:,}")
    print(f"Percentage kept:   {retained_percent:.2f}%")
    print("\nRemoval reasons (counts can overlap):")
    print(f"Invalid chromosome: {removal_reasons['invalid_chromosome']:,}")
    print(f"Invalid position:   {removal_reasons['invalid_position']:,}")
    print(f"Invalid P-value:    {removal_reasons['invalid_p_value']:,}")
    print(f"Invalid beta:       {removal_reasons['invalid_beta']:,}")
    print(f"Invalid REF allele: {removal_reasons['invalid_ref_allele']:,}")
    print(f"Invalid ALT allele: {removal_reasons['invalid_alt_allele']:,}")
    print(f"REF equals ALT:     {removal_reasons['ref_equals_alt']:,}")
    print("\nFUMA columns:")
    print("\t".join(OUTPUT_COLUMNS))


if __name__ == "__main__":
    main()
