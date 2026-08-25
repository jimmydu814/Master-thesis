import argparse
import gzip
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 250_000

REQUIRED_COLUMNS = ["CHR", "POS", "P"]

FUMA_COLUMNS = {
    "CHR": "chromosome",
    "POS": "base_pair_location",
    "P": "p_value",
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare NG00175 cleaned GWAS summary statistics for FUMA GRCh38."
    )

    parser.add_argument(
        "input_file",
        help="Cleaned NG00175 TSV file"
    )

    parser.add_argument(
        "output_file",
        help="Output FUMA TSV file (.tsv or .tsv.gz)"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check header before processing
    header = pd.read_csv(
        input_path,
        sep="\t",
        nrows=0
    ).columns.tolist()

    missing = [c for c in REQUIRED_COLUMNS if c not in header]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {header}"
        )

    total_rows = 0
    written_rows = 0
    removed_rows = 0

    # Allow either normal TSV or gzip-compressed TSV output
    if str(output_path).endswith(".gz"):
        outfile = gzip.open(output_path, "wt", encoding="utf-8")
    else:
        outfile = open(output_path, "w", encoding="utf-8", newline="")

    try:
        first_chunk = True

        for chunk in pd.read_csv(
            input_path,
            sep="\t",
            usecols=REQUIRED_COLUMNS,
            chunksize=CHUNK_SIZE
        ):
            total_rows += len(chunk)

            # Convert to numeric
            chunk["CHR"] = pd.to_numeric(chunk["CHR"], errors="coerce")
            chunk["POS"] = pd.to_numeric(chunk["POS"], errors="coerce")
            chunk["P"] = pd.to_numeric(chunk["P"], errors="coerce")

            # Basic FUMA validity checks
            valid = (
                chunk["CHR"].between(1, 22) &
                chunk["POS"].notna() &
                (chunk["POS"] > 0) &
                chunk["P"].notna() &
                (chunk["P"] > 0) &
                (chunk["P"] <= 1)
            )

            removed_rows += (~valid).sum()

            chunk = chunk.loc[valid].copy()

            # Integer chromosome and position
            chunk["CHR"] = chunk["CHR"].astype(int)
            chunk["POS"] = chunk["POS"].astype(int)

            # Rename to exact FUMA GRCh38 column names
            chunk = chunk.rename(columns=FUMA_COLUMNS)

            # Exact required order
            chunk = chunk[
                [
                    "chromosome",
                    "base_pair_location",
                    "p_value"
                ]
            ]

            chunk.to_csv(
                outfile,
                sep="\t",
                index=False,
                header=first_chunk
            )

            written_rows += len(chunk)
            first_chunk = False

    finally:
        outfile.close()

    print("\nFUMA preparation complete")
    print("-------------------------")
    print(f"Input:         {input_path}")
    print(f"Output:        {output_path}")
    print(f"Rows read:     {total_rows:,}")
    print(f"Rows written:  {written_rows:,}")
    print(f"Rows removed:  {removed_rows:,}")
    print("\nFUMA format:")
    print("chromosome  base_pair_location  p_value")


if __name__ == "__main__":
    main()