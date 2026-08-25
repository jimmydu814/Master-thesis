from pathlib import Path
import argparse
import pandas as pd


GENOME_WIDE_THRESHOLD = 5e-8
SUGGESTIVE_THRESHOLD = 1e-6
CHUNK_SIZE = 250_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=Path,
        help="QC-cleaned NG00182 GWAS file"
    )
    return parser.parse_args()


def phenotype_name(path: Path):
    return path.stem.replace(".cleaned", "")


def main():
    args = parse_args()
    input_path = args.input_file.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    project_root = Path(__file__).resolve().parents[1]
    phenotype = phenotype_name(input_path)

    output_dir = (
        project_root
        / "variant_hits"
        / "NG00182"
        / phenotype
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    significant_file = output_dir / f"{phenotype}_genome_wide_significant.tsv"
    suggestive_file = output_dir / f"{phenotype}_suggestive.tsv"
    summary_file = output_dir / f"{phenotype}_hit_summary.csv"

    significant_file.unlink(missing_ok=True)
    suggestive_file.unlink(missing_ok=True)

    significant_count = 0
    suggestive_count = 0

    sig_header = True
    sug_header = True

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):
        if "P" not in chunk.columns:
            raise ValueError("Input file must contain a P column.")

        chunk["P"] = pd.to_numeric(chunk["P"], errors="coerce")

        # Genome-wide significant
        significant = chunk[
            chunk["P"] < GENOME_WIDE_THRESHOLD
        ]

        # Suggestive only
        suggestive = chunk[
            (chunk["P"] >= GENOME_WIDE_THRESHOLD)
            & (chunk["P"] < SUGGESTIVE_THRESHOLD)
        ]

        if not significant.empty:
            significant.to_csv(
                significant_file,
                sep="\t",
                index=False,
                mode="w" if sig_header else "a",
                header=sig_header
            )
            sig_header = False
            significant_count += len(significant)

        if not suggestive.empty:
            suggestive.to_csv(
                suggestive_file,
                sep="\t",
                index=False,
                mode="w" if sug_header else "a",
                header=sug_header
            )
            sug_header = False
            suggestive_count += len(suggestive)

    summary = pd.DataFrame([{
        "study": "NG00182",
        "phenotype": phenotype,
        "input_file": input_path.name,
        "genome_wide_threshold": GENOME_WIDE_THRESHOLD,
        "suggestive_threshold": SUGGESTIVE_THRESHOLD,
        "genome_wide_significant_variants": significant_count,
        "suggestive_only_variants": suggestive_count
    }])

    summary.to_csv(summary_file, index=False)

    print(f"NG00182: {phenotype}")
    print(f"Genome-wide significant: {significant_count}")
    print(f"Suggestive only: {suggestive_count}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()