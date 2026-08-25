from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


CHUNK_SIZE = 250_000
PLOT_STRIDE = 20  # Plot every 20th variant, plus every variant with P < 1e-5.
NA_TOKENS = {"", "na", "nan", "null", "none", ".", "n/a"}
REQUIRED_COLUMNS = {
    "chr",
    "pos",
    "MarkerName",
    "P.value",
    "Direction",
    "HetISq",
    "HetPVal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="NG00175 p-value-only summary-statistics file")
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def is_missing(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(NA_TOKENS)


def numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values), np.nan)


def phenotype_name(path: Path) -> str:
    suffix = "_Shade_2024_p-valueOnly"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def output_paths(input_path: Path) -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[1]
    phenotype = phenotype_name(input_path)
    clean_dir = project_root / "data_clean" / "NG00175"
    result_dir = project_root / "qc_results" / "NG00175" / phenotype
    clean_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return {
        "clean": clean_dir / f"{phenotype}.cleaned.tsv",
        "flagged": result_dir / "flagged_variants.tsv",
        "summary": result_dir / "qc_summary.csv",
        "missingness": result_dir / "missingness_summary.csv",
        "manhattan": result_dir / "manhattan.png",
        "qq": result_dir / "qq.png",
    }


def read_chunks(path: Path, chunksize: int):
    # Read fields as text first so malformed values can be detected explicitly.
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
        on_bad_lines="error",  # Never silently discard a wrong-width row.
    )


def evaluate(chunk: pd.DataFrame) -> dict[str, object]:
    raw_chr = chunk["chr"]
    raw_pos = chunk["pos"]
    raw_p = chunk["P.value"]

    chromosome = numeric(raw_chr)
    position = numeric(raw_pos)
    p_value = numeric(raw_p)

    chr_missing = is_missing(raw_chr)
    chr_integer = chromosome.notna() & chromosome.eq(np.floor(chromosome))
    chr_valid = chr_integer & chromosome.between(1, 22)

    pos_missing = is_missing(raw_pos)
    pos_integer = position.notna() & position.eq(np.floor(position))
    pos_valid = pos_integer & position.gt(0)

    p_missing = is_missing(raw_p)
    p_zero = p_value.eq(0)
    p_valid = p_value.notna() & p_value.gt(0) & p_value.le(1)

    marker = chunk["MarkerName"].astype(str).str.strip()
    direction = chunk["Direction"].astype(str).str.strip()
    het_p = numeric(chunk["HetPVal"])
    het_i2 = numeric(chunk["HetISq"])

    flags = {
        "MISSING_CHR": chr_missing,
        "INVALID_CHR": (~chr_missing) & (~chr_valid),
        "MISSING_POS": pos_missing,
        "MALFORMED_POS": (~pos_missing) & position.isna(),
        "NONINTEGER_POS": position.notna() & (~pos_integer),
        "NONPOSITIVE_POS": pos_integer & position.le(0),
        "MISSING_P": p_missing,
        "MALFORMED_P": (~p_missing) & p_value.isna(),
        "P_ZERO": p_zero,
        "P_OUT_OF_RANGE": p_value.notna() & ((p_value < 0) | (p_value > 1)),
        "MISSING_VARIANT_ID": is_missing(chunk["MarkerName"]),
        "INVALID_DIRECTION": ~direction.str.fullmatch(r"[+\-?]{3}", na=False),
        "MALFORMED_HETPVAL": (~is_missing(chunk["HetPVal"])) & het_p.isna(),
        "HETPVAL_OUT_OF_RANGE": het_p.notna() & ((het_p < 0) | (het_p > 1)),
        "MALFORMED_HETISQ": (~is_missing(chunk["HetISq"])) & het_i2.isna(),
        "HETISQ_OUT_OF_RANGE": het_i2.notna() & ((het_i2 < 0) | (het_i2 > 100)),
    }

    core_valid = chr_valid & pos_valid & p_valid
    return {
        "chromosome": chromosome,
        "position": position,
        "p_value": p_value,
        "marker": marker,
        "chr_valid": chr_valid,
        "pos_valid": pos_valid,
        "p_valid": p_valid,
        "core_valid": core_valid,
        "flags": flags,
    }


def standardize(chunk: pd.DataFrame) -> pd.DataFrame:
    renamed = chunk.rename(
        columns={"chr": "CHR", "pos": "POS", "P.value": "P"}
    )
    first = ["CHR", "POS", "MarkerName", "P"]
    return renamed[first + [column for column in renamed.columns if column not in first]]


def append_tsv(frame: pd.DataFrame, path: Path, write_header: bool) -> None:
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        mode="w" if write_header else "a",
        header=write_header,
        lineterminator="\n",
    )


def first_pass(input_path: Path, paths: dict[str, Path], chunksize: int):
    raw_rows = 0
    retained_rows = 0
    missing_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    seen_positions: set[int] = set()
    duplicate_positions: set[int] = set()
    duplicate_id_extra = 0
    duplicate_position_extra = 0
    columns: list[str] | None = None
    clean_header = True

    for chunk in read_chunks(input_path, chunksize):
        if columns is None:
            columns = list(chunk.columns)
            missing_required = REQUIRED_COLUMNS - set(columns)
            if missing_required:
                raise ValueError(f"Missing required columns: {sorted(missing_required)}")

        result = evaluate(chunk)
        raw_rows += len(chunk)
        core_valid = result["core_valid"]
        retained_rows += int(core_valid.sum())

        for column in columns:
            missing_counts[column] += int(is_missing(chunk[column]).sum())
        for flag, mask in result["flags"].items():
            flag_counts[flag] += int(mask.sum())

        # Count additional occurrences while remembering which complete groups
        # must be flagged during the second pass.
        for variant_id in result["marker"]:
            if not variant_id or variant_id.lower() in NA_TOKENS:
                continue
            if variant_id in seen_ids:
                duplicate_ids.add(variant_id)
                duplicate_id_extra += 1
            else:
                seen_ids.add(variant_id)

        valid_position = result["chr_valid"] & result["pos_valid"]
        chromosome = result["chromosome"][valid_position].astype(np.int64)
        position = result["position"][valid_position].astype(np.int64)
        for chrom, pos in zip(chromosome, position):
            key = (int(chrom) << 32) | int(pos)
            if key in seen_positions:
                duplicate_positions.add(key)
                duplicate_position_extra += 1
            else:
                seen_positions.add(key)

        append_tsv(standardize(chunk).loc[core_valid], paths["clean"], clean_header)
        clean_header = False

    # The large seen sets are no longer needed before the second file pass.
    del seen_ids, seen_positions
    return {
        "columns": columns or [],
        "raw_rows": raw_rows,
        "retained_rows": retained_rows,
        "missing_counts": missing_counts,
        "flag_counts": flag_counts,
        "duplicate_ids": duplicate_ids,
        "duplicate_positions": duplicate_positions,
        "duplicate_id_extra": duplicate_id_extra,
        "duplicate_position_extra": duplicate_position_extra,
    }


def write_flagged(input_path: Path, paths: dict[str, Path], chunksize: int, state: dict) -> int:
    write_header = True
    flagged_count = 0

    for chunk in read_chunks(input_path, chunksize):
        result = evaluate(chunk)
        flags = dict(result["flags"])

        valid_position = result["chr_valid"] & result["pos_valid"]
        packed_position = pd.Series(-1, index=chunk.index, dtype=np.int64)
        packed_position.loc[valid_position] = (
            result["chromosome"][valid_position].astype(np.int64).to_numpy() << 32
        ) | result["position"][valid_position].astype(np.int64).to_numpy()
        flags["DUPLICATE_POSITION"] = packed_position.isin(state["duplicate_positions"])
        flags["DUPLICATE_VARIANT_ID"] = result["marker"].isin(state["duplicate_ids"])

        names = list(flags)
        matrix = np.column_stack([flags[name].to_numpy(dtype=bool) for name in names])
        any_flag = matrix.any(axis=1)
        if not any_flag.any():
            continue

        output = standardize(chunk).loc[any_flag].copy()
        output["QC_FLAGS"] = [
            ";".join(name for name, active in zip(names, row) if active) for row in matrix[any_flag]
        ]
        output["FILTERED"] = np.where(result["core_valid"].loc[any_flag], "NO", "YES")
        append_tsv(output, paths["flagged"], write_header)
        write_header = False
        flagged_count += len(output)

    if write_header:
        empty_columns = ["CHR", "POS", "MarkerName", "P"] + [
            column for column in state["columns"] if column not in {"chr", "pos", "MarkerName", "P.value"}
        ]
        pd.DataFrame(columns=empty_columns + ["QC_FLAGS", "FILTERED"]).to_csv(
            paths["flagged"], sep="\t", index=False, lineterminator="\n"
        )
    return flagged_count


def create_plots_and_p_statistics(clean_path: Path, paths: dict[str, Path], chunksize: int):
    p_parts: list[np.ndarray] = []
    plot_chr: list[np.ndarray] = []
    plot_pos: list[np.ndarray] = []
    plot_p: list[np.ndarray] = []
    max_position: Counter[int] = Counter()
    row_offset = 0

    for chunk in pd.read_csv(clean_path, sep="\t", usecols=["CHR", "POS", "P"], chunksize=chunksize):
        p = chunk["P"].to_numpy(dtype=float)
        chromosome = chunk["CHR"].to_numpy(dtype=np.int16)
        position = chunk["POS"].to_numpy(dtype=np.int64)
        p_parts.append(p)

        sequence = np.arange(row_offset, row_offset + len(chunk))
        select = (sequence % PLOT_STRIDE == 0) | (p < 1e-5)
        plot_chr.append(chromosome[select])
        plot_pos.append(position[select])
        plot_p.append(p[select])
        row_offset += len(chunk)
        for chrom in np.unique(chromosome):
            max_position[int(chrom)] = max(max_position[int(chrom)], int(position[chromosome == chrom].max()))

    all_p = np.concatenate(p_parts) if p_parts else np.array([], dtype=float)
    if all_p.size == 0:
        raise ValueError("No valid positive p-values remain for plotting")
    all_p.sort()

    median_p = float(np.median(all_p))
    lambda_gc = float(chi2.isf(median_p, 1) / chi2.ppf(0.5, 1))

    # QQ plot: exact extreme ranks plus an even sample of the remaining ranks.
    n = len(all_p)
    tail = np.arange(min(10_000, n), dtype=np.int64)
    broad = np.linspace(len(tail), n - 1, min(490_000, max(0, n - len(tail))), dtype=np.int64)
    indices = np.unique(np.concatenate([tail, broad]))
    expected = (indices + 1) / (n + 1)
    observed = np.maximum(all_p[indices], np.nextafter(0.0, 1.0))

    x = -np.log10(expected)
    y = -np.log10(observed)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=3, alpha=0.55)

    limit = float(x.max())
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set_xlim(0, limit * 1.05)
    ax.set_xlabel("Expected -log10(P)")
    ax.set_ylabel("Observed -log10(P)")
    plot_label = clean_path.stem.removesuffix(".cleaned")
    ax.set_title(f"NG00175 - {plot_label} QQ plot (lambda GC = {lambda_gc:.4f})")
    fig.tight_layout()
    fig.savefig(paths["qq"], dpi=180)
    plt.close(fig)

    chromosome = np.concatenate(plot_chr)
    position = np.concatenate(plot_pos)
    p = np.maximum(np.concatenate(plot_p), np.nextafter(0.0, 1.0))
    offsets: dict[int, int] = {}
    ticks: list[float] = []
    cursor = 0
    for chrom in sorted(max_position):
        offsets[chrom] = cursor
        ticks.append(cursor + max_position[chrom] / 2)
        cursor += max_position[chrom] + 1_000_000
    x = position + np.array([offsets[int(chrom)] for chrom in chromosome])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for index, chrom in enumerate(sorted(np.unique(chromosome))):
        select = chromosome == chrom
        ax.scatter(x[select], -np.log10(p[select]), s=3, alpha=0.7, color=("#315C8C" if index % 2 == 0 else "#73A2C6"))
    ax.axhline(-math.log10(5e-8), color="#B22222", linestyle="--", linewidth=1, label="5e-8")
    ax.axhline(-math.log10(1e-5), color="#D98C00", linestyle=":", linewidth=1, label="1e-5")
    ax.set_xticks(ticks, [str(chrom) for chrom in sorted(max_position)])
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(P)")
    ax.set_title(f"NG00175 - {plot_label} Manhattan plot")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(paths["manhattan"], dpi=180)
    plt.close(fig)

    return {
        "min_p": float(all_p[0]),
        "genome_wide_significant": int((all_p < 5e-8).sum()),
        "suggestive_only": int(((all_p >= 5e-8) & (all_p < 1e-5)).sum()),
        "lambda_gc": lambda_gc,
    }


def write_summaries(input_path: Path, paths: dict[str, Path], state: dict, p_stats: dict, flagged_count: int) -> None:
    raw_rows = state["raw_rows"]
    missingness = pd.DataFrame(
        [
            {
                "column": column,
                "missing_count": state["missing_counts"][column],
                "missing_percent": 100 * state["missing_counts"][column] / raw_rows if raw_rows else np.nan,
            }
            for column in state["columns"]
        ]
    )
    missingness.to_csv(paths["missingness"], index=False)

    flags = state["flag_counts"]
    invalid_position = flags["MISSING_POS"] + flags["MALFORMED_POS"] + flags["NONINTEGER_POS"] + flags["NONPOSITIVE_POS"]
    invalid_p = flags["MISSING_P"] + flags["MALFORMED_P"] + flags["P_ZERO"] + flags["P_OUT_OF_RANGE"]
    summary = {
        "input_filename": input_path.name,
        "genome_build": "GRCh38",
        "raw_variants": raw_rows,
        "retained_variants": state["retained_rows"],
        "filtered_variants": raw_rows - state["retained_rows"],
        "flagged_variants": flagged_count,
        "total_missing_values": sum(state["missing_counts"].values()),
        "invalid_chromosome": flags["MISSING_CHR"] + flags["INVALID_CHR"],
        "invalid_position": invalid_position,
        "invalid_p": invalid_p,
        "p_equals_zero": flags["P_ZERO"],
        "invalid_direction": flags["INVALID_DIRECTION"],
        "malformed_hetpval": flags["MALFORMED_HETPVAL"],
        "hetpval_out_of_range": flags["HETPVAL_OUT_OF_RANGE"],
        "malformed_hetisq": flags["MALFORMED_HETISQ"],
        "hetisq_out_of_range": flags["HETISQ_OUT_OF_RANGE"],
        "duplicate_variant_id_extra_rows": state["duplicate_id_extra"],
        "duplicate_chr_pos_extra_rows": state["duplicate_position_extra"],
        **p_stats,
    }
    pd.DataFrame([summary]).to_csv(paths["summary"], index=False)


def main() -> None:
    args = parse_args()
    input_path = args.input_file.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    paths = output_paths(input_path)
    state = first_pass(input_path, paths, args.chunksize)
    flagged_count = write_flagged(input_path, paths, args.chunksize, state)
    p_stats = create_plots_and_p_statistics(paths["clean"], paths, args.chunksize)
    write_summaries(input_path, paths, state, p_stats, flagged_count)

    print(f"QC complete: {input_path.name}")
    print(f"Cleaned file: {paths['clean']}")
    print(f"QC results: {paths['summary'].parent}")


if __name__ == "__main__":
    main()
