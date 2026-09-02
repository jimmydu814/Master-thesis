# Integrating AD GWAS and SEA-AD Single-Nucleus Multiomics to Study Regulatory Mechanisms of Alzheimer’s Disease Pathology

## Project Overview

This repository contains work from my Master's thesis on the genetic and regulatory mechanisms associated with Alzheimer's disease pathology. I am using NIAGADS NG00175 to study genetic associations with Braak/tau and diffuse amyloid-beta pathology, and NG00182 to study Alzheimer's disease case-control associations in Korean and Japanese populations.

GWAS summary statistics are first quality controlled and then analyzed with FUMA SNP2GENE. MAGMA is used for gene-level and gene-set association analysis, while GENE2FUNC is used to examine pathways, tissue enrichment, and other functional characteristics of the mapped genes. The long-term goal is to integrate these results with SEA-AD single-nucleus RNA-seq and ATAC-seq data to prioritize candidate genes and regulatory mechanisms associated with Alzheimer's disease pathology.

## Current Workflow

`NIAGADS GWAS → QC → FUMA SNP2GENE + MAGMA → GENE2FUNC → candidate gene and pathway integration → planned SEA-AD multi-omic integration`

Genome-wide significant and suggestive variants are also extracted separately for reporting and supporting analyses.

## Repository Structure

- `Part1_GWAS/scripts/` — Python scripts used for GWAS QC, variant extraction, and preparation of FUMA input files.
- `Part1_GWAS/metadata/` — Dataset documentation, source information, and analysis notes.
- `Part1_GWAS/qc_results/` — QC summaries, missingness results, Manhattan plots, and QQ plots.
- `Part1_GWAS/variant_hits/` — Genome-wide significant and suggestive variant results.
- `Part1_GWAS/fuma_result/` — Downloaded results and parameter files from FUMA analyses.

## Scripts

| Script | Description |
| ------ | ----------- |
| `qc_ng00175.py` | Runs chunked QC checks on NG00175 neuropathology GWAS summary statistics and generates cleaned data, QC summaries, Manhattan plots, and QQ plots. |
| `qc_ng00182.py` | Runs chunked QC checks on NG00182 case-control GWAS data, including allele, coordinate, duplicate, and beta/SE/Z consistency checks. |
| `extract_ng00175_hits.py` | Extracts genome-wide significant and suggestive variants from cleaned NG00175 summary statistics. |
| `extract_ng00182_hits.py` | Extracts genome-wide significant and suggestive variants from cleaned NG00182 summary statistics. |
| `prepare_FUMA_ng00175.py` | Converts cleaned NG00175 summary statistics into the three-column GRCh38 format used by FUMA SNP2GENE. |
| `prepare_fuma_ng00182.py` | Converts cleaned NG00182 summary statistics into FUMA's six-column GRCh38 format using ALT as the effect allele and REF as the other allele. |

## FUMA Analysis

FUMA analyses are performed through the online FUMA platform, so the repository contains downloaded results and parameter files rather than the analysis software itself.

For NG00175, SNP2GENE is used to identify genomic risk loci, candidate variants, and mapped genes through positional mapping, xQTL mapping, and chromatin interactions. MAGMA provides gene-level and gene-set association results, and GENE2FUNC is used to examine biological pathways, tissue enrichment, and functional patterns among the mapped genes.

These results will be combined into a candidate-gene and pathway evidence table before integration with the NG00182 and SEA-AD analyses.

## Data

Raw NIAGADS GWAS files, full cleaned summary statistics, FUMA input files, and large reference datasets are not included in this repository.

The datasets used in this project can be obtained directly from NIAGADS under the applicable access requirements:

NG00175: https://dss.niagads.org/datasets/ng00175/

NG00182: https://dss.niagads.org/datasets/ng00182/

This repository contains the analysis scripts, metadata, selected derived results, and documentation needed to describe the workflow.

## Current Status

GWAS QC and significant/suggestive variant extraction have been completed for the NG00175 Braak and diffuse amyloid-beta analyses and for the NG00182 GARD and METAL datasets.

For NG00175, FUMA SNP2GENE, MAGMA, and GENE2FUNC analyses have been completed. The next step is to integrate the gene-level and pathway-level evidence from the Braak and amyloid-beta analyses.

NG00182 will then be used to evaluate whether candidate genes also show evidence in Alzheimer's disease case-control genetics, followed by integration with SEA-AD single-nucleus RNA-seq and ATAC-seq data.