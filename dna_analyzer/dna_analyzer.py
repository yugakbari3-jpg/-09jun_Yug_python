"""
Raw DNA Health Gene Analyzer

Reads a raw DNA download from 23andMe, AncestryDNA, MyHeritage or
FamilyTreeDNA and reports what well-studied gene variants say about
nutrition, alcohol, fitness, traits and health. Everything runs locally.

Usage:
    python dna_analyzer.py my_dna.txt
    python dna_analyzer.py my_dna.zip --html report.html --json report.json
    python dna_analyzer.py my_dna.txt --include-sensitive
    python dna_analyzer.py --list
"""

import argparse
import sys
from pathlib import Path

from analyzer import analyze, wanted_rsids
from dna_parser import parse_raw_dna
from report import html_report, json_report, text_report
from snp_database import APOE, SNPS


def list_snps():
    print(f"{'rsid':<20} {'gene':<20} {'evidence':<9} trait")
    for snp in SNPS:
        flag = "  (sensitive)" if snp["sensitive"] else ""
        print(f"{snp['rsid']:<20} {snp['gene']:<20} {snp['evidence']:<9} {snp['trait']}{flag}")
    print(f"{' + '.join(APOE['rsids']):<20} {APOE['gene']:<20} {APOE['evidence']:<9} "
          f"{APOE['trait']}  (sensitive)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyse a raw DNA file for well-studied health and trait gene variants."
    )
    parser.add_argument("file", nargs="?", help="raw DNA file (.txt, .csv, .zip or .gz)")
    parser.add_argument("--html", metavar="PATH", help="also save an HTML report")
    parser.add_argument("--json", metavar="PATH", help="also save a JSON report")
    parser.add_argument(
        "--include-sensitive", action="store_true",
        help="show medical results (APOE/Alzheimer's, blood clotting, haemochromatosis)",
    )
    parser.add_argument("--list", action="store_true", help="list the variants that are analysed")
    args = parser.parse_args(argv)

    if args.list:
        list_snps()
        return 0
    if not args.file:
        parser.error("give a raw DNA file, or use --list")

    try:
        dna = parse_raw_dna(args.file, wanted=wanted_rsids())
    except (OSError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    findings, hidden = analyze(dna, include_sensitive=args.include_sensitive)
    print(text_report(dna, findings, hidden))

    if args.html:
        Path(args.html).write_text(html_report(dna, findings, hidden), encoding="utf-8")
        print(f"\nHTML report saved to {args.html}")
    if args.json:
        Path(args.json).write_text(json_report(dna, findings, hidden), encoding="utf-8")
        print(f"JSON report saved to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
