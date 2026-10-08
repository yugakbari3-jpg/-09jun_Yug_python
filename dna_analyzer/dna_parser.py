"""
Read raw DNA files from 23andMe, AncestryDNA, MyHeritage and FamilyTreeDNA.

Supported layouts (all may be plain text, .zip or .gz):
    23andMe        tab separated:   rsid  chromosome  position  genotype
    AncestryDNA    tab separated:   rsid  chromosome  position  allele1  allele2
    MyHeritage /   comma separated: "RSID","CHROMOSOME","POSITION","RESULT"
    FamilyTreeDNA
"""

import csv
import gzip
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

NO_CALLS = {"", "-", "--", "0", "00", "NC", "??"}


@dataclass
class DnaData:
    """Everything the analyzer needs from a raw DNA file."""

    source_format: str
    genotypes: dict = field(default_factory=dict)  # rsid -> "AG"; no-calls left out
    no_call_rsids: set = field(default_factory=set)
    total_markers: int = 0
    no_calls: int = 0
    chromosomes: Counter = field(default_factory=Counter)

    @property
    def call_rate(self) -> float:
        if not self.total_markers:
            return 0.0
        return (self.total_markers - self.no_calls) / self.total_markers


def read_lines(path):
    """Return the text lines of a plain, .gz or .zip raw DNA file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = [
                n for n in zf.namelist()
                if not n.endswith("/") and not n.startswith("__MACOSX")
            ]
            if not names:
                raise ValueError(f"{path.name} is an empty zip file")
            # The raw data is always the biggest file in the download.
            name = max(names, key=lambda n: zf.getinfo(n).file_size)
            data = zf.read(name)
    else:
        with open(path, "rb") as f:
            data = f.read()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)

    return data.decode("utf-8", errors="replace").splitlines()


def _detect_format(comments, header):
    text = " ".join(comments).lower()
    if "23andme" in text:
        return "23andMe"
    if "ancestrydna" in text or (header and "allele1" in header):
        return "AncestryDNA"
    if "myheritage" in text:
        return "MyHeritage"
    if "familytreedna" in text or "family tree dna" in text:
        return "FamilyTreeDNA"
    return "Unknown"


def _split(line):
    if "\t" in line:
        return [part.strip().strip('"') for part in line.split("\t")]
    if "," in line:
        return [part.strip() for part in next(csv.reader([line]))]
    return line.split()


def parse_raw_dna(path, wanted=None):
    """
    Parse a raw DNA file.

    `wanted` is an optional set of rsids to keep. Raw files hold ~600,000
    markers, so keeping only the ones we analyse saves a lot of memory.
    Counts in the summary always cover the whole file.
    """
    comments, header = [], None
    result = DnaData(source_format="Unknown")

    for line in read_lines(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue

        fields = _split(line)
        if fields and fields[0].lower() in ("rsid", "rs id", "snp"):
            header = [f.lower() for f in fields]
            continue
        if len(fields) < 4:
            continue

        rsid, chromosome = fields[0], fields[1]
        if len(fields) >= 5 and (header is None or "allele1" in header):
            genotype = fields[3] + fields[4]
        else:
            genotype = fields[3]
        genotype = genotype.upper().replace(" ", "")

        result.total_markers += 1
        result.chromosomes[chromosome] += 1
        if genotype in NO_CALLS or "-" in genotype or "0" in genotype:
            result.no_calls += 1
            if wanted is None or rsid in wanted:
                result.no_call_rsids.add(rsid)
            continue
        if wanted is None or rsid in wanted:
            result.genotypes[rsid] = genotype

    if result.total_markers == 0:
        raise ValueError(
            f"No genotype rows found in {Path(path).name}. Is this a raw DNA "
            "file from 23andMe, AncestryDNA, MyHeritage or FamilyTreeDNA?"
        )

    result.source_format = _detect_format(comments, header)
    return result
