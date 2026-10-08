"""Run with:  python -m unittest discover -s tests   (from the dna_analyzer folder)"""

import gzip
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from analyzer import FOUND, NO_CALL, NOT_TESTED, UNEXPECTED, analyze, apoe_type, orient  # noqa: E402
from dna_analyzer import main  # noqa: E402
from dna_parser import DnaData, parse_raw_dna  # noqa: E402
from report import html_report, json_report, text_report  # noqa: E402

SAMPLE = HERE.parent / "sample_data" / "sample_23andme.txt"


def dna_with(**genotypes):
    return DnaData(source_format="test", genotypes=genotypes, total_markers=len(genotypes))


def finding(findings, rsid):
    return next(f for f in findings if f.rsid == rsid)


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = self.dir / name
        path.write_text(text)
        return path

    def test_23andme(self):
        dna = parse_raw_dna(SAMPLE)
        self.assertEqual(dna.source_format, "23andMe")
        self.assertEqual(dna.genotypes["rs4988235"], "AG")
        self.assertIn("rs10455872", dna.no_call_rsids)
        self.assertEqual(dna.no_calls, 2)

    def test_ancestry(self):
        path = self.write("ancestry.txt", (
            "#AncestryDNA raw data download\n"
            "rsid\tchromosome\tposition\tallele1\tallele2\n"
            "rs4988235\t2\t136608646\tA\tA\n"
            "rs671\t12\t112241766\t0\t0\n"
        ))
        dna = parse_raw_dna(path)
        self.assertEqual(dna.source_format, "AncestryDNA")
        self.assertEqual(dna.genotypes, {"rs4988235": "AA"})
        self.assertIn("rs671", dna.no_call_rsids)

    def test_myheritage_csv(self):
        path = self.write("myheritage.csv", (
            "# MyHeritage DNA raw data.\n"
            "RSID,CHROMOSOME,POSITION,RESULT\n"
            '"rs4988235","2","136608646","GG"\n'
            '"rs671","12","112241766","--"\n'
        ))
        dna = parse_raw_dna(path)
        self.assertEqual(dna.source_format, "MyHeritage")
        self.assertEqual(dna.genotypes, {"rs4988235": "GG"})

    def test_zip_and_gzip(self):
        text = SAMPLE.read_text()
        zipped = self.dir / "dna.zip"
        with zipfile.ZipFile(zipped, "w") as zf:
            zf.writestr("readme.txt", "hi")
            zf.writestr("genome.txt", text)
        gz = self.dir / "dna.txt.gz"
        gz.write_bytes(gzip.compress(text.encode()))
        for path in (zipped, gz):
            self.assertEqual(parse_raw_dna(path).genotypes["rs671"], "GG")

    def test_wanted_filter(self):
        dna = parse_raw_dna(SAMPLE, wanted={"rs671"})
        self.assertEqual(dna.genotypes, {"rs671": "GG"})
        self.assertEqual(dna.total_markers, 29)

    def test_bad_files(self):
        with self.assertRaises(FileNotFoundError):
            parse_raw_dna(self.dir / "missing.txt")
        with self.assertRaises(ValueError):
            parse_raw_dna(self.write("empty.txt", "# nothing here\n"))


class AnalyzerTests(unittest.TestCase):
    def test_orient_flips_opposite_strand(self):
        self.assertEqual(orient("AG", "G", "A"), (["A", "G"], False))
        self.assertEqual(orient("TC", "G", "A"), (["A", "G"], True))
        self.assertEqual(orient("AT", "G", "A"), (None, False))
        self.assertEqual(orient("A", "G", "A"), (None, False))

    def test_effect_allele_counts(self):
        for genotype, expected in (("GG", "No copies"), ("AG", "One copy"), ("AA", "Two copies")):
            findings, _ = analyze(dna_with(rs4988235=genotype))
            self.assertTrue(finding(findings, "rs4988235").result.startswith(expected))

    def test_opposite_strand_gives_same_result(self):
        plus, _ = analyze(dna_with(rs671="AG"))
        minus, _ = analyze(dna_with(rs671="TC"))
        self.assertEqual(finding(plus, "rs671").result, finding(minus, "rs671").result)
        self.assertTrue(finding(minus, "rs671").flipped_strand)

    def test_statuses(self):
        dna = dna_with(rs671="CA")  # impossible on either strand
        dna.no_call_rsids.add("rs4988235")
        findings, _ = analyze(dna)
        self.assertEqual(finding(findings, "rs671").status, UNEXPECTED)
        self.assertEqual(finding(findings, "rs4988235").status, NO_CALL)
        self.assertEqual(finding(findings, "rs1801133").status, NOT_TESTED)

    def test_apoe_types(self):
        cases = {
            (0, 0): "e3/e3", (1, 0): "e3/e4", (2, 0): "e4/e4",
            (0, 1): "e2/e3", (0, 2): "e2/e2", (1, 1): "e2/e4",
            (2, 1): None, (1, 2): None,
        }
        for counts, expected in cases.items():
            self.assertEqual(apoe_type(*counts), expected, counts)

    def test_apoe_from_genotypes(self):
        findings, _ = analyze(dna_with(rs429358="CT", rs7412="CC"), include_sensitive=True)
        apoe = finding(findings, "rs429358 + rs7412")
        self.assertEqual(apoe.status, FOUND)
        self.assertTrue(apoe.genotype.startswith("e3/e4"))

    def test_sensitive_hidden_by_default(self):
        findings, hidden = analyze(dna_with())
        self.assertEqual(hidden, 4)
        self.assertFalse(any(f.sensitive for f in findings))
        findings, hidden = analyze(dna_with(), include_sensitive=True)
        self.assertEqual(hidden, 0)
        self.assertEqual(sum(f.sensitive for f in findings), 4)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.dna = parse_raw_dna(SAMPLE)
        self.findings, self.hidden = analyze(self.dna)

    def test_text(self):
        text = text_report(self.dna, self.findings, self.hidden)
        self.assertIn("Lactose tolerance", text)
        self.assertIn("--include-sensitive", text)
        self.assertNotIn("APOE", text)

    def test_html_escapes_and_renders(self):
        page = html_report(self.dna, self.findings, self.hidden)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("Lactose tolerance", page)

    def test_json(self):
        data = json.loads(json_report(self.dna, self.findings, self.hidden))
        self.assertEqual(data["summary"]["format"], "23andMe")
        self.assertEqual(len(data["findings"]), len(self.findings))

    def test_cli_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            html_path, json_path = Path(tmp, "r.html"), Path(tmp, "r.json")
            code = main([str(SAMPLE), "--html", str(html_path), "--json", str(json_path)])
            self.assertEqual(code, 0)
            self.assertTrue(html_path.exists() and json_path.exists())
        self.assertEqual(main([str(SAMPLE) + ".missing"]), 1)


if __name__ == "__main__":
    unittest.main()
