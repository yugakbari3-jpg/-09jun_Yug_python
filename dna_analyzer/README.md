# Raw DNA Health Gene Analyzer

Reads the raw DNA file you can download from **23andMe, AncestryDNA,
MyHeritage or FamilyTreeDNA** and explains what about 20 well-studied gene
variants say about nutrition, alcohol, fitness, physical traits and health.

- Runs entirely on your computer; your DNA is never uploaded.
- Uses only the Python standard library (Python 3.9+), so there is nothing to install.
- Opens `.txt`, `.csv`, `.zip` and `.gz` downloads directly.
- Prints a text report and can save an HTML and JSON report.

> **For education only.** This is not a medical test. Consumer DNA chips can
> misread variants, and most traits depend on many genes plus lifestyle.
> Confirm anything health-related with a clinical test and a doctor or
> genetic counsellor.

## Usage

```bash
cd dna_analyzer

# Try it with the synthetic sample file
python dna_analyzer.py sample_data/sample_23andme.txt

# Your own file, with an HTML report you can open in a browser
python dna_analyzer.py path/to/your_raw_dna.zip --html report.html

# Also show medical results (see below)
python dna_analyzer.py path/to/your_raw_dna.txt --include-sensitive

# List every variant that is checked
python dna_analyzer.py --list
```

Options:

| Option | What it does |
| --- | --- |
| `--html PATH` | Save a styled HTML report (works in light and dark mode) |
| `--json PATH` | Save the results as JSON |
| `--include-sensitive` | Also show APOE (Alzheimer's), Factor V Leiden, prothrombin and HFE results |
| `--list` | List the variants in the database |

### Sensitive results

APOE (Alzheimer's risk), Factor V Leiden, prothrombin G20210A and HFE
(haemochromatosis) results are **hidden unless you ask for them** with
`--include-sensitive`, the same way genetic testing companies make you opt in.
Think about whether you want to know before turning them on.

## What is analysed

| Category | Variants |
| --- | --- |
| Nutrition | Lactose tolerance (LCT), MTHFR C677T and A1298C, caffeine metabolism (CYP1A2), secretor status (FUT2) |
| Alcohol | Alcohol flush (ALDH2), alcohol metabolism speed (ADH1B) |
| Fitness | Muscle fibre type (ACTN3), dopamine breakdown (COMT) |
| Traits | Eye colour (HERC2), earwax type (ABCC11), cilantro taste (OR6A2) |
| Health | Coeliac disease marker (HLA-DQ2.5), type 2 diabetes (TCF7L2), coronary artery disease (9p21), lipoprotein(a) (LPA) |
| Medical (opt-in) | APOE type, Factor V Leiden (F5), prothrombin (F2), haemochromatosis (HFE C282Y) |

Each result has an **evidence level**:

- **strong**: replicated in large studies; the effect is well established.
- **moderate**: consistently replicated, but the effect is small or depends on other factors.
- **weak**: the biological effect is known, but links to everyday traits are inconsistent.

## How it works

| File | Job |
| --- | --- |
| `dna_analyzer.py` | Command-line entry point |
| `dna_parser.py` | Detects the file format and reads genotypes (handles zip/gzip and no-calls) |
| `snp_database.py` | The curated variants, alleles and interpretations |
| `analyzer.py` | Counts the effect allele for each variant and works out the APOE type |
| `report.py` | Builds the text, HTML and JSON reports |

Genotypes are read on the forward DNA strand. If a file reports a variant on
the opposite strand (for example `TC` instead of `AG`), it is flipped
automatically. Every variant in the database was chosen so that this flip is
never ambiguous.

To add a variant, add an entry to `SNPS` in `snp_database.py` and give its
forward-strand alleles plus a result for 0, 1 and 2 copies of the effect allele.

## Tests

```bash
cd dna_analyzer
python -m unittest discover -s tests
```

`sample_data/sample_23andme.txt` is made up for testing and does not belong to a real person.
