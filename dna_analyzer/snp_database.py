"""
Curated list of well-studied SNPs that consumer DNA chips usually include.

Alleles are given on the forward (+) strand of the GRCh37/GRCh38 reference,
which is how 23andMe, AncestryDNA, MyHeritage and FamilyTreeDNA report them.
Every SNP here has two alleles that are NOT complements of each other
(no A/T or C/G SNPs), so a file reported on the opposite strand can be
detected and flipped safely.

Each entry:
    rsid          SNP identifier
    gene          gene (or region) the SNP sits in or near
    trait         what it is associated with
    category      section of the report
    ref, alt      the two forward-strand alleles
    effect        the allele whose copies are counted (0, 1 or 2)
    evidence      "strong", "moderate" or "weak" (see README)
    sensitive     True for medical findings hidden unless --include-sensitive
    results       interpretation for 0, 1 and 2 copies of the effect allele
    note          extra context shown with every result
"""

EVIDENCE_LEVELS = {
    "strong": "Replicated in large studies; effect is well established.",
    "moderate": "Consistently replicated, but the effect is small or depends on other factors.",
    "weak": "Biological effect is known, but links to everyday traits are inconsistent.",
}

SNPS = [
    # ------------------------------------------------------------------ Nutrition
    {
        "rsid": "rs4988235",
        "gene": "MCM6 / LCT",
        "trait": "Lactose tolerance",
        "category": "Nutrition",
        "ref": "G", "alt": "A", "effect": "A",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "No copies of the lactase-persistence allele. Lactase production "
               "probably declines in adulthood, so lactose intolerance is likely.",
            1: "One copy of the lactase-persistence allele. The effect is dominant, "
               "so you are likely able to digest lactose as an adult.",
            2: "Two copies of the lactase-persistence allele. You are likely able "
               "to digest lactose as an adult.",
        },
        "note": "This variant explains lactase persistence mainly in people of "
                "European ancestry; other populations carry different variants "
                "that this marker does not test.",
    },
    {
        "rsid": "rs1801133",
        "gene": "MTHFR",
        "trait": "Folate metabolism (C677T)",
        "category": "Nutrition",
        "ref": "G", "alt": "A", "effect": "A",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "CC (677CC): typical MTHFR enzyme activity.",
            1: "CT (677CT): modestly reduced enzyme activity (about 65% of typical). "
               "Usually no practical effect.",
            2: "TT (677TT): enzyme activity about 30% of typical. Linked to higher "
               "homocysteine, mainly when folate intake is low; adequate folate "
               "usually brings levels back to normal.",
        },
        "note": "The effect on the enzyme is well established; claims that it "
                "causes most other health problems are not supported.",
    },
    {
        "rsid": "rs1801131",
        "gene": "MTHFR",
        "trait": "Folate metabolism (A1298C)",
        "category": "Nutrition",
        "ref": "T", "alt": "G", "effect": "G",
        "evidence": "moderate",
        "sensitive": False,
        "results": {
            0: "AA (1298AA): typical MTHFR enzyme activity.",
            1: "AC (1298AC): little or no measurable effect on its own.",
            2: "CC (1298CC): mildly reduced enzyme activity, smaller than the C677T effect.",
        },
        "note": "",
    },
    {
        "rsid": "rs762551",
        "gene": "CYP1A2",
        "trait": "Caffeine metabolism",
        "category": "Nutrition",
        "ref": "A", "alt": "C", "effect": "A",
        "evidence": "moderate",
        "sensitive": False,
        "results": {
            0: "CC: slower caffeine metaboliser.",
            1: "AC: slower caffeine metaboliser.",
            2: "AA: fast caffeine metaboliser (the enzyme is more inducible, "
               "especially in smokers).",
        },
        "note": "Smoking, medication and other genes also change caffeine clearance.",
    },
    {
        "rsid": "rs601338",
        "gene": "FUT2",
        "trait": "Secretor status (vitamin B12, norovirus)",
        "category": "Nutrition",
        "ref": "G", "alt": "A", "effect": "A",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "Secretor: blood-group antigens are released into saliva and gut.",
            1: "Secretor (one non-secretor copy, which is recessive).",
            2: "Non-secretor: blood-group antigens are not released into body "
               "fluids. Linked to strong resistance to common norovirus strains "
               "and to higher vitamin B12 levels.",
        },
        "note": "This is the main non-secretor variant in European and African "
                "populations; East Asian populations mostly carry a different one.",
    },
    # ------------------------------------------------------------- Alcohol & drugs
    {
        "rsid": "rs671",
        "gene": "ALDH2",
        "trait": "Alcohol flush reaction",
        "category": "Alcohol",
        "ref": "G", "alt": "A", "effect": "A",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "Typical ALDH2 activity; no genetic alcohol flush from this variant.",
            1: "One copy of ALDH2*2: much lower enzyme activity, so facial flushing "
               "after alcohol is likely. In carriers, drinking is linked to a "
               "markedly higher risk of oesophageal cancer.",
            2: "Two copies of ALDH2*2: almost no enzyme activity; strong flushing "
               "and very poor alcohol tolerance.",
        },
        "note": "Common in East Asian populations, rare elsewhere.",
    },
    {
        "rsid": "rs1229984",
        "gene": "ADH1B",
        "trait": "Alcohol metabolism speed",
        "category": "Alcohol",
        "ref": "C", "alt": "T", "effect": "T",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "Typical ADH1B activity.",
            1: "One copy of the fast ADH1B*2 allele: alcohol turns into acetaldehyde "
               "faster. Linked to a lower risk of alcohol dependence.",
            2: "Two copies of the fast ADH1B*2 allele: alcohol turns into "
               "acetaldehyde much faster. Linked to a lower risk of alcohol dependence.",
        },
        "note": "",
    },
    # ---------------------------------------------------------------- Fitness
    {
        "rsid": "rs1815739",
        "gene": "ACTN3",
        "trait": "Muscle fibre type (R577X)",
        "category": "Fitness",
        "ref": "C", "alt": "T", "effect": "T",
        "evidence": "moderate",
        "sensitive": False,
        "results": {
            0: "RR: makes alpha-actinin-3 in fast-twitch muscle. Slightly more "
               "common among elite sprint and power athletes.",
            1: "RX: makes alpha-actinin-3 in fast-twitch muscle.",
            2: "XX: makes no alpha-actinin-3. Common (about 18% of Europeans) and "
               "harmless; slightly less common among elite sprinters.",
        },
        "note": "The effect matters at elite level; training matters far more "
                "for everyone else.",
    },
    {
        "rsid": "rs4680",
        "gene": "COMT",
        "trait": "Dopamine breakdown (Val158Met)",
        "category": "Fitness",
        "ref": "G", "alt": "A", "effect": "A",
        "evidence": "weak",
        "sensitive": False,
        "results": {
            0: "Val/Val: higher COMT activity, faster dopamine breakdown.",
            1: "Val/Met: intermediate COMT activity.",
            2: "Met/Met: COMT activity roughly 3-4 times lower, slower dopamine breakdown.",
        },
        "note": "The enzyme effect is real; links to personality, stress or pain "
                "are inconsistent across studies.",
    },
    # ---------------------------------------------------------------- Traits
    {
        "rsid": "rs12913832",
        "gene": "HERC2 / OCA2",
        "trait": "Eye colour",
        "category": "Traits",
        "ref": "A", "alt": "G", "effect": "G",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "AA: brown eyes most likely.",
            1: "AG: brown or green/hazel eyes most likely; blue is less likely.",
            2: "GG: blue or other light eyes most likely.",
        },
        "note": "The strongest single eye-colour marker, but other genes also contribute.",
    },
    {
        "rsid": "rs17822931",
        "gene": "ABCC11",
        "trait": "Earwax type and body odour",
        "category": "Traits",
        "ref": "C", "alt": "T", "effect": "T",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "Wet earwax type.",
            1: "Wet earwax type (the dry type is recessive).",
            2: "Dry earwax type; also linked to less underarm odour.",
        },
        "note": "",
    },
    {
        "rsid": "rs72921001",
        "gene": "OR6A2 region",
        "trait": "Cilantro (coriander) tastes soapy",
        "category": "Traits",
        "ref": "C", "alt": "A", "effect": "A",
        "evidence": "weak",
        "sensitive": False,
        "results": {
            0: "Typical odds of finding cilantro soapy.",
            1: "Slightly higher odds of finding cilantro soapy.",
            2: "Higher odds of finding cilantro soapy.",
        },
        "note": "A real but small effect: most people with this variant still like cilantro.",
    },
    # ---------------------------------------------------------- Health risks
    {
        "rsid": "rs2187668",
        "gene": "HLA-DQA1 (DQ2.5)",
        "trait": "Coeliac disease genetic risk",
        "category": "Health",
        "ref": "C", "alt": "T", "effect": "T",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "Does not carry this HLA-DQ2.5 marker. HLA-DQ8 and other types are "
               "not tested here, so coeliac disease is not ruled out.",
            1: "Carries one copy of the HLA-DQ2.5 marker, the main genetic risk "
               "factor for coeliac disease. Many carriers never develop it.",
            2: "Carries two copies of the HLA-DQ2.5 marker, the main genetic risk "
               "factor for coeliac disease. Many carriers never develop it.",
        },
        "note": "Roughly 30% of people of European ancestry carry DQ2/DQ8, but "
                "only about 1% develop coeliac disease.",
    },
    {
        "rsid": "rs7903146",
        "gene": "TCF7L2",
        "trait": "Type 2 diabetes risk",
        "category": "Health",
        "ref": "C", "alt": "T", "effect": "T",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "CC: typical genetic risk from this variant.",
            1: "CT: about 1.4 times the odds of type 2 diabetes compared with CC.",
            2: "TT: about 2 times the odds of type 2 diabetes compared with CC.",
        },
        "note": "The strongest common type 2 diabetes variant, but weight, "
                "activity and diet matter much more.",
    },
    {
        "rsid": "rs10757274",
        "gene": "CDKN2B-AS1 (9p21)",
        "trait": "Coronary artery disease risk",
        "category": "Health",
        "ref": "A", "alt": "G", "effect": "G",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "AA: typical genetic risk from this region.",
            1: "AG: about 1.2-1.3 times the odds of coronary artery disease compared with AA.",
            2: "GG: about 1.5-1.6 times the odds of coronary artery disease compared with AA.",
        },
        "note": "Blood pressure, cholesterol, smoking and activity matter much more.",
    },
    {
        "rsid": "rs10455872",
        "gene": "LPA",
        "trait": "Lipoprotein(a) level",
        "category": "Health",
        "ref": "A", "alt": "G", "effect": "G",
        "evidence": "strong",
        "sensitive": False,
        "results": {
            0: "No copies of this high-Lp(a) marker.",
            1: "One copy: linked to raised lipoprotein(a) and higher heart disease "
               "risk. A one-off Lp(a) blood test gives the real number.",
            2: "Two copies: linked to raised lipoprotein(a) and higher heart disease "
               "risk. A one-off Lp(a) blood test gives the real number.",
        },
        "note": "Lp(a) is mostly set by genes; many other LPA variants also affect it.",
    },
    # ------------------------------------- Sensitive (shown only on request)
    {
        "rsid": "rs6025",
        "gene": "F5",
        "trait": "Factor V Leiden (blood clots)",
        "category": "Medical",
        "ref": "C", "alt": "T", "effect": "T",
        "evidence": "strong",
        "sensitive": True,
        "results": {
            0: "Factor V Leiden not detected.",
            1: "One copy of Factor V Leiden: about 3-8 times the typical risk of "
               "venous blood clots. Worth confirming with a clinical test.",
            2: "Two copies of Factor V Leiden: much higher risk of venous blood "
               "clots. Please confirm with a clinical test and talk to a doctor.",
        },
        "note": "Risk rises further with oestrogen-containing medication, surgery "
                "and long periods of immobility.",
    },
    {
        "rsid": "rs1799963",
        "gene": "F2",
        "trait": "Prothrombin G20210A (blood clots)",
        "category": "Medical",
        "ref": "G", "alt": "A", "effect": "A",
        "evidence": "strong",
        "sensitive": True,
        "results": {
            0: "Prothrombin G20210A not detected.",
            1: "One copy of prothrombin G20210A: about 2-3 times the typical risk "
               "of venous blood clots. Worth confirming with a clinical test.",
            2: "Two copies of prothrombin G20210A: much higher risk of venous "
               "blood clots. Please confirm with a clinical test and talk to a doctor.",
        },
        "note": "",
    },
    {
        "rsid": "rs1800562",
        "gene": "HFE",
        "trait": "Hereditary haemochromatosis (C282Y)",
        "category": "Medical",
        "ref": "G", "alt": "A", "effect": "A",
        "evidence": "strong",
        "sensitive": True,
        "results": {
            0: "HFE C282Y not detected.",
            1: "Carrier of one HFE C282Y copy. Carriers rarely develop iron overload.",
            2: "Two copies of HFE C282Y, the main cause of hereditary "
               "haemochromatosis. Many (not all) people with this develop iron "
               "overload; ferritin and transferrin saturation blood tests can check.",
        },
        "note": "",
    },
]

# APOE is read from two SNPs together, so it is handled separately in analyzer.py.
APOE = {
    "rsids": ("rs429358", "rs7412"),
    "gene": "APOE",
    "trait": "APOE type (Alzheimer's disease and cholesterol)",
    "category": "Medical",
    "evidence": "strong",
    "sensitive": True,
    # rs429358: T/C, C marks e4.   rs7412: C/T, T marks e2.
    "alleles": {"rs429358": ("T", "C"), "rs7412": ("C", "T")},
    "results": {
        "e3/e3": "The most common APOE type; typical Alzheimer's risk.",
        "e2/e3": "Slightly lower Alzheimer's risk than e3/e3.",
        "e2/e2": "Lower Alzheimer's risk; a small share of people with e2/e2 "
                 "develop a cholesterol disorder (type III hyperlipoproteinaemia).",
        "e3/e4": "One e4 copy: about 3 times the typical Alzheimer's risk. Most "
                 "people with one copy do not develop Alzheimer's disease.",
        "e2/e4": "One e2 and one e4 copy: roughly 2-3 times the typical Alzheimer's risk.",
        "e4/e4": "Two e4 copies: about 8-12 times the typical Alzheimer's risk. "
                 "Many people with e4/e4 never develop it; consider genetic "
                 "counselling before acting on this.",
    },
    "note": "APOE results can be upsetting and are not a diagnosis. Raw data "
            "cannot tell e2/e4 apart from the very rare e1/e3.",
}

CATEGORY_ORDER = ["Nutrition", "Alcohol", "Fitness", "Traits", "Health", "Medical"]
