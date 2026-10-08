"""Turn genotypes into findings using the SNP database."""

from dataclasses import dataclass, asdict

from snp_database import APOE, CATEGORY_ORDER, SNPS

COMPLEMENT = str.maketrans("ACGT", "TGCA")

# Finding.status values
FOUND = "found"
NOT_TESTED = "not tested"
NO_CALL = "no call"
UNEXPECTED = "unexpected genotype"


@dataclass
class Finding:
    rsid: str
    gene: str
    trait: str
    category: str
    evidence: str
    sensitive: bool
    status: str
    genotype: str = ""
    result: str = ""
    note: str = ""
    flipped_strand: bool = False

    def to_dict(self):
        return asdict(self)


def orient(genotype, ref, alt):
    """
    Return (alleles, flipped) with the genotype written on the forward strand,
    or (None, False) if it does not match the expected alleles at all.
    """
    alleles = list(genotype)
    if len(alleles) != 2:
        return None, False
    valid = {ref, alt}
    if set(alleles) <= valid:
        return alleles, False
    flipped = [a.translate(COMPLEMENT) for a in alleles]
    if set(flipped) <= valid:
        return flipped, True
    return None, False


def _base_finding(info, rsid, status, **extra):
    return Finding(
        rsid=rsid,
        gene=info["gene"],
        trait=info["trait"],
        category=info["category"],
        evidence=info["evidence"],
        sensitive=info["sensitive"],
        status=status,
        note=info.get("note", ""),
        **extra,
    )


def analyze_snp(snp, dna):
    rsid = snp["rsid"]
    if rsid in dna.no_call_rsids:
        return _base_finding(snp, rsid, NO_CALL)
    genotype = dna.genotypes.get(rsid)
    if genotype is None:
        return _base_finding(snp, rsid, NOT_TESTED)

    alleles, flipped = orient(genotype, snp["ref"], snp["alt"])
    if alleles is None:
        return _base_finding(
            snp, rsid, UNEXPECTED, genotype=genotype,
            result=f"Expected alleles {snp['ref']}/{snp['alt']} but found {genotype}.",
        )

    copies = alleles.count(snp["effect"])
    return _base_finding(
        snp, rsid, FOUND,
        genotype="".join(sorted(alleles)),
        result=snp["results"][copies],
        flipped_strand=flipped,
    )


def apoe_type(e4_markers, e2_markers):
    """APOE type from the count of rs429358-C (e4) and rs7412-T (e2) alleles."""
    table = {
        (0, 0): "e3/e3",
        (1, 0): "e3/e4",
        (2, 0): "e4/e4",
        (0, 1): "e2/e3",
        (0, 2): "e2/e2",
        (1, 1): "e2/e4",
    }
    return table.get((e4_markers, e2_markers))


def analyze_apoe(dna):
    rs_e4, rs_e2 = APOE["rsids"]
    label = f"{rs_e4} + {rs_e2}"

    oriented, flipped_any = {}, False
    for rsid in APOE["rsids"]:
        if rsid in dna.no_call_rsids:
            return _base_finding(APOE, label, NO_CALL)
        genotype = dna.genotypes.get(rsid)
        if genotype is None:
            return _base_finding(APOE, label, NOT_TESTED)
        ref, alt = APOE["alleles"][rsid]
        alleles, flipped = orient(genotype, ref, alt)
        if alleles is None:
            return _base_finding(
                APOE, label, UNEXPECTED, genotype=genotype,
                result=f"{rsid}: expected alleles {ref}/{alt} but found {genotype}.",
            )
        oriented[rsid] = alleles
        flipped_any = flipped_any or flipped

    e4 = oriented[rs_e4].count(APOE["alleles"][rs_e4][1])
    e2 = oriented[rs_e2].count(APOE["alleles"][rs_e2][1])
    genotype_text = f"{''.join(sorted(oriented[rs_e4]))} / {''.join(sorted(oriented[rs_e2]))}"
    kind = apoe_type(e4, e2)
    if kind is None:
        return _base_finding(
            APOE, label, UNEXPECTED, genotype=genotype_text,
            result="This combination needs the very rare e1 allele; the data may be wrong.",
        )
    return _base_finding(
        APOE, label, FOUND,
        genotype=f"{kind}  ({genotype_text})",
        result=APOE["results"][kind],
        flipped_strand=flipped_any,
    )


def wanted_rsids():
    """All rsids the analyzer looks at, for the parser's `wanted` filter."""
    return {snp["rsid"] for snp in SNPS} | set(APOE["rsids"])


def analyze(dna, include_sensitive=False):
    """
    Return (findings, hidden_count). Findings are sorted by report category.
    Sensitive medical findings are skipped unless include_sensitive is True.
    """
    findings = [analyze_snp(snp, dna) for snp in SNPS]
    findings.append(analyze_apoe(dna))

    hidden = sum(1 for f in findings if f.sensitive and not include_sensitive)
    if not include_sensitive:
        findings = [f for f in findings if not f.sensitive]

    findings.sort(key=lambda f: CATEGORY_ORDER.index(f.category))
    return findings, hidden
