# SkinScope — Diet & Acne Evidence Notes

Last reviewed: 2026-10-06

This file backs up the `evidence` labels and `skin` text in the `NUTRIENTS`
table of `skinscope_app.py`. It also checks the diet factors (dairy, sugar /
glycemic index) used by `FACTORS`, `EXPERIMENTS`, `lifestyle_plan()` and
`routine_plan()`.

## Label scale

| Label | Meaning |
|---|---|
| **Moderate** | Supported by a meta-analysis of RCTs, or consistent meta-analyses with some trial support |
| **Limited to moderate** | Consistent associations in meta-analyses, with only small or low-quality trials |
| **Limited** | A few small RCTs, or consistent associations only |
| **Weak** | Isolated studies, conflicting results or mechanism only |

## What was checked and changed

| Factor | Where | Old label | New label | `skin` text |
|---|---|---|---|---|
| Glycemic index / sugar | `EXPERIMENTS`, `lifestyle_plan()`, "Go easier on" list | Moderate | Moderate (no change) | No change. Small RCTs support low-GL diets; a small 2025 observational meta-analysis was null. |
| Dairy | `EXPERIMENTS`, `lifestyle_plan()`, "Go easier on" list | Limited to moderate | Limited to moderate (no change) | No change. Observational only, strongest for skim milk; existing "test it yourself" wording fits. |
| Zinc | `NUTRIENTS` | Limited to moderate | Limited to moderate (no change) | **Updated.** The 2020 meta-analysis shows lower serum zinc and fewer inflamed papules with zinc, so "results are mixed" undersold it. Trials are small and zinc is weaker than standard treatment, so the label stays. |
| Omega-3 | `NUTRIENTS` | Limited | Limited (no change) | No change. Matches JAMA Dermatology 2023. |
| Vitamin D | `NUTRIENTS` | Limited and mixed | **Limited** | **Updated.** The association is consistent across meta-analyses, not mixed. Supplement trials are still few. |
| Vitamin A | `NUTRIENTS` | Weak for food; strong for prescription retinoids | No change | No change. The safety warning was already present. |
| Vitamin C | `NUTRIENTS` | Limited | **Weak** | **Updated.** No evidence that oral or dietary vitamin C affects acne; only topical forms have small trials. |
| Vitamin B12 | `NUTRIENTS` | Not an acne factor | No change | No change. Already says high doses can trigger breakouts (Kang 2015, Veraldi 2018). |
| Iron | `NUTRIENTS` | Not an acne factor | No change | **Updated.** "Iron is not linked to acne" was stronger than the evidence, which is a few small, conflicting case-control studies. |
| `FACTORS` | diary correlation | (no evidence labels) | — | Detection logic only. Nothing to change. |
| `routine_plan()` | skincare steps | — | — | No change. It holds no diet claims; "Vitamin C serum: may help dark marks fade" matches the topical evidence. |

---

## Sources by factor

### Glycemic index / glycemic load / sugar
- **Meixiong J, Ricco C, Vasavda C, Ho BK. "Diet and acne: A systematic review." *JAAD International*, 2022.**
  34 studies. High-GI and high-GL diets were associated with acne onset and severity, supported by RCTs.
  https://pubmed.ncbi.nlm.nih.gov/35373155/
- **Sakhaei & Mohsenpour. "Low Glycemic Load or Index Diet in Association with Acne Vulgaris: A Systematic Review and Meta-analysis." *Clinical and Cellular Biochemistry*.**
  A meta-analysis of 3 trials found lower-GI/GL diets reduced acne severity (Hedges' g −0.91, p = 0.007). Observational results were mixed.
  https://ccbjournal.ssu.ac.ir/article_6.html
- **Penso L et al. "Association Between Adult Acne and Dietary Behaviors: Findings From the NutriNet-Santé Prospective Cohort Study." *JAMA Dermatology*, 2020.**
  In about 24,000 adults, milk, sugary drinks and fatty/sugary foods were associated with current acne.
- **"The Role of Glycemic Load, Dairy, and Fatty Acids in Acne Disorders: A Systematic Review and Meta-Analysis." *Medicinus*, 2025.**
  5 studies, 716 participants. No significant pooled association for GL (SMD 0.09, 95% CI −0.30 to 0.49), GI or dairy. This is small and observational, so it adds uncertainty but does not overturn the trial data.
  https://ojs.uph.edu/index.php/MED/article/view/10771

### Dairy
- **Juhl CR et al. "Dairy Intake and Acne Vulgaris: A Systematic Review and Meta-Analysis of 78,529 Children, Adolescents, and Young Adults." *Nutrients*, 2018.** (pre-2020, still the largest)
  Compared with no intake: any dairy OR 1.25, any milk OR 1.28, low-fat/skim milk OR 1.32. Heterogeneity was high.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6115795/
- **Meixiong et al., *JAAD International*, 2022** (above).
  Dairy may be pro-acnegenic in some populations, mainly those eating a Western diet.
- **Penso et al., *JAMA Dermatology*, 2020** (above). Milk was associated with adult acne.
- **2025 *Medicinus* meta-analysis** (above). Dairy RR 1.04 (95% CI 0.25 to 4.25), not significant.
- **Mediterranean diet and acne vulgaris: a systematic review and meta-analysis, 2025.**
  https://pubmed.ncbi.nlm.nih.gov/41194132/

### Zinc
- **Yee BE et al. "Serum zinc levels and efficacy of zinc treatment in acne vulgaris: A systematic review and meta-analysis." *Dermatologic Therapy*, 2020.**
  Serum zinc was significantly lower in acne patients. Zinc treatment significantly reduced inflammatory papule counts, with no excess side effects.
  https://pubmed.ncbi.nlm.nih.gov/32860489/
- **Case-control study, *Journal of Cosmetic Dermatology*, 2024** (100 cases, 100 controls).
  Acne patients had lower serum zinc, selenium and vitamin D. Levels were lowest in grade-4 acne.
  https://healthday.com/dermatology-special/serum-zinc-selenium-vitamin-d-levels-lower-in-acne-vulgaris-patients

### Omega-3 fatty acids
- **Barbieri JS et al. Systematic review of oral nutraceuticals for acne. *JAMA Dermatology*, 2023.**
  42 RCTs, 3,346 participants, but only 15 were fair or good quality. Fair/good-quality trials suggest benefit from omega-3, vitamin D, vitamin B5, green tea extract and probiotics.
  https://pubmed.ncbi.nlm.nih.gov/37878272/
- **Polish review of acne supplementation, *Farmacja Polska*, 2023** (31 articles, 2010–2022).
  Omega-3 and probiotics eased the side effects of standard treatment. Omega-3 was no more effective than omega-6.
  https://doaj.org/article/b65b71e604964fb9b1fc359be7f924aa

### Vitamin D
- **Hasamoh Y, Thadanipon K, Juntongjin P. "Association between Vitamin D Level and Acne, and Correlation with Disease Severity: A Meta-Analysis." *Dermatology*, 2022 (online 2021).**
  13 studies, 1,362 patients and 1,081 controls. 25(OH)D was 9.02 ng/mL lower in acne patients; deficiency OR 2.97; levels fell as severity rose.
  https://karger.com/drm/article-pdf/238/3/404/3979851/000517514.pdf
- **"Correlation between Serum 25-Hydroxy Vitamin D levels and the severity of acne vulgaris: A systematic review." *Indian Journal of Dermatology*, 2022.**
  8 of 10 studies found lower vitamin D with worse acne.
  https://doaj.org/article/8dec36dab8774d75ae31d22925f9aa3d
- **RCT: "Role of vitamin D supplement adjunct to topical benzoyl peroxide in acne," 2024.**
  Weekly vitamin D2 40,000 IU alongside benzoyl peroxide prevented relapse of inflammatory lesions compared with placebo.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11216666/
- **Barbieri et al., *JAMA Dermatology*, 2023** (above). Vitamin D was among the supplements with fair/good-quality evidence of benefit.

### Vitamin A
- **"Oral Vitamin A for Acne Management: A Possible Substitute for Isotretinoin." *Journal of Drugs in Dermatology*, 2022.**
  9 studies, mostly uncontrolled, at 36,000 to 500,000 IU per day. Acne improved in 8 of them, but mucocutaneous side effects and headaches were common. High-dose vitamin A carries toxicity and teratogenicity risks.
  https://jddonline.com/articles/dermatology/S1545961622P0683X
- Case-control data (e.g. *Open Access Macedonian Journal of Medical Sciences*) show lower plasma vitamin A in acne patients.
  https://oamjms.eu/index.php/mjms/article/view/9991
- No 2020+ meta-analysis supports dietary vitamin A for acne. The proven retinoid route is prescription (topical retinoids, isotretinoin).

### Vitamin C
- **"Clinical Applications of Vitamin C in Dermatology: A Systematic Review." *Journal of Clinical and Aesthetic Dermatology*.**
  Topical 5% sodium ascorbyl phosphate reduced acne lesions in small trials (for example 48.8% fewer lesions at 8 weeks, n = 30). No trials support oral vitamin C for acne.
  https://jcadonline.com/wp-content/uploads/10-15_Clinical-Applications-of-Vitamin-C-in-Dermatology-A-Systematic-Review-.pdf
- **Barbieri et al., *JAMA Dermatology*, 2023** (above). Vitamin C was not among the supplements with fair/good-quality evidence.

### Vitamin B12
- **Kang D et al. "Vitamin B12 modulates the transcriptome of the skin microbiota in acne pathogenesis." *Science Translational Medicine*, 2015.** (mechanistic)
  B12 supplementation shifted *C. acnes* toward producing more inflammatory porphyrins, and 1 of 10 healthy volunteers developed acne.
  https://www.sciencenews.org/article/how-vitamin-b12-makes-pimples-pop
- **Veraldi S et al., 2018.** 5 cases of acneiform eruption after oral or IM vitamin B12, which cleared 3–6 weeks after stopping.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6049814/
- No 2020+ systematic review exists. The evidence points to B12 *aggravating* acne at high doses, not helping.

### Iron
- **Case-control study of postadolescent acne in women** (52 cases, 52 controls).
  No difference in serum iron, ferritin or TIBC, and no correlation with severity.
  https://openaccess.mku.edu.tr/items/5738577f-163e-4705-9114-1e7f29cdcf4d/full
- Another case-control study (100 cases, 100 controls) found iron and TIBC *higher* in acne patients, which conflicts with the first.
- No 2020+ systematic review or meta-analysis exists. Overall evidence is insufficient.

### Guideline context
- **Reynolds RV et al. "Guidelines of care for the management of acne vulgaris." *JAAD*, 2024 (American Academy of Dermatology).**
  The systematic review covered dietary interventions, but the recommendations focus on FDA-approved treatments. Diet should be presented as a supporting step, not a treatment.
  https://www.ovid.com/journals/jaade/fulltext/10.1016/j.jaad.2023.12.017

## Caveats
- Almost all diet–acne data are observational (case-control or cross-sectional) and rely on self-reported intake. Low nutrient levels in acne patients may be a *result* of inflammation rather than a cause.
- Penso 2020 and the Reynolds 2024 guideline are cited from memory and from summaries. Before shipping, check their links and exact wording against the full text.
- B12 and iron have no 2020+ systematic review; their entries rest on older or smaller studies.
