"""
PLCOm2012 — 6-year lung cancer risk from clinical / smoking history.

Tammemägi MC et al. "Selection criteria for lung-cancer screening."
N Engl J Med 2013;368:728-736. Applies to people who have ever smoked.
A 6-year risk >= 1.51% is the commonly used screening-eligibility threshold.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

THRESHOLD = 0.0151

RACE = {  # coefficient relative to White
    "white": 0.0,
    "black": 0.3944778,
    "hispanic": -0.7434744,
    "asian": -0.466585,
    "american_indian_alaska_native": 0.0,
    "native_hawaiian_pacific_islander": 1.027152,
}
EDUCATION = {  # PLCOm2012 education levels 1..6
    "less_than_high_school": 1, "high_school": 2, "post_high_school_training": 3,
    "some_college": 4, "college_graduate": 5, "postgraduate": 6,
}


@dataclass
class Patient:
    age: float                        # years
    smoking_status: str = "current"   # current | former | never
    cigarettes_per_day: float = 20
    smoking_years: float = 30
    years_since_quit: float = 0       # 0 for current smokers
    education: str = "high_school"
    bmi: float = 27
    copd: bool = False
    personal_cancer_history: bool = False
    family_lung_cancer: bool = False
    race: str = "white"
    sex: str = "unspecified"          # not used by PLCOm2012, kept for display


def plcom2012(p: Patient) -> dict:
    """Returns {'risk': 6-year probability, 'logit', 'eligible', 'applicable', 'note'}."""
    if p.smoking_status == "never":
        return {"risk": None, "logit": None, "eligible": False, "applicable": False,
                "note": "PLCOm2012 is defined for people who have ever smoked."}
    cpd = max(float(p.cigarettes_per_day), 1.0)
    quit = 0.0 if p.smoking_status == "current" else max(float(p.years_since_quit), 0.0)
    logit = (
        0.0778868 * (p.age - 62)
        - 0.0812744 * (EDUCATION.get(p.education, 2) - 4)
        - 0.0274194 * (p.bmi - 27)
        + 0.3553063 * bool(p.copd)
        + 0.4589971 * bool(p.personal_cancer_history)
        + 0.587185 * bool(p.family_lung_cancer)
        + 0.2597431 * (p.smoking_status == "current")
        - 1.822606 * (10.0 / cpd - 0.4021541613)
        + 0.0317321 * (p.smoking_years - 27)
        - 0.0308572 * (quit - 10)
        - 4.532506
        + RACE.get(p.race, 0.0)
    )
    risk = 1.0 / (1.0 + math.exp(-logit))
    note = ""
    if not 50 <= p.age <= 80:
        note = "Model was developed on people aged 55-74; outside ~50-80 it extrapolates."
    return {"risk": risk, "logit": logit, "eligible": risk >= THRESHOLD, "applicable": True,
            "threshold": THRESHOLD, "note": note, "inputs": asdict(p)}
