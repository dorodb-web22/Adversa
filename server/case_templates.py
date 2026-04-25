"""Adversa — Case Templates (10 synthetic cases, no external dataset needed)."""
from __future__ import annotations
import random
from dataclasses import dataclass, field


@dataclass
class Evidence:
    id: str
    description: str
    side: str  # "prosecution" | "defense"
    strength: float
    emotional_impact: float
    admissible: bool
    authority_appeal: float = 0.5


@dataclass
class Witness:
    id: str
    name: str
    credibility: float
    side: str


@dataclass
class JurorProfile:
    type: str
    evidence_weight: float
    emotion_weight: float
    authority_weight: float
    consistency_bonus: float
    initial_sentiment: float


@dataclass
class CaseTemplate:
    case_id: str
    name: str
    charges: str
    defendant: str
    difficulty: str  # "easy" | "medium" | "hard"
    ground_truth: str  # "guilty" | "not_guilty"
    prosecution_evidence: list[Evidence]
    defense_evidence: list[Evidence]
    witnesses: list[Witness]
    jurors: list[JurorProfile]
    summary: str
    seed_variance: float = 0.12


DEFAULT_JURORS = [
    JurorProfile("analytical", 1.5, 0.3, 0.5, 0.5, 0.5),
    JurorProfile("empathetic", 0.5, 1.5, 0.7, 0.3, 0.5),
    JurorProfile("skeptical", 1.0, 0.2, 0.3, 2.0, 0.5),
]


def apply_seed(template: CaseTemplate, seed: int) -> CaseTemplate:
    """Apply seed variance to evidence strengths for episode diversity."""
    rng = random.Random(seed)
    v = template.seed_variance

    def jitter(val: float) -> float:
        return max(0.05, min(0.99, val + rng.uniform(-v, v)))

    new_pe = []
    for e in template.prosecution_evidence:
        ne = Evidence(e.id, e.description, e.side,
                      jitter(e.strength), jitter(e.emotional_impact),
                      e.admissible, jitter(e.authority_appeal))
        new_pe.append(ne)

    new_de = []
    for e in template.defense_evidence:
        ne = Evidence(e.id, e.description, e.side,
                      jitter(e.strength), jitter(e.emotional_impact),
                      e.admissible, jitter(e.authority_appeal))
        new_de.append(ne)

    new_jurors = []
    for j in template.jurors:
        nj = JurorProfile(j.type,
                          jitter(j.evidence_weight / 1.5) * 1.5,
                          jitter(j.emotion_weight / 1.5) * 1.5,
                          jitter(j.authority_weight),
                          jitter(j.consistency_bonus / 2.0) * 2.0,
                          j.initial_sentiment)
        new_jurors.append(nj)

    import copy
    t = copy.deepcopy(template)
    t.prosecution_evidence = new_pe
    t.defense_evidence = new_de
    t.jurors = new_jurors
    return t


# ── Case 1: The Teacher ──────────────────────────────────────────────────────
CASE_THE_TEACHER = CaseTemplate(
    case_id="C1",
    name="The Teacher",
    charges="Embezzlement of ₹4.2 lakhs from government school fund",
    defendant="Ananya Sharma, Government School Teacher, rural Maharashtra",
    difficulty="medium",
    ground_truth="not_guilty",
    summary=(
        "Ananya Sharma is accused of embezzling school funds. "
        "The defense has receipts and parent testimonies proving legitimate use. "
        "A key prosecution witness gave a coerced statement."
    ),
    prosecution_evidence=[
        Evidence("E1", "Bank transfer records showing ₹4.2L moved to personal account",
                 "prosecution", 0.9, 0.5, True, 0.6),
        Evidence("E2", "School principal's formal complaint letter",
                 "prosecution", 0.6, 0.4, True, 0.5),
        Evidence("E3", "Clerk statement alleging fund misuse (coerced under pressure)",
                 "prosecution", 0.7, 0.6, False, 0.4),
        Evidence("E4", "Pattern of 6 previous transfers over 18 months",
                 "prosecution", 0.4, 0.3, True, 0.3),
        Evidence("E5", "Missing audit paperwork for Q3",
                 "prosecution", 0.5, 0.2, True, 0.3),
    ],
    defense_evidence=[
        Evidence("E6", "47 itemized receipts matching transfer amounts exactly",
                 "defense", 0.95, 0.6, True, 0.8),
        Evidence("E7", "12 parent testimonies confirming supplies were received",
                 "defense", 0.8, 0.9, True, 0.7),
        Evidence("E8", "Whistleblower affidavit exposing clerk coercion by principal",
                 "defense", 0.85, 0.8, True, 0.7),
        Evidence("E9", "Teacher's salary records showing no lifestyle inflation",
                 "defense", 0.3, 0.5, True, 0.3),
        Evidence("E10", "School supply shortage records pre-transfer",
                 "defense", 0.6, 0.4, True, 0.5),
    ],
    witnesses=[
        Witness("W1", "School Principal Verma", 0.5, "prosecution"),
        Witness("W2", "Parent Association President", 0.85, "defense"),
        Witness("W3", "Clerk Ramesh (coerced)", 0.3, "prosecution"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 2: The Startup ──────────────────────────────────────────────────────
CASE_THE_STARTUP = CaseTemplate(
    case_id="C2",
    name="The Startup",
    charges="Trade secret theft and IP misappropriation",
    defendant="Vikram Nair, Co-founder, TechNova Pvt Ltd",
    difficulty="hard",
    ground_truth="guilty",
    summary=(
        "Vikram Nair left his employer to start a competing company using "
        "stolen proprietary algorithms. Digital forensics show data exfiltration."
    ),
    prosecution_evidence=[
        Evidence("E1", "Git history showing commits 48h before resignation",
                 "prosecution", 0.85, 0.4, True, 0.8),
        Evidence("E2", "USB drive access logs from secure server room",
                 "prosecution", 0.9, 0.3, True, 0.9),
        Evidence("E3", "Code similarity report: 87% match between products",
                 "prosecution", 0.95, 0.5, True, 0.9),
        Evidence("E4", "Email chain discussing 'taking the algorithm'",
                 "prosecution", 0.8, 0.6, True, 0.7),
        Evidence("E5", "NDA signed on joining explicitly covering algorithms",
                 "prosecution", 0.7, 0.2, True, 0.6),
        Evidence("E6", "VC pitch deck using proprietary benchmark results",
                 "prosecution", 0.75, 0.5, True, 0.7),
    ],
    defense_evidence=[
        Evidence("E7", "Expert testimony that algorithm is based on public paper",
                 "defense", 0.5, 0.3, True, 0.8),
        Evidence("E8", "Defendant's personal notebook predating employment",
                 "defense", 0.4, 0.4, False, 0.3),
        Evidence("E9", "HR records showing toxic work environment complaints",
                 "defense", 0.2, 0.8, True, 0.2),
        Evidence("E10", "Evidence that NDA clause was unenforceable per IT Act",
                 "defense", 0.55, 0.2, True, 0.6),
        Evidence("E11", "Open-source library implementing similar logic",
                 "defense", 0.6, 0.2, True, 0.6),
        Evidence("E12", "Character witnesses from previous employer praising ethics",
                 "defense", 0.3, 0.5, True, 0.3),
    ],
    witnesses=[
        Witness("W1", "CTO of victim company", 0.8, "prosecution"),
        Witness("W2", "Forensic expert", 0.9, "prosecution"),
        Witness("W3", "Academic algorithm researcher", 0.7, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 3: The Accident ─────────────────────────────────────────────────────
CASE_THE_ACCIDENT = CaseTemplate(
    case_id="C3",
    name="The Accident",
    charges="Medical negligence causing patient death",
    defendant="Dr. Priya Menon, Senior Surgeon, City Hospital",
    difficulty="medium",
    ground_truth="not_guilty",
    summary=(
        "Dr. Menon is accused of negligence during a high-risk surgery. "
        "Evidence shows the patient's condition was already critical and "
        "all protocols were followed."
    ),
    prosecution_evidence=[
        Evidence("E1", "Patient death certificate citing surgical complications",
                 "prosecution", 0.6, 0.9, True, 0.4),
        Evidence("E2", "Second surgeon's statement noting 'unusual technique'",
                 "prosecution", 0.65, 0.5, True, 0.5),
        Evidence("E3", "Hospital complaint record from patient's family",
                 "prosecution", 0.4, 0.8, True, 0.2),
        Evidence("E4", "Statistics showing hospital's above-average mortality rate",
                 "prosecution", 0.5, 0.6, True, 0.4),
    ],
    defense_evidence=[
        Evidence("E5", "Pre-op report showing patient at 95% mortality risk regardless",
                 "defense", 0.9, 0.5, True, 0.7),
        Evidence("E6", "Full surgical log showing protocol compliance at every step",
                 "defense", 0.85, 0.3, True, 0.8),
        Evidence("E7", "Expert testimony: surgery was the only viable option",
                 "defense", 0.8, 0.4, True, 0.9),
        Evidence("E8", "Dr. Menon's 98.2% success rate on comparable procedures",
                 "defense", 0.7, 0.4, True, 0.6),
    ],
    witnesses=[
        Witness("W1", "Patient's family representative", 0.4, "prosecution"),
        Witness("W2", "Medical board expert", 0.85, "defense"),
        Witness("W3", "Anesthesiologist present during surgery", 0.75, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 4: The Whistleblower ────────────────────────────────────────────────
CASE_THE_WHISTLEBLOWER = CaseTemplate(
    case_id="C4",
    name="The Whistleblower",
    charges="Unauthorized data disclosure and corporate espionage",
    defendant="Arjun Kapoor, Data Analyst, PharmaCorp India",
    difficulty="hard",
    ground_truth="not_guilty",
    summary=(
        "Arjun leaked internal documents showing illegal drug trial data "
        "manipulation. He is being prosecuted by the corporation. "
        "His actions were a protected whistleblower disclosure."
    ),
    prosecution_evidence=[
        Evidence("E1", "Proof of unauthorized document download",
                 "prosecution", 0.8, 0.3, True, 0.6),
        Evidence("E2", "Confidentiality agreement signed by defendant",
                 "prosecution", 0.75, 0.2, True, 0.7),
        Evidence("E3", "Media articles tracing leak to defendant",
                 "prosecution", 0.6, 0.5, True, 0.4),
        Evidence("E4", "Company's claim of ₹10Cr reputational damage",
                 "prosecution", 0.3, 0.7, True, 0.2),
        Evidence("E5", "IT logs of after-hours data access",
                 "prosecution", 0.7, 0.2, True, 0.6),
    ],
    defense_evidence=[
        Evidence("E6", "Internal documents proving falsified Phase 3 trial data",
                 "defense", 0.95, 0.7, True, 0.8),
        Evidence("E7", "Defendant's prior internal complaint (ignored for 6 months)",
                 "defense", 0.85, 0.8, True, 0.7),
        Evidence("E8", "Whistleblower Protection Act 2014 — applicable clause",
                 "defense", 0.9, 0.4, True, 0.9),
        Evidence("E9", "CDSCO investigation triggered by leak (ongoing)",
                 "defense", 0.7, 0.6, True, 0.6),
        Evidence("E10", "Defendant's refusal of ₹50L settlement offer",
                 "defense", 0.5, 0.8, True, 0.4),
    ],
    witnesses=[
        Witness("W1", "PharmaCorp Legal Head", 0.4, "prosecution"),
        Witness("W2", "Independent pharmacovigilance expert", 0.9, "defense"),
        Witness("W3", "Journalist who received documents", 0.6, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 5: The Contract ─────────────────────────────────────────────────────
CASE_THE_CONTRACT = CaseTemplate(
    case_id="C5",
    name="The Contract",
    charges="Breach of service agreement and fraud",
    defendant="Ravi Sharma, Director, BuildRight Construction",
    difficulty="easy",
    ground_truth="guilty",
    summary=(
        "BuildRight took a ₹2Cr advance for a commercial building project "
        "and delivered 40% of contracted work while pocketing the full payment."
    ),
    prosecution_evidence=[
        Evidence("E1", "Signed contract specifying 100% delivery by March 2025",
                 "prosecution", 0.95, 0.3, True, 0.8),
        Evidence("E2", "Independent survey: only 40% of work completed",
                 "prosecution", 0.9, 0.4, True, 0.9),
        Evidence("E3", "Bank records showing full ₹2Cr received",
                 "prosecution", 0.95, 0.2, True, 0.7),
    ],
    defense_evidence=[
        Evidence("E4", "Force majeure claim: COVID supply disruption",
                 "defense", 0.4, 0.5, True, 0.3),
        Evidence("E5", "Partial delivery receipts signed by complainant",
                 "defense", 0.5, 0.2, True, 0.4),
        Evidence("E6", "Ongoing litigation delaying remaining materials",
                 "defense", 0.3, 0.3, True, 0.3),
    ],
    witnesses=[
        Witness("W1", "Building inspector", 0.9, "prosecution"),
        Witness("W2", "Project manager for defendant", 0.4, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 6: The Hack ─────────────────────────────────────────────────────────
CASE_THE_HACK = CaseTemplate(
    case_id="C6",
    name="The Hack",
    charges="Unauthorized access to government computer systems (IT Act S.66)",
    defendant="Sneha Reddy, Security Researcher",
    difficulty="medium",
    ground_truth="guilty",
    summary=(
        "Sneha accessed the municipal water system SCADA network without "
        "authorization. She claims responsible disclosure, but accessed "
        "operational systems beyond the publicly-facing portal."
    ),
    prosecution_evidence=[
        Evidence("E1", "Server logs showing unauthorized SCADA access",
                 "prosecution", 0.9, 0.4, True, 0.8),
        Evidence("E2", "IP trace linking access to defendant's home network",
                 "prosecution", 0.85, 0.2, True, 0.8),
        Evidence("E3", "Evidence of data exfiltration (pump config files)",
                 "prosecution", 0.8, 0.5, True, 0.7),
        Evidence("E4", "Absence of bug bounty program or authorization",
                 "prosecution", 0.7, 0.2, True, 0.6),
        Evidence("E5", "Prior warning from municipal IT team",
                 "prosecution", 0.75, 0.3, True, 0.5),
    ],
    defense_evidence=[
        Evidence("E6", "Defendant's published responsible disclosure ethics policy",
                 "defense", 0.4, 0.5, True, 0.4),
        Evidence("E7", "Email to municipal team warning of vulnerability (pre-access)",
                 "defense", 0.6, 0.5, True, 0.5),
        Evidence("E8", "Security conference talk on water system vulnerabilities",
                 "defense", 0.35, 0.4, True, 0.5),
        Evidence("E9", "VPN logs showing limited scope of access",
                 "defense", 0.5, 0.2, True, 0.4),
        Evidence("E10", "Character reference from CERT-In official",
                 "defense", 0.3, 0.4, True, 0.4),
    ],
    witnesses=[
        Witness("W1", "Municipal CISO", 0.8, "prosecution"),
        Witness("W2", "Cybersecurity law professor", 0.7, "defense"),
        Witness("W3", "CERT-In analyst", 0.65, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 7: The Landlord ─────────────────────────────────────────────────────
CASE_THE_LANDLORD = CaseTemplate(
    case_id="C7",
    name="The Landlord",
    charges="Illegal tenant eviction and housing discrimination",
    defendant="Mahesh Gupta, Property Owner",
    difficulty="easy",
    ground_truth="guilty",
    summary=(
        "Mahesh Gupta forcibly evicted a Muslim tenant family mid-lease "
        "with no legal notice, changed locks, and damaged belongings."
    ),
    prosecution_evidence=[
        Evidence("E1", "Active lease agreement with 8 months remaining",
                 "prosecution", 0.95, 0.5, True, 0.8),
        Evidence("E2", "Locksmith invoice showing lock change without court order",
                 "prosecution", 0.9, 0.4, True, 0.8),
        Evidence("E3", "WhatsApp messages from landlord with discriminatory content",
                 "prosecution", 0.85, 0.9, True, 0.6),
    ],
    defense_evidence=[
        Evidence("E4", "Claim of 3-month rent default (disputed)",
                 "defense", 0.4, 0.3, True, 0.3),
        Evidence("E5", "Property maintenance costs exceeding rent",
                 "defense", 0.2, 0.3, True, 0.2),
        Evidence("E6", "Witness claiming tenant caused property damage",
                 "defense", 0.3, 0.4, False, 0.2),
    ],
    witnesses=[
        Witness("W1", "Tenant family representative", 0.8, "prosecution"),
        Witness("W2", "Neighbor who witnessed eviction", 0.75, "prosecution"),
        Witness("W3", "Landlord's property manager", 0.35, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 8: The Chemist ──────────────────────────────────────────────────────
CASE_THE_CHEMIST = CaseTemplate(
    case_id="C8",
    name="The Chemist",
    charges="Illegal industrial effluent dumping causing environmental damage",
    defendant="DyeChem Industries, represented by CEO Suresh Patil",
    difficulty="hard",
    ground_truth="not_guilty",
    summary=(
        "DyeChem is accused of dumping toxic dye into the Godavari river. "
        "The contamination actually originated from an upstream municipality "
        "— DyeChem had recently upgraded to a closed-loop system."
    ),
    prosecution_evidence=[
        Evidence("E1", "Water samples showing dye contamination downstream of DyeChem",
                 "prosecution", 0.75, 0.7, True, 0.5),
        Evidence("E2", "Farmers' testimonies about crop damage",
                 "prosecution", 0.55, 0.9, True, 0.3),
        Evidence("E3", "Historical violations record (10 years ago)",
                 "prosecution", 0.4, 0.5, True, 0.3),
        Evidence("E4", "Anonymous tip to pollution board naming DyeChem",
                 "prosecution", 0.3, 0.3, True, 0.2),
        Evidence("E5", "Timeline showing contamination started near plant location",
                 "prosecution", 0.6, 0.4, True, 0.5),
        Evidence("E6", "CPCB preliminary report (before full investigation)",
                 "prosecution", 0.5, 0.4, True, 0.4),
    ],
    defense_evidence=[
        Evidence("E7", "Third-party audit: DyeChem closed-loop system installed 2024",
                 "defense", 0.9, 0.3, True, 0.9),
        Evidence("E8", "Hydrological study showing upstream municipal discharge point",
                 "defense", 0.95, 0.4, True, 0.9),
        Evidence("E9", "CPCB final report exonerating DyeChem",
                 "defense", 0.85, 0.3, True, 0.8),
        Evidence("E10", "Same dye compound found in municipal treatment plant",
                 "defense", 0.9, 0.3, True, 0.8),
        Evidence("E11", "DyeChem's effluent treatment compliance certificates",
                 "defense", 0.8, 0.2, True, 0.7),
        Evidence("E12", "Municipality's expired treatment plant maintenance records",
                 "defense", 0.75, 0.5, True, 0.6),
    ],
    witnesses=[
        Witness("W1", "Affected farmer representative", 0.5, "prosecution"),
        Witness("W2", "Environmental forensics expert", 0.9, "defense"),
        Witness("W3", "CPCB chief investigator", 0.85, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 9: The Intern ───────────────────────────────────────────────────────
CASE_THE_INTERN = CaseTemplate(
    case_id="C9",
    name="The Intern",
    charges="Sexual harassment at the workplace (POSH Act violation)",
    defendant="Rohit Malhotra, Senior Manager, FinServ Corp",
    difficulty="medium",
    ground_truth="guilty",
    summary=(
        "Multiple interns reported a pattern of harassment by the manager. "
        "HR records show prior complaints that were suppressed. "
        "Digital evidence corroborates the complainants' accounts."
    ),
    prosecution_evidence=[
        Evidence("E1", "Complainant's sworn affidavit with specific incidents",
                 "prosecution", 0.8, 0.9, True, 0.6),
        Evidence("E2", "WhatsApp messages with inappropriate content",
                 "prosecution", 0.9, 0.7, True, 0.7),
        Evidence("E3", "Two corroborating intern testimonies",
                 "prosecution", 0.75, 0.8, True, 0.6),
        Evidence("E4", "HR records showing prior complaint buried by manager's superior",
                 "prosecution", 0.7, 0.7, True, 0.6),
    ],
    defense_evidence=[
        Evidence("E5", "Defendant's claim of professional-only communication",
                 "defense", 0.2, 0.3, True, 0.2),
        Evidence("E6", "Performance review praising complainant (cited as motive)",
                 "defense", 0.15, 0.4, True, 0.1),
        Evidence("E7", "Alibi for one specific incident date",
                 "defense", 0.45, 0.2, True, 0.4),
        Evidence("E8", "Character witnesses from prior colleagues",
                 "defense", 0.25, 0.4, True, 0.2),
    ],
    witnesses=[
        Witness("W1", "Primary complainant", 0.8, "prosecution"),
        Witness("W2", "HR director who buried complaint", 0.3, "prosecution"),
        Witness("W3", "Defendant's prior colleague", 0.3, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)

# ── Case 10: The Mayor ───────────────────────────────────────────────────────
CASE_THE_MAYOR = CaseTemplate(
    case_id="C10",
    name="The Mayor",
    charges="Corruption — quid pro quo bribery for municipal contracts",
    defendant="Dinesh Rao, Mayor, Tier-2 City Municipal Corporation",
    difficulty="hard",
    ground_truth="not_guilty",
    summary=(
        "Mayor Rao is accused of taking bribes for road contracts. "
        "The accusation originates from a political rival who fabricated "
        "evidence. The 'bribe' was a legal campaign donation properly disclosed."
    ),
    prosecution_evidence=[
        Evidence("E1", "Cash transfer of ₹25L traced to contractor",
                 "prosecution", 0.7, 0.6, True, 0.5),
        Evidence("E2", "Contractor awarded 3 consecutive tenders",
                 "prosecution", 0.6, 0.4, True, 0.4),
        Evidence("E3", "Recorded phone call (authenticity disputed)",
                 "prosecution", 0.5, 0.7, False, 0.4),
        Evidence("E4", "Whistleblower (later found to be political operative)",
                 "prosecution", 0.4, 0.5, True, 0.3),
        Evidence("E5", "Mayor's unexplained property purchase",
                 "prosecution", 0.55, 0.6, True, 0.4),
        Evidence("E6", "Timeline: contract awarded 2 weeks after transfer",
                 "prosecution", 0.65, 0.4, True, 0.5),
        Evidence("E7", "Opposition party's demand for investigation",
                 "prosecution", 0.2, 0.5, True, 0.1),
    ],
    defense_evidence=[
        Evidence("E8", "Election Commission donation disclosure form (₹25L declared)",
                 "defense", 0.95, 0.4, True, 0.8),
        Evidence("E9", "Contractor won tenders via lowest-bid public process",
                 "defense", 0.9, 0.3, True, 0.9),
        Evidence("E10", "Audio forensics: recording is edited composite",
                 "defense", 0.85, 0.5, True, 0.8),
        Evidence("E11", "Whistleblower's affiliation with opposition (conflicts of interest)",
                 "defense", 0.8, 0.6, True, 0.6),
        Evidence("E12", "Property purchased via bank loan pre-dating tenure",
                 "defense", 0.85, 0.3, True, 0.7),
        Evidence("E13", "All tender minutes public record — no Mayor sign-off required",
                 "defense", 0.75, 0.2, True, 0.8),
        Evidence("E14", "CBI preliminary closure report (insufficient evidence)",
                 "defense", 0.7, 0.3, True, 0.7),
    ],
    witnesses=[
        Witness("W1", "Opposition party operative (whistleblower)", 0.2, "prosecution"),
        Witness("W2", "Forensic audio expert", 0.85, "defense"),
        Witness("W3", "Election Commission officer", 0.9, "defense"),
    ],
    jurors=DEFAULT_JURORS,
)


# ── Registry ─────────────────────────────────────────────────────────────────
ALL_CASES: dict[str, CaseTemplate] = {
    "C1": CASE_THE_TEACHER,
    "C2": CASE_THE_STARTUP,
    "C3": CASE_THE_ACCIDENT,
    "C4": CASE_THE_WHISTLEBLOWER,
    "C5": CASE_THE_CONTRACT,
    "C6": CASE_THE_HACK,
    "C7": CASE_THE_LANDLORD,
    "C8": CASE_THE_CHEMIST,
    "C9": CASE_THE_INTERN,
    "C10": CASE_THE_MAYOR,
}


def get_case(case_id: str, seed: int = 0) -> CaseTemplate:
    """Return a seeded case template. Raises KeyError for unknown case_id."""
    if case_id not in ALL_CASES:
        raise KeyError(f"Unknown case_id '{case_id}'. Valid: {list(ALL_CASES)}")
    return apply_seed(ALL_CASES[case_id], seed)
