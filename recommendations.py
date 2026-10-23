"""Diet and medication suggestions for each predicted disease."""
from pathlib import Path

DIETS_DIR = Path(__file__).parent / "diets"

# The dataset's labels contain a few misspellings; map them to the clean names used below
LABEL_ALIASES = {
    "(vertigo) paroymsal  positional vertigo": "(vertigo) paroxysmal positional vertigo",
    "dimorphic hemmorhoids(piles)": "dimorphic hemorrhoids (piles)",
    "osteoarthristis": "osteoarthritis",
    "peptic ulcer diseae": "peptic ulcer disease",
}

DRUGS = {
    'fungal infection': ['Antifungal Cream', 'Fluconazole', 'Itraconazole'],
    'allergy': ['Antihistamines', 'Corticosteroids', 'Epinephrine'],
    'gerd': ['Proton Pump Inhibitors', 'H2 Blockers', 'Antacids'],
    'chronic cholestasis': ['Ursodeoxycholic Acid'],
    'drug reaction': ['Antihistamines', 'Corticosteroids', 'Epinephrine'],
    'peptic ulcer disease': ['Proton Pump Inhibitors', 'H2 Blockers', 'Antacids'],
    'aids': ['Antiretroviral Therapy (ART)'],
    'diabetes': ['Metformin', 'Insulin', 'Sitagliptin'],
    'gastroenteritis': ['Oral Rehydration Solution', 'Antibiotics (in bacterial gastroenteritis)'],
    'bronchial asthma': ['Bronchodilators', 'Inhaled Corticosteroids', 'Leukotriene Modifiers'],
    'hypertension': ['ACE Inhibitors', 'Beta-blockers', 'Calcium Channel Blockers'],
    'migraine': ['Triptans', 'NSAIDs', 'Beta-blockers'],
    'cervical spondylosis': ['Pain Relievers', 'Muscle Relaxants', 'Physical Therapy'],
    'paralysis (brain hemorrhage)': ['Rehabilitation', 'Physical Therapy', 'Occupational Therapy'],
    'jaundice': ['Ursodeoxycholic Acid', 'Vitamin K', 'Liver Supportive Medications'],
    'malaria': ['Chloroquine', 'Artemether', 'Lumefantrine'],
    'chicken pox': ['Acyclovir', 'Antihistamines', 'Calamine Lotion'],
    'dengue': ['Fluid Replacement Therapy', 'Pain Relievers', 'Monitoring'],
    'typhoid': ['Antibiotics (e.g., Ciprofloxacin, Ceftriaxone)'],
    'hepatitis a': ['Hepatitis A Vaccine', 'Supportive Care'],
    'hepatitis b': ['Hepatitis B Vaccine', 'Antiviral Medications (e.g., Tenofovir)'],
    'hepatitis c': ['Antiviral Medications (e.g., Sofosbuvir, Ledipasvir)'],
    'hepatitis d': ['Hepatitis D Vaccine', 'Antiviral Medications'],
    'hepatitis e': ['Supportive Care', 'Avoidance of Alcohol'],
    'alcoholic hepatitis': ['Corticosteroids', 'Pentoxifylline', 'Nutritional Support'],
    'tuberculosis': ['Isoniazid', 'Rifampin', 'Pyrazinamide', 'Ethambutol'],
    'common cold': ['Pain Relievers', 'Decongestants', 'Antihistamines'],
    'pneumonia': ['Antibiotics (e.g., Amoxicillin, Azithromycin, Levofloxacin)'],
    'dimorphic hemorrhoids (piles)': ['Pain Relievers', 'Stool Softeners', 'Dietary Changes'],
    'heart attack': ['Aspirin', 'Beta-blockers', 'ACE Inhibitors'],
    'varicose veins': ['Compression Stockings', 'Venotonics'],
    'hypothyroidism': ['Levothyroxine'],
    'hyperthyroidism': ['Beta-blockers', 'Antithyroid Drugs (e.g., Methimazole, Propylthiouracil)'],
    'hypoglycemia': ['Oral Glucose', 'Glucagon', 'Dextrose'],
    'osteoarthritis': ['Pain Relievers', 'Topical Treatments', 'Physical Therapy'],
    'arthritis': ['NSAIDs', 'DMARDs', 'Biologic Agents'],
    '(vertigo) paroxysmal positional vertigo': ['Epley Maneuver', 'Vestibular Rehabilitation'],
    'acne': ['Topical Treatments (e.g., Benzoyl Peroxide, Retinoids)', 'Oral Medications (e.g., Antibiotics, Isotretinoin)'],
    'urinary tract infection': ['Antibiotics (e.g., Trimethoprim-Sulfamethoxazole, Ciprofloxacin)'],
    'psoriasis': ['Topical Treatments (e.g., Corticosteroids, Vitamin D Analogues)', 'Systemic Treatments (e.g., Methotrexate, Biologic Agents)'],
    'impetigo': ['Topical Antibiotics (e.g., Mupirocin)', 'Oral Antibiotics (e.g., Cephalexin, Dicloxacillin)'],
}


def display_name(label):
    return LABEL_ALIASES.get(label, label)


def get_drugs(label):
    return DRUGS.get(display_name(label), [])


def get_diet(label):
    path = DIETS_DIR / f"{label}.txt"
    if not path.exists():
        path = DIETS_DIR / "others.txt"
    return path.read_text(encoding="utf-8", errors="replace").strip()
