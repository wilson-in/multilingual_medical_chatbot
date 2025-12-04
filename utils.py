# utils.py
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    def _detect_with_langdetect(text: str) -> str:
        text = (text or "").strip()
        if not text or len(text) < 3:
            return "en"
        return detect(text)
    _lang_detector = "langdetect"
except Exception:
    _detect_with_langdetect = None
    _lang_detector = None

# fallback to langid if langdetect unavailable
try:
    import langid
    def _detect_with_langid(text: str) -> str:
        text = (text or "").strip()
        if not text or len(text) < 3:
            return "en"
        return langid.classify(text)[0]
    _langid_available = True
except Exception:
    _detect_with_langid = None
    _langid_available = False

def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "en"
    if _detect_with_langdetect:
        try:
            return _detect_with_langdetect(text)
        except Exception:
            pass
    if _detect_with_langid:
        try:
            return _detect_with_langid(text)
        except Exception:
            pass
    return "en"


MEDICAL_KEYWORDS = {
    # Symptoms / conditions
    "pain", "ache", "hurt", "sore", "numb", "swelling", "fever", "cough", "cold", "flu",
    "infection", "rash", "itch", "allergy", "dizziness", "headache", "migraine", "nausea", "vomit",
    "diarrhea", "bleeding", "fracture", "broken", "sprain", "strain", "burn", "cut", "wound",
    # Body parts / systems
    "hand", "arm", "leg", "foot", "back", "stomach", "abdomen", "chest", "heart", "lung", "kidney",
    "liver", "skin", "eye", "ear", "throat", "neck",
    # Medical context words
    "symptom", "symptoms", "disease", "condition", "medicine", "medication", "tablet", "capsule",
    "antibiotic", "vaccine", "treatment", "therapy", "diagnosis", "doctor", "hospital", "clinic",
    "dose", "dosage", "prescription", "lab", "blood", "pressure", "diabetes", "hypertension", "covid",
}


def is_medical_query(text: str) -> bool:
    """Heuristic check to flag whether a query is medical-related."""
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in MEDICAL_KEYWORDS)
