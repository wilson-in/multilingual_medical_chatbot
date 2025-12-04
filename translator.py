# translator.py
from transformers import MarianTokenizer, MarianMTModel, pipeline
import functools

# normalize some lang codes
NORMALIZE = {"zh-cn": "zh", "zh-tw": "zh", "iw": "he", "in": "id"}

_cache = {}

def _norm(code):
    return NORMALIZE.get(code, code)

def load_opus_pair(src, tgt):
    src_n = _norm(src); tgt_n = _norm(tgt)
    if src_n == tgt_n:
        return None
    key = f"{src_n}-{tgt_n}"
    if key in _cache:
        return _cache[key]
    model_name = f"Helsinki-NLP/opus-mt-{src_n}-{tgt_n}"
    try:
        tok = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        p = pipeline("translation", model=model, tokenizer=tok, device=-1)
        _cache[key] = p
        return p
    except Exception:
        _cache[key] = None
        return None

def translate_text(text, src, tgt):
    """Translate with simple fallback. Returns translated text or original text on failure."""
    if not text or src == tgt:
        return text
    # try direct pair
    p = load_opus_pair(src, tgt)
    if p:
        try:
            out = p(text, max_length=512)
            return out[0].get("translation_text", text)
        except Exception:
            return text
    # fallback: try src->en then en->tgt if needed externally
    if tgt == "en":
        p2 = load_opus_pair(src, "en")
        if p2:
            try:
                out = p2(text, max_length=512)
                return out[0].get("translation_text", text)
            except Exception:
                return text
    return text
