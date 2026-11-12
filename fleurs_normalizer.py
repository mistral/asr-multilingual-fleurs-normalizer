"""Text normalizer used to score Voxtral Transcribe 3 transcriptions on FLEURS.

Requires `transformers`. Apply `normalize` to both the reference and the hypothesis,
then compute WER on the results, or CER for languages without word spacing
(Mandarin, Japanese, Thai, Burmese, Khmer) and for Korean.

    >>> normalize("Hello, World!", "fr_fr")
    'hello world'

The pipeline ends with Whisper's text normalizers. Note that Whisper's basic
normalizer replaces every combining mark (Unicode category M) with a space. In Indic
scripts this splits words at each vowel sign and virama, so WER on Bengali,
Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu is
computed over word fragments, following Whisper's convention.

For English, the first call downloads Whisper's British-to-American spelling mapping
from GitHub and caches it in `~/.cache/fleurs_normalizer/`.
"""

import hashlib
import json
import re
import unicodedata
from functools import cache
from pathlib import Path
from urllib.request import urlopen

from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)

_ENGLISH_SPELLING_MAPPING_URL = (
    "https://raw.githubusercontent.com/openai/whisper/v20250625/"
    "whisper/normalizers/english.json"
)
_ENGLISH_SPELLING_MAPPING_SHA256 = (
    "6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465"
)
_DOWNLOAD_TIMEOUT_SECONDS = 30
# Named after the hash, so that changing the pinned file never reuses a stale cache.
_ENGLISH_SPELLING_MAPPING_CACHE_PATH = (
    Path.home()
    / ".cache"
    / "fleurs_normalizer"
    / f"english_{_ENGLISH_SPELLING_MAPPING_SHA256[:16]}.json"
)

# FLEURS locale ids of the languages Voxtral Transcribe 3 supports.
FLEURS_LANGUAGES = frozenset(
    {
        "ar_eg", "bn_in", "bg_bg", "ca_es", "cmn_hans_cn", "cs_cz", "da_dk",
        "nl_nl", "en_us", "et_ee", "fi_fi", "fr_fr", "de_de", "el_gr", "gu_in",
        "hi_in", "hu_hu", "is_is", "id_id", "it_it", "ja_jp", "kn_in", "ko_kr",
        "lv_lv", "lt_lt", "mk_mk", "ms_my", "ml_in", "mr_in", "nb_no", "fa_ir",
        "pl_pl", "pt_br", "ro_ro", "ru_ru", "sk_sk", "sl_si", "es_419", "sv_se",
        "fil_ph", "te_in", "th_th", "tr_tr", "uk_ua", "vi_vn", "my_mm", "km_kh",
        "lb_lu", "mt_mt", "or_in", "pa_in", "sr_rs", "ta_in",
    }
)  # fmt: skip

# Locales whose normalized text has all spaces removed. Korean keeps its spaces.
_SPACE_STRIPPED_LANGUAGES = frozenset(
    {"cmn_hans_cn", "ja_jp", "th_th", "my_mm", "km_kh"}
)

# Characters deleted in every language:
# - Parenthesis and bracket delimiters, so that their content is kept: Whisper's
#   normalizers would otherwise delete whole parenthesized or bracketed spans.
#   Spans between angle brackets are still deleted by Whisper's normalizers.
# - Zero-width space and zero-width non-joiner, which are inconsistently present in
#   labels.
_DELETED_CHARACTERS = str.maketrans("", "", "()[]\u200b\u200c")

# Arabic tatweel, harakat/tashkeel (U+064B-U+065F) and superscript alef. Removing
# them is required: Whisper's normalizer would replace each haraka with a space.
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0640\u064B-\u065F\u0670]")
_ENGLISH_TYPOGRAPHIC_APOSTROPHES = str.maketrans({"‘": "'", "’": "'"})

# Malayalam chillu letters written as consonant + virama + ZWJ, folded to their
# atomic code points. The sources are multi-codepoint, so str.translate cannot be used.
_MALAYALAM_EQUIVALENT_SEQUENCES = (
    ("ണ്\u200d", "ൺ"),
    ("ന്\u200d", "ൻ"),
    ("ര്\u200d", "ർ"),
    ("ല്\u200d", "ൽ"),
    ("ള്\u200d", "ൾ"),
    ("ക്\u200d", "ൿ"),
    ("മ്\u200d", "ൔ"),
    ("യ്\u200d", "ൕ"),
    ("ഴ്\u200d", "ൖ"),
    ("ന്റ", "ൻ്റ"),
)

# Serbian Cyrillic to Gaj's Latin alphabet. The Latin direction avoids the ambiguity
# of Latin digraphs (e.g. "nj" can be "њ" or "нј").
_SERBIAN_CYRILLIC_TO_LATIN = str.maketrans(
    {
        **dict(
            zip(
                "АБВГДЂЕЖЗИЈКЛМНОПРСТЋУФХЦЧШ",
                "ABVGDĐEŽZIJKLMNOPRSTĆUFHCČŠ",
                strict=True,
            )
        ),
        **dict(
            zip(
                "абвгдђежзијклмнопрстћуфхцчш",
                "abvgdđežzijklmnoprstćufhcčš",
                strict=True,
            )
        ),
        "Љ": "Lj",
        "Њ": "Nj",
        "Џ": "Dž",
        "љ": "lj",
        "њ": "nj",
        "џ": "dž",
    }
)


def _english_spelling_mapping() -> dict[str, str]:
    cache_path = _ENGLISH_SPELLING_MAPPING_CACHE_PATH
    is_cached = cache_path.exists()
    if is_cached:
        content = cache_path.read_bytes()
    else:
        with urlopen(
            _ENGLISH_SPELLING_MAPPING_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            content = response.read()
    sha256 = hashlib.sha256(content).hexdigest()
    if sha256 != _ENGLISH_SPELLING_MAPPING_SHA256:
        source = cache_path if is_cached else _ENGLISH_SPELLING_MAPPING_URL
        raise ValueError(
            f"Unexpected SHA-256 {sha256} for {source}, "
            f"expected {_ENGLISH_SPELLING_MAPPING_SHA256}"
        )
    if not is_cached:
        # Write then rename, so that an interrupted write never leaves a partial file.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_bytes(content)
        temporary_path.replace(cache_path)
    return json.loads(content)


@cache
def _whisper_normalizer(language: str) -> EnglishTextNormalizer | BasicTextNormalizer:
    if language == "en_us":
        return EnglishTextNormalizer(_english_spelling_mapping())
    return BasicTextNormalizer()


def _normalize_decimal_digits(text: str) -> str:
    """Map every Unicode decimal digit (Devanagari, Arabic-Indic, ...) to ASCII."""
    return "".join(str(unicodedata.decimal(character, character)) for character in text)


def _apply_language_specific_norm(text: str, language: str) -> str:
    match language:
        case "en_us":
            # Whisper's English normalizer only expands contractions with ASCII
            # apostrophes.
            return text.translate(_ENGLISH_TYPOGRAPHIC_APOSTROPHES)
        case "ar_eg":
            return _ARABIC_DIACRITICS_RE.sub("", text)
        case "kn_in" | "or_in":
            # FLEURS labels omit optional zero-width joiners in Kannada and Odia.
            return text.replace("\u200d", "")
        case "ml_in":
            for source, canonical in _MALAYALAM_EQUIVALENT_SEQUENCES:
                text = text.replace(source, canonical)
            return text
        case "sr_rs":
            return text.translate(_SERBIAN_CYRILLIC_TO_LATIN)
        case _:
            # No language-specific step for the other locales.
            return text


def normalize(sentence: str, language: str) -> str:
    """Normalize a reference or hypothesis transcription for FLEURS scoring.

    Args:
        language: FLEURS locale id, one of `FLEURS_LANGUAGES` (e.g. "fr_fr").
    """
    if language not in FLEURS_LANGUAGES:
        raise ValueError(
            f"{language=} is not a supported FLEURS locale id: "
            f"{sorted(FLEURS_LANGUAGES)}"
        )
    text = unicodedata.normalize("NFKC", sentence)
    text = text.translate(_DELETED_CHARACTERS)
    text = _normalize_decimal_digits(text)
    text = _apply_language_specific_norm(text, language)
    text = _whisper_normalizer(language)(text).strip()
    if language in _SPACE_STRIPPED_LANGUAGES:
        text = text.replace(" ", "")
    return text
