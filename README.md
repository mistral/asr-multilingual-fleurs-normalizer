# FLEURS normalizer

Normalizer used before WER and CER computation during evaluation of Voxtral Transcribe 3
and API competitors on the [FLEURS](https://huggingface.co/datasets/google/fleurs) test
set.

## Usage

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

```python
from fleurs_normalizer import normalize

normalize("Hello, World!", "fr_fr")  # 'hello world'
```

Apply it to both the reference and the hypothesis. CER is used for `cmn_hans_cn`,
`ja_jp`, `ko_kr`, `th_th`, `my_mm` and `km_kh`, WER for the other languages.

## What it does

1. Unicode NFKC normalization.
2. Removes parentheses and square brackets (keeping their content), zero-width spaces
   and zero-width non-joiners.
3. Maps every Unicode decimal digit to ASCII.
4. Language-specific steps:
   - English: curly apostrophes become ASCII.
   - Arabic: removes tatweel, harakat and superscript alef.
   - Kannada and Odia: removes zero-width joiners.
   - Malayalam: folds chillu letters written with a zero-width joiner to their atomic
     code points.
   - Serbian: transliterates Cyrillic to Latin.
5. Whisper's text normalizer from `transformers` (the English one for English, the
   basic one otherwise).
