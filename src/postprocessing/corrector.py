# src/postprocessing/corrector.py
"""
Post-processing for learner speech transcriptions.

Handles:
- Filler word removal (um, uh, eh — very common in L2 speech)
- Lowercasing and punctuation cleanup
- Custom EIT lexicon corrections (common mis-transcriptions)
- Punctuation restoration
"""

import logging
import re
import string

logger = logging.getLogger(__name__)

# Default filler words common in L2 learner speech
DEFAULT_FILLER_WORDS = {"um", "uh", "eh", "hmm", "ah", "mm", "er", "hm"}

# Common Whisper mis-transcriptions for Spanish learner speech
# Format: "wrong transcription" → "corrected form"
# Expand this list based on your data analysis
COMMON_CORRECTIONS = {
    # Whisper sometimes mishears Spanish function words
    "lo es": "lo es",
    "una el": "un el",
    # Add more as you discover patterns in your error analysis
}


def remove_filler_words(
    text: str,
    filler_words: set[str] | None = None,
) -> str:
    """
    Remove filler words from transcribed text.

    L2 learners produce many more disfluencies (um, uh, eh) than native
    speakers. These are not part of the EIT target sentence and should
    be removed before scoring.

    Args:
        text:         Raw transcription string.
        filler_words: Set of words to remove. Defaults to DEFAULT_FILLER_WORDS.

    Returns:
        Cleaned string with filler words removed.
    """
    if filler_words is None:
        filler_words = DEFAULT_FILLER_WORDS

    words = text.split()
    cleaned = [w for w in words if w.lower().strip(string.punctuation) not in filler_words]
    return " ".join(cleaned)


def lowercase_and_strip(text: str) -> str:
    """Lowercase, strip leading/trailing whitespace, collapse multiple spaces."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_punctuation(text: str, keep: str = "") -> str:
    """
    Remove punctuation from text.

    Args:
        text: Input string.
        keep: Characters to keep (e.g. keep="'" to preserve apostrophes).

    Returns:
        String with punctuation removed.
    """
    punct = "".join(c for c in string.punctuation if c not in keep)
    return text.translate(str.maketrans("", "", punct))


def apply_lexicon_corrections(
    text: str,
    corrections: dict[str, str] | None = None,
    lexicon_path: str | None = None,
) -> str:
    """
    Apply a lookup table of known transcription errors.

    You build this table by analyzing systematic errors in your
    Whisper output vs. human transcripts. Common patterns:
    - Whisper mishearing specific Spanish phonemes
    - Consistent errors on certain EIT target sentences

    Args:
        text:           Input transcription.
        corrections:    Dict of {wrong: correct} substitutions.
        lexicon_path:   Path to a text file with "wrong → correct" lines.

    Returns:
        Corrected string.
    """
    all_corrections = dict(COMMON_CORRECTIONS)

    if corrections:
        all_corrections.update(corrections)

    if lexicon_path:
        try:
            with open(lexicon_path) as f:
                for line in f:
                    line = line.strip()
                    if "→" in line:
                        wrong, correct = line.split("→", 1)
                        all_corrections[wrong.strip()] = correct.strip()
        except FileNotFoundError:
            logger.warning(f"Lexicon file not found: {lexicon_path}")

    for wrong, correct in all_corrections.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)

    return text


def postprocess(
    text: str,
    lowercase: bool = True,
    strip_fillers: bool = True,
    apply_corrections: bool = True,
    filler_words: set[str] | None = None,
    lexicon_path: str | None = None,
) -> str:
    """
    Full post-processing pipeline for a single transcription.

    Args:
        text:               Raw ASR output string.
        lowercase:          Whether to lowercase.
        strip_fillers:      Whether to remove filler words.
        apply_corrections:  Whether to apply lexicon corrections.
        filler_words:       Custom filler word set.
        lexicon_path:       Path to custom correction lexicon.

    Returns:
        Cleaned, post-processed transcription string.
    """
    if lowercase:
        text = lowercase_and_strip(text)

    if strip_fillers:
        text = remove_filler_words(text, filler_words=filler_words)

    if apply_corrections:
        text = apply_lexicon_corrections(text, lexicon_path=lexicon_path)

    # Final cleanup — collapse any extra spaces left behind
    text = re.sub(r"\s+", " ", text).strip()

    return text


def postprocess_batch(
    results: list[dict],
    **kwargs,
) -> list[dict]:
    """
    Apply postprocess() to a list of transcription result dicts.

    Args:
        results:  [{"path": Path, "text": str, ...}, ...]
        **kwargs: Passed to postprocess().

    Returns:
        Same list with "text" replaced by post-processed version,
        and original stored in "raw_text".
    """
    for item in results:
        item["raw_text"] = item.get("text", "")
        item["text"] = postprocess(item["raw_text"], **kwargs)
    return results
