#!/usr/bin/env python3
"""Generate audio for testScript.txt with natural sentence pauses.

Why this exists:
VoxCPM's `clone` command feeds the whole text file in as a single string, and
core.py collapses every newline / run of whitespace into one space. The model
then renders the entire script as one continuous utterance, which sounds fast
with almost no gaps between sentences.

This script fixes that by:
  1. Loading the model once.
  2. Splitting testScript.txt into paragraphs and sentences.
  3. Synthesizing each sentence separately with the chosen reference voice.
  4. Concatenating the pieces with a chunk of silence inserted between them
     (short pause between sentences, longer pause between paragraphs).
  5. Writing one output wav per reference voice.

Run from the project root:

    uv run python generate_with_pauses.py
"""

import json
import hashlib
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from voxcpm import VoxCPM

# --- configuration ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_DIR / "reference"
OUTPUT_DIR = PROJECT_DIR / "out" / "story_reads"

SENTENCE_PAUSE_S = 0.35   # silence inserted between sentences
PARAGRAPH_PAUSE_S = 0.70  # silence inserted between paragraphs

CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 10
SEED = 43  # 42 tends to hallucinate on short lines; 43 is the default base seed

# Per-call reference denoising is DISABLED: VoxCPM's internal enhance() runs a
# loudness-normalize step that needs torchaudio's torchcodec/FFmpeg backend,
# which isn't installed here (RuntimeError: Could not load libtorchcodec).
# Instead we pre-clean noisy references OFFLINE with the ZipEnhancer denoiser and
# point the VOICE_* constants at the cleaned files (e.g. dragon_clean.wav). So
# keep this False unless FFmpeg is installed.
DENOISE_REFERENCE = False

# --- Duration guard (hallucination safety net) -----------------------------
# If a generated clip is far longer than the text should take, VoxCPM2 has
# probably appended unrelated speech. Regenerate with a different seed until the
# duration is sane or the retry budget runs out (then keep the shortest clip).
DURATION_GUARD_ENABLED = True
DURATION_MAX_RETRIES = 3          # extra attempts after the first generation
WORDS_PER_SECOND = 0.7            # rough speaking budget: words * this + base
DURATION_BASE_S = 1.5            # fixed overhead per sentence
DURATION_MIN_LIMIT_S = 2.5       # never flag anything under this as too long

# --- Pitch guard (timbre-drift safety net) ---------------------------------
# VoxCPM samples each sentence independently, so an unlucky seed can render a
# sentence far below/above the reference's natural pitch (the "deep last line"
# problem). For each sentence we try seeds in order and EARLY-STOP on the first
# candidate whose duration is sane AND whose median pitch is within tolerance of
# the reference voice's own median pitch. Only problem sentences pay for extra
# candidates, so normal sentences stay at one generation (~no speed change).
PITCH_GUARD_ENABLED = True
PITCH_TOLERANCE_SEMITONES = 3.0   # accept if within +/- this of the reference median f0
MAX_CANDIDATE_SEEDS = 3           # total seeds tried per sentence (incl. the first)
PITCH_FMIN_HZ = 65.0              # ~C2, low male
PITCH_FMAX_HZ = 400.0            # covers female range

# --- Audio-quality guard (defect detection -> reseed, no signal processing) -
# Extends the duration/pitch guard with per-sentence defect checks. A candidate
# is REJECTED (and a new seed tried) if it clips, has a leading click, contains
# an over-long internal silence, or drifts far in timbre from the reference. All
# checks are detection-only: they trigger regeneration, they never alter audio.
AUDIO_QUALITY_GUARD_ENABLED = True
CLIP_PEAK_THRESH = 0.999          # samples at/above this magnitude count as clipped
CLIP_MAX_FRACTION = 0.0005        # reject if more than this fraction of samples clip
ONSET_CLICK_ENABLED = True
ONSET_WINDOW_MS = 12.0            # inspect this leading window for a click/pop
ONSET_CLICK_PEAK = 0.5           # leading-window peak must exceed this to be a click
ONSET_CLICK_RATIO = 3.0          # ...and be this many x the following short-term level
INTERNAL_SILENCE_MAX_S = 0.9     # reject if an internal silence gap exceeds this
SILENCE_RMS_FLOOR = 0.01         # frame RMS below this counts as silence
CENTROID_GUARD_ENABLED = True
CENTROID_TOLERANCE_RATIO = 1.6   # accept if within this ratio of the reference centroid

# --- Loudness normalization (pure gain, no timbre change) -------------------
# Per-sentence gain toward the story's median loudness, then a story-level peak
# normalize. Scalar gain only: it changes volume, not spectral content.
LOUDNESS_NORMALIZE_ENABLED = True
LOUDNESS_MAX_GAIN = 4.0          # never amplify/attenuate a sentence beyond this factor
LOUDNESS_MIN_RMS = 0.005         # sentences below this are treated as silent, not boosted
STORY_PEAK_TARGET = 0.95         # peak-normalize the finished story to this

# --- Edge silence trim (pure crop) -----------------------------------------
# Trim dead air at the head/tail of each generated sentence so the fixed pauses
# stay even. Cropping only; the kept audio is untouched.
TRIM_EDGES_ENABLED = True
TRIM_RMS_FLOOR = 0.008           # head/tail frames below this are trimmed
TRIM_KEEP_MS = 40.0              # leave this much margin so onsets aren't clipped

# --- Sentence cache --------------------------------------------------------
# Cache each generated sentence on disk keyed by (voice, text, gen params). A
# re-run reuses unchanged sentences (incl. ones shared across stories) and only
# regenerates edited ones. Generation params are in the key, so changing them
# auto-invalidates stale entries. Never auto-cleared; use --clear-cache to wipe.
SENTENCE_CACHE_ENABLED = True
SENTENCE_CACHE_DIR = OUTPUT_DIR.parent / "sentence_cache"
CACHE_LOGIC_VERSION = 1          # bump to invalidate ALL cached sentences at once
_CACHE_BYPASS = False            # set by --no-cache: this run neither reads nor writes cache

# --- Text pre-check --------------------------------------------------------
# Static checks on the story text BEFORE spending time generating audio. Warns
# only (never blocks); findings go to the review queue.
TEXT_CHECK_ENABLED = True
TEXT_LONG_SENTENCE_WORDS = 45    # warn on sentences longer than this (hallucination risk)

# --- Review queue ----------------------------------------------------------
# All guard rejects, timbre outliers and text warnings are written here so you
# have a concrete "listen/regenerate this" to-do list instead of scrollback.
REVIEW_QUEUE_PATH = OUTPUT_DIR.parent / "review_queue.json"

# --- Story set + voice assignment (calibration run) ------------------------
DREAMCANVAS = Path(
    "/Users/xi/Desktop/Projects/DreamCanvasWithPrebuiltInStory/DreamCanvas"
)
FIRST100 = DREAMCANVAS / "StoryFilesProduction" / "First100"
TTS_STORIES = DREAMCANVAS / "tts_stories"

# --- Voice references -------------------------------------------------------
# Grouped by the age-5 kid test. Nothing is deleted — unused voices are kept for
# the compare/audition script and future (esp. age 6-7) use.
#
# TIER 1 — ACCEPTED (production). Kid favourites: high, animated, slower female
# narration. Young children strongly prefer these.
VOICE_ROOSTER = "YTMP3GG_YouTube_The-Rooster-Who-Would-Not-Be-Quiet-read-_Media_9m1Ui-3Nt_4_009_128k.wav"
VOICE_KAIA = "reference2/'The Three Little Pigs and the Somewhat Bad Wolf' read by Kaia Gerber.wav"

# TIER 2 — ACCEPTABLE male voices. Kids are fine with all of them; the only
# complaint is they read a bit fast. simu_liu is the SLOWEST of them (best pace).
# Good candidates for age 6-7 / adventure stories.
VOICE_DRAGON = "YTMP3GG_YouTube_When-a-Dragon-Moves-In-read-by-Mark-Dupl_Media_6s7aSNUCkiM_009_128k.wav"
VOICE_ABDUL = "YTMP3GG_YouTube_Abdul-s-Story-read-by-Tramell-Tillman_Media_W3PUTWCqPBo_009_128k.wav"
VOICE_SIMU_LIU = "reference2/'The Sound of Silence' read by Simu Liu.wav"

# jabari — female, acceptable but not a favourite at age 5.
VOICE_JABARI = "YTMP3GG_YouTube_Jabari-Jumps-read-by-Sheryl-Lee-Ralph_Media_XwNNlgtHFiU_009_128k.wav"

# TIER 3 — NOT liked. fox_crow is the only voice the kids disliked. Kept for
# reference; do not assign in JOBS.
VOICE_FOX_CROW = "The Fox and the Crow (UK English — TheFableCottage.com) - The Fable Cottage (128k).wav"

# Other auditioned voices — not in rotation.
VOICE_NO_PICTURES = "The Book With No Pictures – 📄 Hilarious read aloud of a kids book with no pictures! - Buddy Son Storytime (128k).wav"
VOICE_MADDI = "'Maddi's Fridge' read by Jennifer Garner - StorylineOnline (128k).wav"
VOICE_TOO_MUCH_GLUE = "YTMP3GG_YouTube_Too-Much-Glue-read-by-Nicole-Byer_Media_5ISKUMy1980_009_128k.wav"

# Short aliases used by the CLI voice-override syntax (story:voice).
VOICE_ALIASES = {
    "rooster": VOICE_ROOSTER,
    "fox_crow": VOICE_FOX_CROW,
    "foxcrow": VOICE_FOX_CROW,
    "jabari": VOICE_JABARI,
    "kaia": VOICE_KAIA,
    "dragon": VOICE_DRAGON,
    "simu_liu": VOICE_SIMU_LIU,
    "simu": VOICE_SIMU_LIU,
    "abdul": VOICE_ABDUL,
    "no_pictures": VOICE_NO_PICTURES,
    "maddi": VOICE_MADDI,
    "too_much_glue": VOICE_TOO_MUCH_GLUE,
}


def resolve_voice(key: str) -> str | None:
    """Resolve a CLI voice token to a reference filename.

    Accepts an exact alias, an alias substring, or a substring of the reference
    filename. Returns None if nothing matches.
    """
    key = key.strip().lower()
    if not key:
        return None
    if key in VOICE_ALIASES:
        return VOICE_ALIASES[key]
    for name, fn in VOICE_ALIASES.items():
        if key in name:
            return fn
    for fn in VOICE_ALIASES.values():
        if key in fn.lower():
            return fn
    return None


def voice_tag_for(voice_name: str) -> str:
    """Short, filename-safe tag for a reference (its alias, else a cleaned stem)."""
    for name, fn in VOICE_ALIASES.items():
        if fn == voice_name:
            return name
    return safe_name(Path(voice_name).stem)

# (output label, story txt path, reference filename)
# VOICE DECISION: production uses ROOSTER as the single voice. Other VOICE_*
# constants are kept so any story can still be regenerated with a different
# voice on demand via the CLI `story:voice` override.
#
# STORY SET is NOT finalized yet — the text is still being reviewed. So the full
# 223-story auto-list is intentionally NOT active. `_discover_jobs()` is ready
# for when the review is done: flip the default by setting `JOBS = _discover_jobs()`.
# Until then JOBS holds a small placeholder set (all rooster) and specific
# stories are generated ad hoc via the CLI (e.g. `age4-5_story_251`, using
# _discover_jobs()'s labels once enabled, or a `story:rooster` override).
def _discover_jobs() -> list[tuple[str, Path, str]]:
    """Build the full job list from the production story folders (rooster voice).

    Covers the ~223 production stories: First100 (Age2-3 repetitive/narrative,
    Age4-5, Age6-7) plus tts_stories/batch_01..NN. `.bak` and non-story folders
    are excluded because only the exact source dirs are globbed. Labels match the
    scheme used elsewhere, e.g. age2-3_rep_story_001, age4-5_story_251,
    batch01_story_003.

    NOT called by default yet — enable with `JOBS = _discover_jobs()` once the
    story text review is finished and the story set is frozen.
    """
    jobs: list[tuple[str, Path, str]] = []
    first100_sources = [
        (FIRST100 / "Age2-3" / "REPETITIVE" / "txt", "age2-3_rep_"),
        (FIRST100 / "Age2-3" / "NARRATIVE" / "txt", "age2-3_nar_"),
        (FIRST100 / "Age4-5" / "txt", "age4-5_"),
        (FIRST100 / "Age6-7" / "txt", "age6-7_"),
    ]
    for d, prefix in first100_sources:
        if d.is_dir():
            for txt in sorted(d.glob("story_*.txt")):
                jobs.append((f"{prefix}{txt.stem}", txt, VOICE_ROOSTER))
    if TTS_STORIES.is_dir():
        for d in sorted(TTS_STORIES.glob("batch_*")):
            if d.is_dir():
                bnum = d.name.replace("batch_", "batch")  # batch_01 -> batch01
                for txt in sorted(d.glob("story_*.txt")):
                    jobs.append((f"{bnum}_{txt.stem}", txt, VOICE_ROOSTER))
    return jobs


# Placeholder story set while the text is still under review — all rooster.
# Replace with `JOBS = _discover_jobs()` when the story set is frozen.
JOBS = [
    ("batch01_story_001", TTS_STORIES / "batch_01" / "story_001.txt", VOICE_ROOSTER),
    ("batch01_story_002", TTS_STORIES / "batch_01" / "story_002.txt", VOICE_ROOSTER),
    ("batch01_story_003", TTS_STORIES / "batch_01" / "story_003.txt", VOICE_ROOSTER),
]

# --- Level 2 tone selection -------------------------------------------------
# Narration (no quotes) gets a calm storytelling tone. Dialogue (quoted speech)
# gets an expressive tone; if an attribution verb is present (shouted,
# whispered, laughed, ...) the tone is chosen to match that emotion, otherwise
# a gentle default is used. These parentheticals are VoxCPM2 style hints; the
# model reads them as instructions and does not speak them aloud.
NARRATION_TONE = "(slower pace, calm storytelling tone, clear pauses)"
DEFAULT_DIALOGUE_TONE = "(gentle, expressive voice)"

# Very short lines (fewer than this many words) get NO tone parenthetical.
# Measured cause of hallucination: on ultra-short lines like "Goodnight, Cow."
# (2 words) the (style) prefix occasionally makes VoxCPM2 append 5-7s of
# unrelated speech. A/B probe: tone = 2 hallucinations / 72 gens (max 7.2s);
# plain = 0 / 72 (max 2.4s). So suppress the prefix below this word count.
MIN_WORDS_FOR_TONE = 4

# Attribution verb / adverb -> tone. Matched as whole words, case-insensitive.
DIALOGUE_EMOTION = {
    # excited / loud
    "shouted": "(excited, loud voice)",
    "cried": "(excited, loud voice)",
    "yelled": "(excited, loud voice)",
    "exclaimed": "(excited, loud voice)",
    "shrieked": "(excited, loud voice)",
    "called": "(bright voice, calling out)",
    "announced": "(bright, clear voice)",
    # soft / gentle
    "whispered": "(soft, gentle whisper)",
    "murmured": "(soft, gentle voice)",
    "softly": "(soft, gentle voice)",
    "quietly": "(soft, gentle voice)",
    "gently": "(soft, gentle voice)",
    "sighed": "(soft, weary voice)",
    # stern / angry
    "growled": "(stern, sharp voice)",
    "snapped": "(stern, sharp voice)",
    "demanded": "(stern, firm voice)",
    "warned": "(stern, serious voice)",
    "hissed": "(stern, sharp voice)",
    # happy
    "laughed": "(warm, cheerful voice)",
    "giggled": "(warm, cheerful voice)",
    "chuckled": "(warm, cheerful voice)",
    "smiled": "(warm, cheerful voice)",
    # sad / afraid
    "sobbed": "(sad, trembling voice)",
    "wailed": "(sad, trembling voice)",
    "trembled": "(sad, trembling voice)",
}
# Precompiled word patterns for the emotion cues.
_EMOTION_PATTERNS = [
    (re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), tone)
    for word, tone in DIALOGUE_EMOTION.items()
]


def tone_for(sentence: str) -> str:
    """Pick a VoxCPM2 style tone for a single sentence (Level 2)."""
    # Ultra-short lines hallucinate when given a (style) prefix -> no tone.
    if len(re.findall(r"[A-Za-z0-9']+", sentence)) < MIN_WORDS_FOR_TONE:
        return ""
    if '"' not in sentence:
        return NARRATION_TONE
    for pattern, tone in _EMOTION_PATTERNS:
        if pattern.search(sentence):
            return tone
    return DEFAULT_DIALOGUE_TONE
# ---------------------------------------------------------------------------


def _word_count(sentence: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", sentence))


def expected_max_seconds(sentence: str) -> float:
    """Loose upper bound on how long a sane reading of this sentence should be."""
    return max(
        DURATION_MIN_LIMIT_S,
        _word_count(sentence) * WORDS_PER_SECOND + DURATION_BASE_S,
    )


# --- Pitch measurement (for the pitch guard) -------------------------------
_REFERENCE_F0_CACHE: dict[str, float | None] = {}


def _median_f0(audio: np.ndarray, sample_rate: int) -> float | None:
    """Median voiced fundamental frequency (Hz) of mono audio, or None if unvoiced."""
    if audio.size == 0:
        return None
    try:
        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            audio.astype(np.float32),
            fmin=PITCH_FMIN_HZ,
            fmax=PITCH_FMAX_HZ,
            sr=sample_rate,
        )
    except Exception:
        return None
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        return None
    return float(np.median(voiced))


def reference_median_f0(ref_path: Path) -> float | None:
    """Median pitch (Hz) of the reference clip, computed once and cached."""
    key = str(ref_path)
    if key in _REFERENCE_F0_CACHE:
        return _REFERENCE_F0_CACHE[key]
    value: float | None = None
    try:
        audio, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        value = _median_f0(np.asarray(audio, dtype=np.float32), sr)
    except Exception:
        value = None
    _REFERENCE_F0_CACHE[key] = value
    return value


def _semitone_diff(f0: float, ref_f0: float) -> float:
    """Absolute pitch distance in semitones between two frequencies."""
    return abs(12.0 * float(np.log2(f0 / ref_f0)))


# --- Sentence cache helpers ------------------------------------------------
def _sentence_cache_path(voice_tag: str, sentence: str) -> Path | None:
    """Disk path for a cached sentence, keyed by voice + prompt + gen params.

    The generation-affecting parameters (and the tone-decorated prompt) are all
    folded into the hash, so changing any of them yields a different key and the
    stale entry is simply missed rather than silently reused.
    """
    if not SENTENCE_CACHE_ENABLED:
        return None
    prompt = f"{tone_for(sentence)}{sentence}"
    key = "|".join([
        f"v={voice_tag}",
        f"logic={CACHE_LOGIC_VERSION}",
        f"cfg={CFG_VALUE}",
        f"ts={INFERENCE_TIMESTEPS}",
        f"seed={SEED}",
        f"txt={prompt}",
    ])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return SENTENCE_CACHE_DIR / f"{voice_tag}_{digest}.wav"


def _write_cache(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    """Persist a finished sentence wav to the cache (best-effort, never fatal)."""
    try:
        SENTENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.asarray(wav, dtype=np.float32), sample_rate)
    except Exception:
        pass


# --- Audio metrics (guard + normalization + trim) --------------------------
_REFERENCE_CENTROID_CACHE: dict[str, float | None] = {}


def _rms(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))


def _peak(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.max(np.abs(wav)))


def _clip_fraction(wav: np.ndarray) -> float:
    """Fraction of samples at/above the clipping threshold."""
    if wav.size == 0:
        return 0.0
    return float(np.mean(np.abs(wav) >= CLIP_PEAK_THRESH))


def _has_onset_click(wav: np.ndarray, sample_rate: int) -> bool:
    """Detect a loud transient in the leading window (a click/pop before speech)."""
    n = wav.size
    if n == 0:
        return False
    w = max(1, int(ONSET_WINDOW_MS / 1000.0 * sample_rate))
    follow = wav[w:w + int(0.2 * sample_rate)]
    if follow.size == 0:
        return False
    head_peak = float(np.max(np.abs(wav[:w])))
    follow_level = float(np.median(np.abs(follow)) + 1e-6)
    return head_peak >= ONSET_CLICK_PEAK and head_peak > ONSET_CLICK_RATIO * follow_level


def _max_internal_silence(wav: np.ndarray, sample_rate: int) -> float:
    """Longest silent gap (seconds) BETWEEN the first and last voiced frame."""
    n = wav.size
    if n == 0:
        return 0.0
    frame = max(1, int(0.02 * sample_rate))
    count = n // frame
    if count < 3:
        return 0.0
    block = wav[:count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
    silent = rms < SILENCE_RMS_FLOOR
    voiced_idx = np.where(~silent)[0]
    if voiced_idx.size == 0:
        return 0.0
    lo, hi = int(voiced_idx[0]), int(voiced_idx[-1])
    interior = silent[lo:hi + 1]
    longest = cur = 0
    for v in interior:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest * frame / sample_rate


def _spectral_centroid_median(wav: np.ndarray, sample_rate: int) -> float | None:
    """Median spectral centroid (Hz) — a proxy for brightness (dull vs sharp)."""
    if wav.size < 512:
        return None
    try:
        c = librosa.feature.spectral_centroid(y=wav.astype(np.float32), sr=sample_rate)
    except Exception:
        return None
    c = c[np.isfinite(c)]
    if c.size == 0:
        return None
    return float(np.median(c))


def _trim_edges(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Crop leading/trailing dead air, keeping a small margin. Cropping only."""
    n = wav.size
    if n == 0:
        return wav
    frame = max(1, int(0.01 * sample_rate))
    count = n // frame
    if count < 3:
        return wav
    block = wav[:count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
    voiced = np.where(rms >= TRIM_RMS_FLOOR)[0]
    if voiced.size == 0:
        return wav
    keep = int(TRIM_KEEP_MS / 1000.0 * sample_rate)
    start = max(0, int(voiced[0]) * frame - keep)
    end = min(n, (int(voiced[-1]) + 1) * frame + keep)
    return wav[start:end]


def reference_median_centroid(ref_path: Path) -> float | None:
    """Median spectral centroid (Hz) of the reference clip, computed once."""
    key = str(ref_path)
    if key in _REFERENCE_CENTROID_CACHE:
        return _REFERENCE_CENTROID_CACHE[key]
    value: float | None = None
    try:
        audio, sr = sf.read(str(ref_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        value = _spectral_centroid_median(np.asarray(audio, dtype=np.float32), sr)
    except Exception:
        value = None
    _REFERENCE_CENTROID_CACHE[key] = value
    return value


def check_story_text(paragraphs: list[list[str]]) -> list[dict]:
    """Static text checks run BEFORE generation. Returns a list of issue dicts."""
    issues: list[dict] = []
    if not paragraphs:
        return [{"kind": "text", "detail": "empty story"}]
    placeholder = re.compile(
        r"[\[\{<][^\]\}>]*(name|child|kid|placeholder|todo|xxx)[^\]\}>]*[\]\}>]",
        re.IGNORECASE,
    )
    for p_i, sentences in enumerate(paragraphs):
        # Quote balance is checked per paragraph (a quoted line may be split
        # across sentences, so per-sentence balance would false-positive).
        if " ".join(sentences).count('"') % 2 != 0:
            issues.append({
                "kind": "text", "paragraph": p_i,
                "detail": "unbalanced quotes in paragraph",
            })
        for s_i, s in enumerate(sentences):
            wc = _word_count(s)
            if wc == 0:
                issues.append({"kind": "text", "paragraph": p_i, "sentence": s_i,
                               "detail": "empty sentence", "text": s})
            elif wc > TEXT_LONG_SENTENCE_WORDS:
                issues.append({"kind": "text", "paragraph": p_i, "sentence": s_i,
                               "detail": f"long sentence ({wc} words)", "text": s[:60]})
            if placeholder.search(s):
                issues.append({"kind": "text", "paragraph": p_i, "sentence": s_i,
                               "detail": "possible placeholder", "text": s[:60]})
    return issues


def generate_sentence(model, sentence: str, ref_path: Path, sample_rate: int,
                      voice_tag: str = "voice") -> tuple[np.ndarray, list[str]]:
    """Generate one sentence, early-stopping on the first clean candidate.

    Returns (wav, issues). `issues` is empty when a clean candidate was found;
    otherwise it lists the defect reasons of the best-scoring fallback so the
    caller can record them in the review queue.

    A disk cache keyed by (voice, prompt, gen params) short-circuits everything:
    a cache hit returns immediately without running the model or any guard.

    Seeds are tried in order (SEED, SEED+1, ...). A candidate is accepted
    immediately when duration, pitch, clipping, onset-click, internal-silence and
    timbre (spectral centroid vs the reference) are all within tolerance. Normal
    sentences pass on the first seed; only defective ones pay for extra
    candidates. If none pass, the lowest-penalty candidate is kept.
    """
    # --- cache lookup (skips model + guards entirely on hit) ---
    cache_path = _sentence_cache_path(voice_tag, sentence)
    if (SENTENCE_CACHE_ENABLED and not _CACHE_BYPASS
            and cache_path is not None and cache_path.exists()):
        try:
            cached, _sr = sf.read(str(cache_path), dtype="float32", always_2d=False)
            if cached.ndim > 1:
                cached = cached.mean(axis=1)
            return np.asarray(cached, dtype=np.float32), []
        except Exception:
            pass  # unreadable cache -> fall through and regenerate

    prompt = f"{tone_for(sentence)}{sentence}"
    limit = expected_max_seconds(sentence)
    ref_f0 = reference_median_f0(ref_path) if PITCH_GUARD_ENABLED else None
    ref_centroid = (
        reference_median_centroid(ref_path)
        if (AUDIO_QUALITY_GUARD_ENABLED and CENTROID_GUARD_ENABLED) else None
    )

    any_guard = DURATION_GUARD_ENABLED or PITCH_GUARD_ENABLED or AUDIO_QUALITY_GUARD_ENABLED
    n_seeds = max(1, MAX_CANDIDATE_SEEDS if any_guard else 1)

    best = None
    best_score = float("inf")
    best_reasons: list[str] = []
    for i in range(n_seeds):
        wav = np.asarray(
            model.generate(
                text=prompt,
                reference_wav_path=str(ref_path),
                cfg_value=CFG_VALUE,
                inference_timesteps=INFERENCE_TIMESTEPS,
                denoise=DENOISE_REFERENCE,
                seed=SEED + i,  # vary seed so retries actually differ
            ),
            dtype=np.float32,
        )
        dur = len(wav) / sample_rate
        dur_over = max(0.0, dur - limit) if DURATION_GUARD_ENABLED else 0.0

        # Pitch deviation from the reference voice (0 if unmeasurable).
        semi = 0.0
        if PITCH_GUARD_ENABLED and ref_f0:
            cand_f0 = _median_f0(wav, sample_rate)
            if cand_f0:
                semi = _semitone_diff(cand_f0, ref_f0)

        # Audio-quality measurements.
        clip_frac = 0.0
        onset_click = False
        internal_sil = 0.0
        centroid_ratio = 1.0
        if AUDIO_QUALITY_GUARD_ENABLED:
            clip_frac = _clip_fraction(wav)
            if ONSET_CLICK_ENABLED:
                onset_click = _has_onset_click(wav, sample_rate)
            internal_sil = _max_internal_silence(wav, sample_rate)
            if CENTROID_GUARD_ENABLED and ref_centroid:
                cand_c = _spectral_centroid_median(wav, sample_rate)
                if cand_c and ref_centroid > 0:
                    centroid_ratio = max(cand_c / ref_centroid, ref_centroid / cand_c)

        reasons: list[str] = []
        if DURATION_GUARD_ENABLED and dur > limit:
            reasons.append(f"long {dur:.1f}s>{limit:.1f}s")
        if PITCH_GUARD_ENABLED and semi > PITCH_TOLERANCE_SEMITONES:
            reasons.append(f"pitch {semi:.1f}st")
        if AUDIO_QUALITY_GUARD_ENABLED:
            if clip_frac > CLIP_MAX_FRACTION:
                reasons.append(f"clip {clip_frac * 100:.2f}%")
            if onset_click:
                reasons.append("onset-click")
            if internal_sil > INTERNAL_SILENCE_MAX_S:
                reasons.append(f"gap {internal_sil:.1f}s")
            if CENTROID_GUARD_ENABLED and ref_centroid and centroid_ratio > CENTROID_TOLERANCE_RATIO:
                reasons.append(f"timbre x{centroid_ratio:.2f}")

        if not reasons:
            if SENTENCE_CACHE_ENABLED and not _CACHE_BYPASS and cache_path is not None:
                _write_cache(cache_path, wav, sample_rate)
            return wav, []

        score = (
            dur_over
            + semi
            + clip_frac * 1000.0
            + (5.0 if onset_click else 0.0)
            + max(0.0, internal_sil - INTERNAL_SILENCE_MAX_S)
            + max(0.0, centroid_ratio - 1.0)
        )
        if score < best_score:
            best, best_score, best_reasons = wav, score, reasons
        print(
            f"    guard: seed {SEED + i} rejected ({', '.join(reasons)}) "
            f'"{sentence[:36]}" - trying next',
            file=sys.stderr,
        )
    print(
        f"    guard: kept best candidate (score {best_score:.2f}; "
        f"{', '.join(best_reasons)}) after {n_seeds} seeds",
        file=sys.stderr,
    )
    if (SENTENCE_CACHE_ENABLED and not _CACHE_BYPASS
            and cache_path is not None and best is not None):
        _write_cache(cache_path, best, sample_rate)
    return best, best_reasons


def safe_name(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return cleaned or "voice"


def split_into_paragraphs(text: str) -> list[list[str]]:
    """Return a list of paragraphs, each a list of sentence strings."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        # Split after sentence-ending punctuation, allowing an optional closing
        # quote before the whitespace, so a sentence like `... ever." Clover...`
        # is split into two (otherwise it glues to the next sentence and gets no
        # inter-sentence pause). Matches check_tts.py's splitter.
        sentences = re.split(r"(?<=[.!?])[\"']?\s+", block)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            paragraphs.append(sentences)
    return paragraphs


def main() -> int:
    global _CACHE_BYPASS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Separate flags (--...) from positional filter/override tokens.
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    positional = [a for a in args if not a.startswith("--")]

    if "--no-cache" in flags:
        _CACHE_BYPASS = True
        print("cache: bypassed for this run (--no-cache)", file=sys.stderr)
    if "--clear-cache" in flags:
        removed = 0
        if SENTENCE_CACHE_DIR.exists():
            for f in SENTENCE_CACHE_DIR.glob("*.wav"):
                try:
                    f.unlink()
                    removed += 1
                except Exception:
                    pass
        print(f"cache: cleared {removed} cached sentence(s) (--clear-cache)", file=sys.stderr)

    # Optional CLI filter / voice override.
    #  - Plain token filters jobs by story label or voice (substring match):
    #       uv run python generate_with_pauses.py jabari
    #       uv run python generate_with_pauses.py story_252
    #  - "story:voice" regenerates a specific story with a specific voice,
    #    overriding the voice assigned in JOBS:
    #       uv run python generate_with_pauses.py story_252:rooster
    #       uv run python generate_with_pauses.py story_651:fox_crow
    #  - Flags: --no-cache (ignore cache), --clear-cache (wipe cache first).
    plain_filters: list[str] = []
    overrides: list[tuple[str, str]] = []
    for a in positional:
        sep = ":" if ":" in a else ("=" if "=" in a else None)
        if sep:
            story_key, voice_key = a.split(sep, 1)
            overrides.append((story_key.strip().lower(), voice_key.strip()))
        else:
            plain_filters.append(a.lower())

    jobs: list[tuple] = []
    seen: set[str] = set()

    # Voice-override jobs (story:voice).
    for story_key, voice_key in overrides:
        ref = resolve_voice(voice_key)
        if ref is None:
            print(f"Unknown voice '{voice_key}'. Known: {sorted(VOICE_ALIASES)}",
                  file=sys.stderr)
            return 1
        matched = [job for job in JOBS if story_key in job[0].lower()]
        if not matched:
            print(f"No story matches '{story_key}'. Known labels: {[j[0] for j in JOBS]}",
                  file=sys.stderr)
            return 1
        # Tag the output with the voice so it doesn't overwrite the default take.
        for label, txt_path, _old_voice in matched:
            jobs.append((label, txt_path, ref))
            seen.add(label)

    # Plain filter jobs (story or voice substring), using their JOBS voice.
    for w in plain_filters:
        for job in JOBS:
            if job[0] in seen:
                continue
            if w in job[0].lower() or w in str(job[2]).lower():
                jobs.append(job)
                seen.add(job[0])

    if not overrides and not plain_filters:
        jobs = JOBS
    elif not jobs:
        print(f"No jobs match {positional}. Known labels: {[j[0] for j in JOBS]}",
              file=sys.stderr)
        return 1

    # CLI-invoked runs (any positional argument) tag their output with _cmd to
    # distinguish them from the default full-batch generation.
    cli_run = bool(positional)

    print("Loading VoxCPM model (once)...", file=sys.stderr)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=DENOISE_REFERENCE)
    sample_rate = model.tts_model.sample_rate

    sentence_gap = np.zeros(int(SENTENCE_PAUSE_S * sample_rate), dtype=np.float32)
    paragraph_gap = np.zeros(int(PARAGRAPH_PAUSE_S * sample_rate), dtype=np.float32)

    failures = []
    review_entries: list[dict] = []
    for i, (label, txt_path, voice_name) in enumerate(jobs, start=1):
        ref_path = REFERENCE_DIR / voice_name
        # Every output name carries the story label AND the voice; CLI runs add _cmd.
        stem = f"{label}_{voice_tag_for(voice_name)}"
        if cli_run:
            stem += "_cmd"
        output_path = OUTPUT_DIR / f"{stem}.wav"
        print(f"[{i}/{len(jobs)}] {label}  <- {txt_path.name}  voice={voice_name[:32]}")

        if not txt_path.exists():
            print(f"  MISSING story txt: {txt_path}", file=sys.stderr)
            failures.append(label)
            continue
        if not ref_path.exists():
            print(f"  MISSING reference: {ref_path}", file=sys.stderr)
            failures.append(label)
            continue

        paragraphs = split_into_paragraphs(txt_path.read_text(encoding="utf-8"))
        if not paragraphs:
            print("  empty story, skipping", file=sys.stderr)
            failures.append(label)
            continue

        vtag = voice_tag_for(voice_name)

        # (Text pre-check) — warn, never block. Findings go to the review queue.
        if TEXT_CHECK_ENABLED:
            for issue in check_story_text(paragraphs):
                issue.update({"label": label, "voice": vtag})
                review_entries.append(issue)
                loc = ""
                if "paragraph" in issue:
                    loc = f" (p{issue['paragraph']}"
                    if "sentence" in issue:
                        loc += f"s{issue['sentence']}"
                    loc += ")"
                print(f"  text-check{loc}: {issue['detail']}", file=sys.stderr)

        try:
            # Pass 1 — generate every sentence (cache-aware) + trim dead air.
            items: list[dict] = []
            for p_index, sentences in enumerate(paragraphs):
                for s_index, sentence in enumerate(sentences):
                    wav, s_issues = generate_sentence(
                        model, sentence, ref_path, sample_rate, vtag
                    )
                    wav = np.asarray(wav, dtype=np.float32)
                    if TRIM_EDGES_ENABLED:
                        wav = _trim_edges(wav, sample_rate)
                    items.append({"p": p_index, "s": s_index, "text": sentence, "wav": wav})
                    for reason in s_issues:
                        review_entries.append({
                            "kind": "guard", "label": label, "voice": vtag,
                            "paragraph": p_index, "sentence": s_index,
                            "detail": reason, "text": sentence[:60],
                        })

            # Pass 2 — per-sentence loudness normalize toward the story median
            # (scalar gain only: changes volume, not timbre).
            if LOUDNESS_NORMALIZE_ENABLED and items:
                rmss = [_rms(it["wav"]) for it in items]
                voiced = [r for r in rmss if r >= LOUDNESS_MIN_RMS]
                if voiced:
                    target = float(np.median(voiced))
                    lo, hi = 1.0 / LOUDNESS_MAX_GAIN, LOUDNESS_MAX_GAIN
                    for it, r in zip(items, rmss):
                        if r >= LOUDNESS_MIN_RMS:
                            gain = min(hi, max(lo, target / r))
                            it["wav"] = (it["wav"] * gain).astype(np.float32)

            # Relative brightness outliers (report only — no EQ, no auto-fix).
            if items:
                centroids = [_spectral_centroid_median(it["wav"], sample_rate) for it in items]
                valid = [c for c in centroids if c]
                if valid:
                    med_c = float(np.median(valid))
                    for it, c in zip(items, centroids):
                        if c and med_c > 0:
                            ratio = max(c / med_c, med_c / c)
                            if ratio > CENTROID_TOLERANCE_RATIO:
                                tag = "dull" if c < med_c else "sharp"
                                review_entries.append({
                                    "kind": "timbre-outlier", "label": label, "voice": vtag,
                                    "paragraph": it["p"], "sentence": it["s"],
                                    "detail": f"{tag} (centroid x{ratio:.2f} vs story median)",
                                    "text": it["text"][:60],
                                })

            # Assemble with fixed pauses.
            pieces: list[np.ndarray] = []
            segments: list[dict] = []
            cursor = 0  # running sample offset into the concatenated audio
            prev_p: int | None = None
            for it in items:
                if prev_p is not None:
                    gap = sentence_gap if it["p"] == prev_p else paragraph_gap
                    pieces.append(gap)
                    cursor += len(gap)
                start = cursor
                pieces.append(it["wav"])
                cursor += len(it["wav"])
                segments.append({
                    "paragraph": it["p"], "sentence": it["s"], "text": it["text"],
                    "start": round(start / sample_rate, 3),
                    "end": round(cursor / sample_rate, 3),
                })
                prev_p = it["p"]

            audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

            # Story-level peak normalize (scalar gain).
            if LOUDNESS_NORMALIZE_ENABLED and audio.size:
                peak = _peak(audio)
                if peak > 1e-6:
                    audio = (audio * (STORY_PEAK_TARGET / peak)).astype(np.float32)

            sf.write(str(output_path), audio, sample_rate)

            # Sidecar timestamps for future text-scroll / karaoke highlighting.
            timing_path = output_path.with_suffix(".json")
            timing = {
                "audio": output_path.name,
                "label": label,
                "voice": vtag,
                "sample_rate": sample_rate,
                "duration": round(len(audio) / sample_rate, 3),
                "segments": segments,
            }
            timing_path.write_text(
                json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  saved {output_path.name} ({len(audio) / sample_rate:.1f}s) + {timing_path.name}")
        except Exception as exc:  # noqa: BLE001 - report and continue to next job
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append(label)

    print()
    print(f"Done. {len(jobs) - len(failures)} succeeded, {len(failures)} failed.")

    # Write the review queue: a concrete "listen/regenerate this" to-do list.
    if review_entries:
        try:
            REVIEW_QUEUE_PATH.write_text(
                json.dumps(review_entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"review queue: {len(review_entries)} item(s) -> {REVIEW_QUEUE_PATH.name}",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"could not write review queue: {exc}", file=sys.stderr)

    if failures:
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
