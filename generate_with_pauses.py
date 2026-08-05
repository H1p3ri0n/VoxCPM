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
REFERENCE_DIR = PROJECT_DIR / "ref"
# OUTPUT_BASE is the shared root: the sentence cache, review queue and dur-report
# all live directly under it so they are shared across age groups. Each story's
# audio is routed into an age-group subfolder by _output_dir_for_label(), so
# age2-3 / age4-5 / age6-7 outputs never collide. OUTPUT_DIR is kept only so the
# cache/review paths below (OUTPUT_DIR.parent) stay anchored at OUTPUT_BASE.
OUTPUT_BASE = PROJECT_DIR / "out" / "first100"
OUTPUT_DIR = OUTPUT_BASE / "age2-3"

SENTENCE_PAUSE_S = 0.45   # silence between sentences within a paragraph (single \n).
                          # Per-age via AGE_PROFILES (age2-3 uses 0.7).
PARAGRAPH_PAUSE_S = 0.80  # silence between paragraphs (blank line / \n\n).
                          # Per-age via AGE_PROFILES (age2-3 uses 1.2).

# A quote + its short attribution tag are merged into one chunk only if the
# combined result stays at/under this word count. Set generously (16): the tail
# is already capped at <=6 words and the quote is validator-capped (LINE_MAX_WORDS
# 18/22), so 16 covers virtually all real dialogue while still blocking a runaway
# merge. A lower value (e.g. 10) orphans common 9-word questions + tag as tiny
# mumble-prone clips — the very thing the merge exists to prevent.
MERGE_MAX_WORDS = 16

# When True, each sentence is prefixed with a VoxCPM style tag (see tone_for).
# --notone turns this off for A/B testing a plain, tag-free prompt.
TONE_ENABLED = True

# Read granularity. Default False = one-sentence-at-a-time (split -> generate each
# -> stitch with silence). --paragraph flips this True for age4-5 + age6-7 + tts
# batch stories: each WHOLE paragraph is generated as a single utterance so the
# model flows naturally across its sentences (its own internal prosody/pauses),
# and paragraphs are stitched with the paragraph pause. Output is tagged _para so
# it never overwrites the per-sentence production take. Paragraph prompts use the
# narration tone (a mixed narration+dialogue paragraph shouldn't be tagged one
# dialogue emotion), and each paragraph is a SINGLE generation at the fixed base
# seed with NO guards (no duration/pitch/quality checks, no seed retries) - the
# plain "read it once" behaviour. Scope is intentional: age2-3 stays per-sentence.
READ_PARAGRAPH_MODE = False
# cfg used specifically in paragraph mode: a touch tighter than the 2.5 default
# for the most faithful cloned timbre over the longer whole-paragraph utterance
# (user-chosen). An explicit --cfg= on the command line still overrides it.
PARAGRAPH_CFG_VALUE = 2.6
# Fixed storytelling narration tone for direct / paragraph story reads. Used
# VERBATIM (no mood-matching, no per-age/lively override, and no PACE_HINT
# folding) so a directly-read story gets one clean, consistent narration voice.
# It carries its own pace control. Paragraph mode has NO guards, so a STRONGER
# "slow" wording is safe here (there's no pitch guard for it to fight, unlike the
# old sentence-mode reason "very slow" was retired): a soft "slower pace" hint
# only got ~0.35 spw (too fast); "slow, unhurried" pushes the native read slower
# WITHOUT any time-stretch (which would colour the voice). Still not "very slow".
PARAGRAPH_NARRATION_TONE = "(slow, unhurried storytelling pace, expressive, clear pauses)"

# Playback-speed multiplier applied per sentence (pitch-preserving time-stretch).
# 1.0 = model's native pace; <1.0 = slower, >1.0 = faster. Set via --speed=X.
# The model has no native speed knob, so this is the lever to match an older,
# slower-paced reference read.
SPEED = 1.0

CFG_VALUE = 2.5  # 2.5 = the sweet spot: clean articulation + stable cloned timbre with a
                 # touch more expressive prosody than 2.6. 2.3 was TRIED for even more
                 # expression but it loosened the reference lock too much: swallowed leading
                 # words (`"We..."`), stray extra phonemes, and a coarse/rough voice on some
                 # lines. 2.5 keeps the --lively + emotion-tone expressiveness WITHOUT those
                 # artifacts (listener-confirmed clean). Do NOT chase expression by lowering
                 # cfg further — that reintroduces the 2.3 defects; expressiveness must come
                 # from --lively + the tones. cfg is in the cache key -> changing it
                 # regenerates every sentence. (Override per-run with --cfg=X.)
INFERENCE_TIMESTEPS = 10
SEED = 45  # 45 is the default base seed (43 hallucinated an extra line on story_001)

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
WORDS_PER_SECOND = 0.5            # rough speaking budget: words * this + base
                                 # (0.5 calibrated from rooster story_001 dur-report:
                                 # natural spw 0.31-0.61, onomatopoeia up to 0.84;
                                 # this limit clears the slowest natural read yet
                                 # still catches ~2x appended-speech hallucinations)
DURATION_BASE_S = 1.5            # fixed overhead per sentence
DURATION_MIN_LIMIT_S = 2.5       # never flag anything under this as too long
PAUSE_MARK_ALLOWANCE_S = 1.0     # extra max-duration headroom per intentional long
                                 # pause mark (ellipsis '…' / em-dash '—'), which the
                                 # word-count budget doesn't account for (multi-syllable
                                 # onomatopoeia like 'Sparkle…' draws the pause out ~1.5-2s)

# --- Duration LOWER bound (rushed / too-fast guard) ------------------------
# Mirror of the upper bound, for the FAST side. Reject a candidate whose read is
# rushed: shorter than words*MIN_WORDS_PER_SECOND + MIN_DURATION_BASE_S. Values
# calibrated from the story_251 guard-report (accepted spw 0.27-0.80, median
# ~0.48-0.51 which the listener confirmed is an acceptable pace): this floor
# catches only the clearly rushed tail (~9% of sentences, effective spw < ~0.33
# on mid/long lines) without carpet-bombing the bulk, so it slows the worst
# offenders while keeping generation cheap. Raise MIN_WORDS_PER_SECOND toward
# 0.33 to be stricter (more re-rolls). Disable via DURATION_MIN_GUARD_ENABLED=False.
#
# ENABLED for ALL sentences (fast side): reject a rushed read. Gentle floor
# (0.28 coeff + 0.5s base -> effective ~0.36 spw on a 6-word line) so only the
# clearly rushed tail (~9% in the story_251 data) is caught, not the bulk. It
# pairs with the always-on upper bound so every sentence now has both a too-slow
# AND a too-fast check. Per-age via AGE_PROFILES; lower toward 0.25 for age2-3 if
# it over-rejects short lines (base dominates there and the pitch guard is busy).
DURATION_MIN_GUARD_ENABLED = True
MIN_WORDS_PER_SECOND = 0.28
MIN_DURATION_BASE_S = 0.5

# --- Quote-pace guard (intra-sentence rushed dialogue) ---------------------
# The duration guards measure the WHOLE chunk's AVERAGE pace, so a fast quote
# followed by a slower attribution tag ("...," said Coco.) averages out and slips
# through. This guard splits the generated audio at the pause before the
# attribution and checks the QUOTED span's rate on its own. It only fires on
# sentences that end in a short attribution tail AND have a detectable pause
# before it (no clear pause -> not checked, a deliberate false-negative rather
# than a false-positive). Conservative floor; raise QUOTE_MIN_WORDS_PER_SECOND to
# be stricter. Per-age via AGE_PROFILES. Disable with QUOTE_PACE_GUARD_ENABLED=False.
QUOTE_PACE_GUARD_ENABLED = True
QUOTE_MIN_WORDS_PER_SECOND = 0.37   # quoted span read faster than this = rushed
                                    # (0.40 -> 0.37: the model tops out at ~0.38-0.39
                                    # on some quotes, so 0.40 just burned all 8 seeds
                                    # and kept ~0.38 anyway; 0.37 accepts that quickly
                                    # while still catching the clearly rushed <0.37)
QUOTE_MIN_WORDS = 3                 # only judge quotes with at least this many words

# --- ASR verification (Whisper) for dialogue -------------------------------
# Opt-in (default OFF; enable per-age via AGE_PROFILES or the --asr CLI flag).
# For quote+attribution sentences ONLY (a small subset), transcribe each
# candidate with Whisper to: (1) verify the words are intelligible/correct —
# catches mumble AND hallucination via a fuzzy transcript match; (2) use
# word-level timestamps to measure the QUOTED span's pace accurately (replaces
# the fragile silence-split heuristic). The audio array is fed straight to the
# model, so NO FFmpeg/torchcodec is needed. Model auto-downloads once
# (~290MB whisper-base). Cost: one extra ASR pass per dialogue candidate —
# bounded because most lines are narration. ASR can mis-hear ultra-short clips,
# so the match ratio is lenient and QUOTE_MIN_WORDS keeps tiny quotes out.
ASR_VERIFY_ENABLED = False
ASR_MODEL = "openai/whisper-base"
ASR_MATCH_MIN_RATIO = 0.60          # transcript vs text similarity below this = mumble/halluc

# --- Pitch guard (timbre-drift safety net) ---------------------------------
# VoxCPM samples each sentence independently, so an unlucky seed can render a
# sentence far below/above the reference's natural pitch (the "deep last line"
# problem). For each sentence we try seeds in order and EARLY-STOP on the first
# candidate whose duration is sane AND whose median pitch is within tolerance of
# the reference voice's own median pitch. Only problem sentences pay for extra
# candidates, so normal sentences stay at one generation (~no speed change).
PITCH_GUARD_ENABLED = True
PITCH_TOLERANCE_SEMITONES = 3.0   # accept if within +/- this of the reference median f0
                                 # (back to the original 3.0: the 4.5 bump only made
                                 # sense with the global slow-pace prompt, which is now
                                 # disabled, so the systematic pitch offset is gone)
MAX_CANDIDATE_SEEDS = 8           # total seeds tried per sentence (incl. the first)
                                 # (raised 6 -> 8: the quote-pace guard adds another
                                 # constraint, so give more tries before a compromise)
PITCH_FMIN_HZ = 65.0              # ~C2, low male
PITCH_FMAX_HZ = 400.0            # covers female range

# --- Falling-ending guard (declarative sentences must not end RISING) --------
# Some period-ended sentences still land on a RISING (question-like) pitch, which
# sounds "floaty" / unfinished for storytelling. This guard measures the pitch
# slope over the last ENDING_WINDOW_S of a candidate and, for sentences that
# SHOULD fall (they end in "." — questions "?", exclamations "!" and suspense
# "..."/"…" are EXEMPT), REJECTS a clearly-rising take so another seed is tried,
# keeping the most-falling one. Slope units are Hz per pyin frame (~10.7ms at
# 48kHz); >0 rises, <0 falls. Set FALLING_ENDING_GUARD_ENABLED=False to disable.
FALLING_ENDING_GUARD_ENABLED = True
ENDING_WINDOW_S = 0.6             # analyse the pitch slope over this much of the tail
ENDING_RISE_SLOPE_MAX = 1.0       # reject if end-slope exceeds this (clearly rising);
                                 # raise to tolerate more, lower to be stricter
ENDING_RISE_PENALTY_WEIGHT = 0.5  # score weight per Hz/frame above the threshold
ENDING_SILENCE_TOP_DB = 30.0      # trailing audio this many dB below peak is treated
                                 # as silence and trimmed BEFORE the window is taken,
                                 # so narration (which leaves a pause after the last
                                 # word) is measured on real voiced tail, not blank

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
_CACHE_REFRESH = False           # set by --refresh-cache: skip the cache READ (force
                                 # regen) but still WRITE, i.e. overwrite these jobs'
                                 # cache entries without wiping other stories' cache

# --- Text pre-check --------------------------------------------------------
# Static checks on the story text BEFORE spending time generating audio. Warns
# only (never blocks); findings go to the review queue.
TEXT_CHECK_ENABLED = True
TEXT_LONG_SENTENCE_WORDS = 45    # warn on sentences longer than this (hallucination risk)

# --- Review queue ----------------------------------------------------------
# All guard rejects, timbre outliers and text warnings are written here so you
# have a concrete "listen/regenerate this" to-do list instead of scrollback.
REVIEW_QUEUE_PATH = OUTPUT_DIR.parent / "review_queue.json"

# --- Guard diagnostics log -------------------------------------------------
# When --guard-report is passed, EVERY per-seed candidate measurement (duration,
# seconds-per-word, pitch deviation, word count, limit, and accept/reject) is
# appended here. This captures the accepted seed too (the normal log only prints
# rejects), so a full run can be saved and analysed to calibrate the duration
# lower/upper bounds. None = off (default).
GUARD_LOG_PATH: Path | None = None

# --- Lively mode (--lively): trade stability for expressiveness -------------
# The stacked guards (pitch / centroid-timbre / quote-pace / duration-lower) plus
# loudness normalization tend to REJECT the natural, characterful FIRST take and
# re-roll to a safe, near-median "average" one, which sounds lifeless. The
# listener confirmed the original (few-guard) reads were livelier despite the
# occasional artifact. --lively thins the guards back to a minimal safety net —
# keep only hallucination-length, onset-click, internal-silence, and a LOOSE
# pitch bound (gross drift only) — loosens (does NOT disable) the too-fast floor
# so only genuinely rushed lines are caught, turns OFF loudness-norm + centroid +
# quote-pace, and cuts the seed budget so the lively first take usually survives.
# It ALSO forces the expressive narration tone across all ages. Applied in
# apply_age_profile() (so it overrides per-age profiles) AND main() (for the
# non-profile globals). NOTE: because lively changes NARRATION_TONE it also
# changes the sentence cache key, so lively and normal runs never collide in
# cache; still pair with --no-cache for a clean A/B.
#
# DEFAULT = True: lively IS the validated production config, so it runs with no
# flag needed (`... --no-cache` is the production command). The full-guard path
# is NOT deleted — it is still reachable with `--safe`, which flips this back to
# False and restores loudness-norm + centroid + quote-pace + tight pitch. Keep it
# that way: --safe is the A/B baseline for re-validating any future change (new
# voice, age6-7 set, etc.).
LIVELY_MODE = True

# --- Story set + voice assignment (calibration run) ------------------------
DREAMCANVAS = Path(r"D:\Repos\DreamCanvas")
FIRST100 = DREAMCANVAS / "StoryFilesProduction" / "First100"
TTS_STORIES = DREAMCANVAS / "tts_stories"

# --- Voice references -------------------------------------------------------
# Grouped by the age-5 kid test. Nothing is deleted — unused voices are kept for
# the compare/audition script and future (esp. age 6-7) use.
#
# TIER 1 — ACCEPTED (production). Kid favourites: high, animated, slower female
# narration. Young children strongly prefer these.
VOICE_ROOSTER = "YTMP3GG_YouTube_The-Rooster-Who-Would-Not-Be-Quiet-read-_Media_9m1Ui-3Nt_4_009_128k.wav"
# kaia clip sits directly in ref/ (no reference2/ subfolder), so the path is the
# bare filename — otherwise resolve_voice("kaia") points at a missing file.
VOICE_KAIA = "'The Three Little Pigs and the Somewhat Bad Wolf' read by Kaia Gerber.wav"

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
# VOICE DECISION (2026-07): production uses ROOSTER as the single voice for ALL
# stories (user decision). Other VOICE_* constants are kept so any story can still
# be regenerated with a different voice on demand via the CLI `story:voice`
# override. Flip PRODUCTION_VOICE below to switch the whole batch's voice at once.
# (jabari's VOICE_PROFILES entry is kept for A/B / on-demand use, but rooster has
# no per-voice override so it reads with the age-group pace directly: age2-3 one
# rhythm, age4-5 / age6-7 / all tts_stories the shared _OLDER_PROFILE rhythm.)
PRODUCTION_VOICE = VOICE_ROOSTER
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
                jobs.append((f"{prefix}{txt.stem}", txt, PRODUCTION_VOICE))
    if TTS_STORIES.is_dir():
        for d in sorted(TTS_STORIES.glob("batch_*")):
            if d.is_dir():
                bnum = d.name.replace("batch_", "batch")  # batch_01 -> batch01
                for txt in sorted(d.glob("story_*.txt")):
                    jobs.append((f"{bnum}_{txt.stem}", txt, PRODUCTION_VOICE))
    return jobs


# Active sample set for THIS run: the FIRST N stories of each category, so the
# whole run is just `... --no-cache` with no long positional list. _discover_jobs()
# returns them sorted, so each slice = the lowest-numbered N (e.g. rep 001..005,
# nar 011..015, age4-5 251..255). age6-7 stays excluded until its text is frozen.
# Bump a count to widen that category; set to a large number (e.g. 999) to take
# ALL of it. Each story's audio is routed to its own age-group subfolder under
# OUTPUT_BASE. Run with NO positional arg to write production names (overwrite); a
# positional filter (e.g. `... age4-5_story_251`) still works but tags output _cmd.
REP_SAMPLE_COUNT = 0      # age2-3 REPETITIVE stories (0 = skip this run; kept for later)
NAR_SAMPLE_COUNT = 0      # age2-3 NARRATIVE stories (0 = skip this run; kept for later)
AGE45_SAMPLE_COUNT = 15   # age4-5 stories (from the start)
AGE67_SAMPLE_COUNT = 15   # age6-7 stories (from the start)

# Specific tts_stories (fairy tales) selected by title -> discovered batch label,
# mapped by a content search. Only these batch stories run this time.
TTS_SELECTED_LABELS = [
    "batch01_story_001",  # 灰姑娘 Cinderella
    "batch01_story_003",  # 长发公主 Rapunzel
    "batch01_story_005",  # 美人鱼 The Little Mermaid
    "batch09_story_083",  # 阿里巴巴和四十大盗 Ali Baba and the Forty Thieves
    "batch05_story_042",  # 青蛙王子 The Frog Prince
    "batch07_story_064",  # 神笔马良 Magic Paintbrush (Ma Liang)
    "batch02_story_011",  # 三只熊和金发公主 Goldilocks and the Three Bears
    "batch04_story_038",  # 金斧头银斧头 The Honest Woodcutter (gold/silver axe)
    "batch04_story_034",  # 渔夫和金鱼的故事 The Fisherman and the Fish
    "batch02_story_015",  # 杰克和魔豆 Jack and the Beanstalk
]

_discovered = _discover_jobs()
_by_label = {j[0]: j for j in _discovered}
# THIS RUN: first 15 age4-5 + first 15 age6-7 + the selected tts_stories. age2-3
# is kept in code (counts = 0) so it can be re-enabled later without edits.
JOBS = (
    [j for j in _discovered if j[0].startswith("age2-3_rep")][:REP_SAMPLE_COUNT]
    + [j for j in _discovered if j[0].startswith("age2-3_nar")][:NAR_SAMPLE_COUNT]
    + [j for j in _discovered if j[0].startswith("age4-5")][:AGE45_SAMPLE_COUNT]
    + [j for j in _discovered if j[0].startswith("age6-7")][:AGE67_SAMPLE_COUNT]
    + [_by_label[lbl] for lbl in TTS_SELECTED_LABELS if lbl in _by_label]
)

# --- Level 2 tone selection -------------------------------------------------
# Narration (no quotes) gets an expressive storytelling tone. Dialogue (quoted speech)
# gets an expressive tone; if an attribution verb is present (shouted,
# whispered, laughed, ...) the tone is chosen to match that emotion, otherwise
# a gentle default is used. These parentheticals are VoxCPM2 style hints; the
# model reads them as instructions and does not speak them aloud.
NARRATION_TONE = "(slower pace, expressive storytelling tone, clear pauses)"
DEFAULT_DIALOGUE_TONE = "(gentle, expressive voice)"

# Global pacing hint folded into EVERY non-empty style tag by build_prompt(), so
# the whole story reads slower while keeping the cloned rooster timbre. This is
# model-level pacing (the "Style Control" from the VoxCPM2 cookbook), NOT a
# post-hoc time-stretch like --speed, so it slows delivery without colouring the
# voice. Dial it up if still too fast (e.g. "very slow, deliberate pace") or set
# it to "" to disable. Changing this text changes the sentence cache key, so
# affected sentences auto-regenerate on the next run. Ultra-short lines
# (< MIN_WORDS_FOR_TONE) get no tag at all, so they are unaffected on purpose.
#
# PERMANENT LOCKED MECHANISM (user rule) — the VALUE may be retuned, but the
# MECHANISM must never be deleted and PACE_HINT must never be blank / must always
# contain "slow" (enforced by _assert_pace_hint). It applies to EVERY story and
# EVERY age group.
# CURRENT DECISION (user): ONE uniform pace across ALL ages —
# "gentle, natural pace, only slightly slow, clear pauses". "natural pace" pulls
# the read toward the model's own speed, "only slightly slow" keeps a light brake
# (and satisfies the "slow" lock), "clear pauses" keeps sentence boundaries crisp.
# This REPLACES the earlier per-age split (age2-3 = full slow global, age4-5 =
# lighter override); age4-5 no longer overrides PACE_HINT, both inherit this.
# NOTE: this is a touch faster than the old age2-3 global, so age2-3 speeds up to
# match age4-5 — intended.
# HISTORY / DO NOT REPEAT: "very slow", "very slow, deliberate, unhurried pace",
# and "very slow, calm, measured pace" were all TRIED and REJECTED — they
# over-slowed AND fought the pitch guard. Never retry "very slow".
PACE_HINT = "gentle, natural pace, only slightly slow, clear pauses"


def _assert_pace_hint(value: str) -> None:
    """Enforce the PACE_HINT lock. It can NEVER be disabled (blank) or set to a
    non-slow value; if it ever is, fail loudly here rather than silently ship a
    fast read. Checked at import AND on every per-age apply (apply_age_profile)."""
    if not value or "slow" not in value.lower():
        raise ValueError(
            "PACE_HINT is a locked constant and must never be disabled: it must "
            "be non-empty and contain 'slow'. Refusing to run with "
            f"PACE_HINT={value!r}."
        )


_assert_pace_hint(PACE_HINT)  # fail at import if the locked constant was tampered with

# --- Per-age-group tuning profiles -----------------------------------------
# Short repetitive age2-3 and longer dialogue-heavy age4-5 have different sweet
# spots, so each age group overrides the pace/guard knobs independently. Keys not
# listed in a profile fall back to the defaults snapshotted below.
# apply_age_profile() sets these globals per story (matched by label prefix)
# BEFORE that story is generated, so age2-3 and age4-5 never share or clobber
# each other's pace/guards. Because PACE_HINT is part of the sentence cache key,
# different per-age paces also get separate cache entries automatically.
_TUNABLE_DEFAULTS = {
    "PACE_HINT": PACE_HINT,
    "PITCH_TOLERANCE_SEMITONES": PITCH_TOLERANCE_SEMITONES,
    "DURATION_MIN_GUARD_ENABLED": DURATION_MIN_GUARD_ENABLED,
    "MIN_WORDS_PER_SECOND": MIN_WORDS_PER_SECOND,
    "MIN_DURATION_BASE_S": MIN_DURATION_BASE_S,
    "WORDS_PER_SECOND": WORDS_PER_SECOND,
    "DURATION_BASE_S": DURATION_BASE_S,
    "MAX_CANDIDATE_SEEDS": MAX_CANDIDATE_SEEDS,
    "QUOTE_PACE_GUARD_ENABLED": QUOTE_PACE_GUARD_ENABLED,
    "QUOTE_MIN_WORDS_PER_SECOND": QUOTE_MIN_WORDS_PER_SECOND,
    "QUOTE_MIN_WORDS": QUOTE_MIN_WORDS,
    "ASR_VERIFY_ENABLED": ASR_VERIFY_ENABLED,
    "ASR_MATCH_MIN_RATIO": ASR_MATCH_MIN_RATIO,
    "SENTENCE_PAUSE_S": SENTENCE_PAUSE_S,
    "PARAGRAPH_PAUSE_S": PARAGRAPH_PAUSE_S,
    "NARRATION_TONE": NARRATION_TONE,
}

# Only list keys that DIFFER from the defaults. Both ages start at the original
# known-good config (empty = defaults); tune each age here as you audition it.
# Example: to make age4-5 read slower (the setup the guard-report validated),
# fill its dict like:
#     "age4-5": {
#         "PACE_HINT": "slow, unhurried pace",
#         "PITCH_TOLERANCE_SEMITONES": 4.5,
#         "DURATION_MIN_GUARD_ENABLED": True,
#         "MIN_WORDS_PER_SECOND": 0.28,
#     },
# Story-read pace is a THREE-LEVEL gradient (user decision 2026-07):
#   age2-3 (slowest)  <  age4-5 (medium)  <  age6-7 (fastest)
# The two "older tier" profiles below split what used to be one shared profile.
# age4-5 reads a touch slower than age6-7 via a lighter pace hint ("only slightly
# slow" vs "barely slow") AND slightly longer pauses. tts_stories (batch...) are
# MIXED-AGE (some are toddler-friendly, some older), so they follow the MIDDLE
# (age4-5) pace as a safe compromise. Levers are weak on their own (see repo
# notes): the clearest speed difference comes from the pauses, the PACE_HINT is a
# best-effort nudge. Retune each tier here; both keep "slow" (locked-constant rule).
_AGE67_PROFILE = {          # fastest older tier (= the old shared _OLDER_PROFILE)
    "PITCH_TOLERANCE_SEMITONES": 5.0,
    "PACE_HINT": "natural, flowing pace, barely slow, clear pauses",
    "NARRATION_TONE": "(warm, expressive storytelling, natural rise and fall)",
    "PARAGRAPH_PAUSE_S": 0.9,
}
_AGE45_PROFILE = {          # MEDIUM: slower than 6-7 (lighter brake + longer pauses)
    "PITCH_TOLERANCE_SEMITONES": 5.0,
    "PACE_HINT": "gentle, natural pace, only slightly slow, clear pauses",
    "NARRATION_TONE": "(warm, expressive storytelling, natural rise and fall)",
    "SENTENCE_PAUSE_S": 0.55,
    "PARAGRAPH_PAUSE_S": 1.0,
}
AGE_PROFILES: dict[str, dict] = {
    # Pitch tolerance is looser than the 3.0 default: the guard-reports show the
    # model routinely lands short lines at 3.0-3.5st from the reference median,
    # so 3.0 forced seed-exhaustion + kept-at-3.0 compromises anyway (see the
    # story_011 Coco report: 'Mom nodded' / 'Coco gave a soft sigh' burned all 8
    # seeds and shipped 3.0 regardless). 3.5 accepts those first-try; 4.0 for
    # age4-5 whose longer dialogue clusters even higher.
    "age2-3": {
        # Pace was confirmed good, so the pace knobs (MIN_WORDS_PER_SECOND,
        # pauses, the inherited global PACE_HINT) are LEFT ALONE. The only
        # expressiveness change is a warmer narration tone + a small pitch-
        # tolerance bump so a little emotion/rhythm gets through without speeding
        # up. This tone is GENTLER than age4-5's (toddlers still want a steady,
        # sing-song read, not a big dramatic arc): "warm, gentle" keeps it soft,
        # "a little playful, natural rise and fall" adds the missing lilt.
        "NARRATION_TONE": "(warm, gentle storytelling, a little playful, natural rise and fall)",
        # 3.5 -> 4.0: let mildly-excited/soothing lines (which sit a touch above/
        # below the median) pass instead of getting reseeded back to flat. Still
        # well under age4-5's 5.0, so the toddler read stays close to the
        # reference and never lurches off-register.
        "PITCH_TOLERANCE_SEMITONES": 4.0,
        # PACE_HINT ("slow, unhurried pace") is now the GLOBAL default, so it is
        # NOT repeated here — age2-3 inherits it. This override only sets the
        # too-fast floor. It was 0.34 (to force a slower, even take), but that
        # rejected the model's natural ~0.40 spw takes and re-rolled to slower
        # seeds — which read a touch too deliberate AND caused kept:fast
        # compromises on 7-8 word action lines (see the story_012 report:
        # 'The bunny bounced and did a flop.' burned all 8 seeds at ~0.40 spw).
        # 0.30 lets those slightly-faster, good-pitch takes pass first-try, so the
        # average pace nudges up a little without touching the locked PACE_HINT.
        "MIN_WORDS_PER_SECOND": 0.30,
        # New age2-3 story structure -> longer, more deliberate breaks: 0.7s
        # between sentences (single \n) and 1.2s between paragraphs (\n\n).
        # age2-3 ONLY; other ages keep the 0.45/0.80 defaults.
        "SENTENCE_PAUSE_S": 0.7,
        "PARAGRAPH_PAUSE_S": 1.2,
    },
    # THREE-LEVEL pace gradient (user 2026-07): age2-3 slowest, age4-5 MEDIUM,
    # age6-7 fastest. tts_stories (batch...) are mixed-age -> follow the MIDDLE
    # age4-5 pace. `--fast67` can still swap age6-7 to variant B for an A/B.
    "age4-5": _AGE45_PROFILE,
    "age6-7": _AGE67_PROFILE,
    "batch": _AGE45_PROFILE,
}

# --- age6-7 pace A/B ---------------------------------------------------------
# Variant B ("slightly faster"): "conversational" pulls toward natural speech
# pace and the pauses tighten to the shorter defaults. Toggled per-run with the
# --fast67 flag (see main()); when on, apply_age_profile() overrides the age6-7
# PACE_HINT + pauses with these and the output filename gets a `_fast` tag so it
# sits next to the variant-A take for a direct listen.
AGE67_PACE_HINT_FAST = "natural, conversational pace, barely slow, clear pauses"
AGE67_SENTENCE_PAUSE_FAST_S = 0.40
AGE67_PARAGRAPH_PAUSE_FAST_S = 0.80
AGE67_PACE_FAST = False   # flipped True by --fast67
# --as6-7 (A/B helper): force ANY story (incl. batch/tts) to read with the age6-7
# (fastest) profile for this run, output tagged `_as67` so it sits beside the
# story's normal-profile take. Lets you compare a story at its default pace (e.g.
# tts -> middle age4-5) vs the fast age6-7 pace on the SAME text.
FORCE_AGE67 = False

# Very short lines (fewer than this many words) get NO tone parenthetical.
# Measured cause of hallucination: on ultra-short lines like "Goodnight, Cow."
# (2 words) the (style) prefix occasionally makes VoxCPM2 append 5-7s of
# unrelated speech. A/B probe: tone = 2 hallucinations / 72 gens (max 7.2s);
# plain = 0 / 72 (max 2.4s). So suppress the prefix below this word count.
MIN_WORDS_FOR_TONE = 4

# Attribution verb / adverb -> tone. Matched as whole words, case-insensitive.
# Each tag is kept to ~2-3 vivid, picture-rich descriptors (NOT a pile): build_prompt
# folds PACE_HINT in on top, so a 2-3 word emotion + a 3-4 word pace cue already
# fills the tag; more would over-stack and muddy the read (cookbook: "spoil the broth").
DIALOGUE_EMOTION = {
    # excited / loud — bright, high-energy exclamations
    "shouted": "(loud, excited voice)",
    "cried": "(bright, excited voice)",
    "yelled": "(loud, excited voice)",
    "exclaimed": "(surprised, delighted voice)",
    "shrieked": "(high, startled voice)",
    "called": "(bright, calling voice)",
    "announced": "(proud, clear voice)",
    "gasped": "(surprised, breathless voice)",
    "cheered": "(joyful, celebrating voice)",
    "boomed": "(big, booming voice)",
    # New (from tts_stories scan): more loud/excited attributions.
    "roared": "(loud, roaring voice)",
    "bellowed": "(big, booming voice)",
    "squealed": "(high, delighted voice)",
    "loudly": "(loud, strong voice)",
    # soft / gentle — hushed and tender
    "whispered": "(soft, hushed whisper)",
    "murmured": "(soft, gentle murmur)",
    "softly": "(soft, tender voice)",
    "quietly": "(soft, tender voice)",
    "gently": "(soft, tender voice)",
    "sighed": "(soft, wistful voice)",
    "whimpered": "(small, trembling voice)",
    "squeaked": "(tiny, squeaky voice)",
    # New (from tts_stories scan): soft/calm attributions.
    "breathed": "(soft, breathy voice)",
    "calmly": "(calm, steady voice)",
    # New (from First100 scan): soft, contented attribution.
    "purred": "(soft, content voice)",
    # stern / angry — low and sharp
    "growled": "(low, stern voice)",
    "snapped": "(sharp, irritated voice)",
    "demanded": "(firm, insistent voice)",
    "warned": "(serious, cautioning voice)",
    "hissed": "(sharp, hushed voice)",
    # New (from tts_stories scan): sharp/mocking attributions.
    "barked": "(sharp, commanding voice)",
    "sneered": "(sharp, mocking voice)",
    # happy — warm and giggly
    "laughed": "(warm, giggly voice)",
    "giggled": "(light, giggly voice)",
    "chuckled": "(warm, amused voice)",
    "smiled": "(warm, bright voice)",
    "happily": "(bright, cheerful voice)",
    "cheerfully": "(bright, cheerful voice)",
    "teased": "(playful, teasing voice)",
    # New (from tts_stories scan): bright, chirpy attribution.
    "chirped": "(bright, cheery voice)",
    # New (from First100 scan): bright manner adverb.
    "brightly": "(bright, cheerful voice)",
    # proud / warm / kind / curious
    "proudly": "(proud, beaming voice)",
    "warmly": "(warm, tender voice)",
    "kindly": "(warm, gentle voice)",
    "shyly": "(soft, shy voice)",
    "slowly": "(slow, thoughtful voice)",
    "firmly": "(firm, steady voice)",
    "wondered": "(curious, wondering voice)",
    # New (from tts_stories scan): proud/warm/earnest/polite attributions.
    "declared": "(bold, proud voice)",
    "promised": "(warm, earnest voice)",
    "agreed": "(warm, agreeable voice)",
    "admitted": "(soft, sheepish voice)",
    "politely": "(polite, gentle voice)",
    "thoughtfully": "(thoughtful, reflective voice)",
    # New (from First100 scan): explaining/grateful/brave attributions.
    "explained": "(clear, explaining voice)",
    "gratefully": "(warm, grateful voice)",
    "bravely": "(brave, steady voice)",
    # sad / afraid — tender or trembling
    "sobbed": "(sad, tearful voice)",
    "wailed": "(distressed, tearful voice)",
    "trembled": "(frightened, trembling voice)",
    "begged": "(pleading, desperate voice)",
    "groaned": "(weary, grumbling voice)",
    # New (from tts_stories scan): sad manner adverb.
    "sadly": "(soft, sad voice)",
    # New (from First100 scan): worried manner adverb.
    "worriedly": "(worried, anxious voice)",
    # --- New (2026-07 First100 + tts_stories emotion scan) -------------------
    # -ly manner-of-speaking adverbs found next to quotes in the stories:
    "eagerly": "(eager, keen voice)",
    "curiously": "(curious, wondering voice)",
    "anxiously": "(anxious, worried voice)",
    "hopefully": "(hopeful, wishful voice)",
    "excitedly": "(bright, excited voice)",
    "nervously": "(nervous, uneasy voice)",
    "fearfully": "(frightened, fearful voice)",
    "earnestly": "(earnest, heartfelt voice)",
    "urgently": "(urgent, pressing voice)",
    "honestly": "(honest, sincere voice)",
    "simply": "(plain, matter-of-fact voice)",
    "hungrily": "(eager, hungry voice)",
    "helplessly": "(helpless, pleading voice)",
    "seriously": "(serious, earnest voice)",
    "sweetly": "(sweet, tender voice)",
    "angrily": "(angry, sharp voice)",
    "weakly": "(weak, faint voice)",
    "longingly": "(longing, wistful voice)",
    "slyly": "(sly, sneaky voice)",
    "grandly": "(grand, proud voice)",
    "steadily": "(steady, calm voice)",
    "greedily": "(greedy, grasping voice)",
    "sleepily": "(sleepy, drowsy voice)",
    "joyfully": "(joyful, delighted voice)",
    "encouragingly": "(warm, encouraging voice)",
    "confidently": "(confident, assured voice)",
    "gravely": "(grave, solemn voice)",
    "smoothly": "(smooth, easy voice)",
    "admiringly": "(admiring, warm voice)",
    "lazily": "(lazy, drowsy voice)",
    "sheepishly": "(soft, sheepish voice)",
    "miserably": "(miserable, sad voice)",
    "mysteriously": "(mysterious, secretive voice)",
    "wisely": "(wise, knowing voice)",
    "uneasily": "(uneasy, nervous voice)",
    "thankfully": "(warm, grateful voice)",
    "wearily": "(weary, tired voice)",
    "unhappily": "(soft, unhappy voice)",
    "approvingly": "(warm, approving voice)",
    "impatiently": "(impatient, hurried voice)",
    "tenderly": "(tender, loving voice)",
    # -ed speech / vocalization verbs used as attributions (incl. talking animals):
    "huffed": "(annoyed, huffing voice)",
    "yelped": "(sharp, startled voice)",
    "puffed": "(breathy, puffing voice)",
    "beamed": "(warm, beaming voice)",
    "boasted": "(proud, boastful voice)",
    "scoffed": "(scornful, dismissive voice)",
    "commanded": "(firm, commanding voice)",
    "honked": "(loud, honking voice)",
    "neighed": "(loud, neighing voice)",
    "brayed": "(loud, braying voice)",
}
# Precompiled word patterns for the emotion cues.
_EMOTION_PATTERNS = [
    (re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE), tone)
    for word, tone in DIALOGUE_EMOTION.items()
]

# Scenario/mood-aware NARRATION tone. Narration lines (no quotes) otherwise all
# get the single per-age NARRATION_TONE, which reads evenly no matter what is
# happening. This picks a mood-matched narration tone from content keywords so an
# action beat sounds livelier, a magical beat fuller of wonder, a sad beat softer,
# a silly beat more playful. First match wins (list order = priority); no match
# falls back to NARRATION_TONE. Each tag is kept to 2-3 descriptors so build_prompt
# can still fold PACE_HINT in without over-stacking. Toggle NARRATION_MOOD_ENABLED.
# Applies to ALL ages (age2-3 too: an action/silly line reading livelier suits the
# livelier direction); flip the toggle off to go back to one flat narration tone.
NARRATION_MOOD_ENABLED = True
_NARRATION_MOODS = [
    ("sad", re.compile(
        r"\b(sad|sadly|cried|crying|tears|tearful|alone|lonely|lost|missed|"
        r"sorry|sniffl\w*|wept|sighed)\b", re.IGNORECASE),
     "(gentle, tender, wistful storytelling)"),
    ("silly", re.compile(
        r"\b(silly|giggl\w*|funny|goofy|wobbl\w*|wiggl\w*|jiggl\w*|flop\w*|"
        r"tumbl\w*|bounc\w*|splat)\b", re.IGNORECASE),
     "(playful, giggly storytelling)"),
    ("wonder", re.compile(
        r"\b(glow\w*|sparkl\w*|shimmer\w*|shone|shine|magic\w*|star|stars|moon|"
        r"moonlight|rainbow|glitter\w*|twinkl\w*|wonder\w*|gleam\w*|dream\w*)\b",
        re.IGNORECASE),
     "(warm, wonder-filled storytelling)"),
    ("suspense", re.compile(
        r"\b(dark|crept|creep\w*|tiptoe\w*|hush\w*|shadow\w*|waited|listen\w*|"
        r"silent\w*|peek\w*)\b", re.IGNORECASE),
     "(hushed, curious storytelling)"),
    ("action", re.compile(
        r"\b(ran|race\w*|raced|jump\w*|leap\w*|leapt|dash\w*|splash\w*|flew|"
        r"zoom\w*|burst|chas\w*|rush\w*|hopp\w*|climb\w*|dove|dived|flapp\w*|"
        r"scrambl\w*)\b", re.IGNORECASE),
     "(bright, lively, energetic storytelling)"),
]


def narration_tone_for(sentence: str) -> str:
    """Mood-matched narration tone (falls back to the per-age NARRATION_TONE)."""
    if NARRATION_MOOD_ENABLED:
        for _name, pattern, tone in _NARRATION_MOODS:
            if pattern.search(sentence):
                return tone
    return NARRATION_TONE


# Content cues for dialogue that has NO attribution verb (e.g. a bare
# `"Wow, look at that!"` with no `said/asked/...`). Instead of always falling
# back to the flat DEFAULT_DIALOGUE_TONE, infer the emotion from the QUOTE's own
# punctuation / keywords so plain direct speech isn't read lifeless. Checked in
# order: scared/urgent and sad keywords FIRST (so `"Help!"` reads worried and
# `"I'm sorry."` reads tender rather than merely "excited" from the `!`), then
# `!` -> excited, `?` -> curious, else the gentle default. Cues are matched on
# the quoted span only (not the attribution) so narration words don't leak in.
_DIALOGUE_SCARED_RE = re.compile(
    r"\b(help|oh no|watch out|look out|run|careful|stop|no no|uh oh)\b", re.IGNORECASE)
_DIALOGUE_SAD_RE = re.compile(
    r"\b(sorry|sad|afraid|scared|miss|missed|poor|please|don'?t cry|i can'?t)\b",
    re.IGNORECASE)


def _dialogue_fallback_tone(sentence: str) -> str:
    """Emotion tone for a quote with no attribution verb (punctuation/keywords)."""
    quoted = " ".join(re.findall(r'"([^"]*)"', sentence)) or sentence
    if _DIALOGUE_SCARED_RE.search(quoted):
        return "(worried, urgent voice)"
    if _DIALOGUE_SAD_RE.search(quoted):
        return "(soft, tender voice)"
    if "!" in quoted:
        return "(bright, excited voice)"
    if "?" in quoted:
        return "(curious, wondering voice)"
    return DEFAULT_DIALOGUE_TONE


def tone_for(sentence: str) -> str:
    """Pick a VoxCPM2 style tone for a single sentence (Level 2)."""
    # Ultra-short lines hallucinate when given a (style) prefix -> no tone.
    if len(re.findall(r"[A-Za-z0-9']+", sentence)) < MIN_WORDS_FOR_TONE:
        return ""
    if '"' not in sentence:
        return narration_tone_for(sentence)
    for pattern, tone in _EMOTION_PATTERNS:
        if pattern.search(sentence):
            return tone
    # No attribution verb matched -> infer from the quote's punctuation/keywords
    # instead of always returning the flat gentle default.
    return _dialogue_fallback_tone(sentence)


def build_prompt(sentence: str, narration_only: bool = False) -> str:
    """Exact text handed to the model: style tag + SPACE + sentence.

    The space matters: gluing the closing paren onto the first word (")Cow")
    makes VoxCPM swallow the leading consonant. When TONE_ENABLED is False
    (--notone) or the line is too short for a tag, the bare sentence is used.
    `narration_only` (paragraph / direct story read) uses the ONE fixed
    PARAGRAPH_NARRATION_TONE verbatim - no mood-match, no per-age/lively override,
    and NO PACE_HINT folding (the tone already carries its own pace).
    """
    if narration_only:
        tone = PARAGRAPH_NARRATION_TONE if TONE_ENABLED else ""
        return f"{tone} {sentence}" if tone else sentence
    tone = tone_for(sentence) if TONE_ENABLED else ""
    if tone and PACE_HINT and tone.endswith(")"):
        # Fold the global pacing hint INSIDE the parenthetical style tag, e.g.
        # "(gentle, expressive voice)" -> "(gentle, expressive voice, slow, ...)".
        tone = f"{tone[:-1].rstrip()}, {PACE_HINT})"
    return f"{tone} {sentence}" if tone else sentence
# ---------------------------------------------------------------------------


def _word_count(sentence: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", sentence))


def expected_max_seconds(sentence: str) -> float:
    """Loose upper bound on how long a sane reading of this sentence should be.

    Ellipsis '…' and em-dash '—' are INTENTIONAL long pauses (the word budget
    doesn't know about them), so each one adds PAUSE_MARK_ALLOWANCE_S of headroom
    — otherwise e.g. 'Munch… munch… munch.' is wrongly flagged as too long.
    """
    pause_marks = sentence.count("\u2026") + sentence.count("\u2014")
    return max(
        DURATION_MIN_LIMIT_S,
        _word_count(sentence) * WORDS_PER_SECOND + DURATION_BASE_S
        + pause_marks * PAUSE_MARK_ALLOWANCE_S,
    )


def expected_min_seconds(sentence: str) -> float:
    """Lower bound on a non-rushed reading; shorter than this reads as hurried."""
    return _word_count(sentence) * MIN_WORDS_PER_SECOND + MIN_DURATION_BASE_S


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


def _should_end_falling(sentence: str) -> bool:
    """True if this sentence should END on a falling (declarative) pitch.

    Statements ending in "." should fall. Questions ("?"), exclamations ("!") and
    suspense / ellipsis ("..." / "…") are EXEMPT — a rise there is natural or
    intended. Trailing quotes are ignored so `"I am hungry."` still counts as a
    statement.
    """
    t = sentence.rstrip()
    while t and t[-1] in '"\u201d\u2019\'\u300d\u300f':  # strip trailing quotes
        t = t[:-1].rstrip()
    if not t:
        return False
    if t.endswith("\u2026") or t.endswith("..."):
        return False
    return t.endswith(".")


def _end_pitch_slope(audio: np.ndarray, sample_rate: int) -> float:
    """Linear pitch slope (Hz per frame) over the last ENDING_WINDOW_S of audio.

    Positive = the pitch rises at the end (question-like); negative = it falls
    (declarative). Returns 0.0 (neutral) if the pitch can't be tracked.

    Trailing near-silence is trimmed FIRST: narration usually leaves a pause /
    breath after the last word, which would otherwise fill the window with blank
    and read as a false 0.0 (missing a real rise). Trimming makes the window sit
    on the actual spoken tail so narration endings measure as reliably as dialogue.
    """
    if audio.size < int(0.2 * sample_rate):
        return 0.0
    try:
        trimmed, _idx = librosa.effects.trim(
            audio.astype(np.float32), top_db=ENDING_SILENCE_TOP_DB
        )
    except Exception:
        trimmed = audio
    if trimmed.size < int(0.15 * sample_rate):
        trimmed = audio  # trim removed too much (very short/quiet) -> use as-is
    tail = trimmed[-int(ENDING_WINDOW_S * sample_rate):]
    try:
        f0, _voiced, _prob = librosa.pyin(
            tail.astype(np.float32),
            fmin=PITCH_FMIN_HZ,
            fmax=PITCH_FMAX_HZ,
            sr=sample_rate,
        )
    except Exception:
        return 0.0
    voiced = f0[np.isfinite(f0)]
    if voiced.size < 4:
        return 0.0
    # Drop pyin octave-jump outliers (occasional doubled/halved frames) that would
    # otherwise spike a short-window slope: keep frames within ~7 semitones of the
    # tail median.
    med = float(np.median(voiced))
    if med > 0:
        keep = np.abs(12.0 * np.log2(voiced / med)) <= 7.0
        voiced = voiced[keep]
        if voiced.size < 4:
            return 0.0
    n = voiced.size
    x = np.arange(n, dtype=np.float64)
    slope_all = float(np.polyfit(x, voiced, 1)[0])
    # NOTE (2026-07): an earlier version also fit the LAST HALF of the tail and
    # took max(slope_all, slope_end) to catch a late final-syllable uptick. On a
    # real story that made the guard ~5x more trigger-happy (33 vs 6 reseeded
    # sentences on story_003) — the last-half slope is noisy and flagged many
    # natural +1..+2 endings — WITHOUT fixing the target case ("...whenever he
    # could." measured only +0.1 either way, i.e. it is near-FLAT, not truly
    # rising). So we use the whole-tail slope only: it still catches the genuine
    # rises (+2 and up) at a fraction of the thrash. Near-flat endings ship as-is.
    return slope_all


# --- Sentence cache helpers ------------------------------------------------
def _sentence_cache_path(voice_tag: str, sentence: str, narration_only: bool = False) -> Path | None:
    """Disk path for a cached sentence, keyed by voice + prompt + gen params.

    The generation-affecting parameters (and the tone-decorated prompt) are all
    folded into the hash, so changing any of them yields a different key and the
    stale entry is simply missed rather than silently reused. `narration_only`
    flows into build_prompt, so paragraph-read clips get their own cache entries.
    """
    if not SENTENCE_CACHE_ENABLED:
        return None
    prompt = build_prompt(sentence, narration_only)
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


def _guard_log(line: str) -> None:
    """Append one per-seed guard diagnostic line to the guard-report file (if on)."""
    if GUARD_LOG_PATH is None:
        return
    try:
        with GUARD_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
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
                      voice_tag: str = "voice",
                      narration_only: bool = False,
                      no_guard: bool = False) -> tuple[np.ndarray, list[str]]:
    """Generate one sentence, early-stopping on the first clean candidate.

    Returns (wav, issues). `issues` is empty when a clean candidate was found;
    otherwise it lists the defect reasons of the best-scoring fallback so the
    caller can record them in the review queue.

    A disk cache keyed by (voice, prompt, gen params) short-circuits everything:
    a cache hit returns immediately without running the model or any guard.

    `no_guard` (paragraph read mode) does the plain thing: ONE generation at the
    fixed base seed, NO duration/pitch/quality guards and NO seed retries.

    Otherwise seeds are tried in order (SEED, SEED+1, ...). A candidate is accepted
    immediately when duration, pitch, clipping, onset-click, internal-silence and
    timbre (spectral centroid vs the reference) are all within tolerance. Normal
    sentences pass on the first seed; only defective ones pay for extra
    candidates. If none pass, the lowest-penalty candidate is kept.
    """
    # --- cache lookup (skips model + guards entirely on hit) ---
    cache_path = _sentence_cache_path(voice_tag, sentence, narration_only)
    if (SENTENCE_CACHE_ENABLED and not _CACHE_BYPASS and not _CACHE_REFRESH
            and cache_path is not None and cache_path.exists()):
        try:
            cached, _sr = sf.read(str(cache_path), dtype="float32", always_2d=False)
            if cached.ndim > 1:
                cached = cached.mean(axis=1)
            return np.asarray(cached, dtype=np.float32), []
        except Exception:
            pass  # unreadable cache -> fall through and regenerate

    prompt = build_prompt(sentence, narration_only)

    # no_guard (paragraph read): the plain "read it once" behaviour the user
    # wants -> a SINGLE generation at the fixed base seed, with NO duration/pitch/
    # quality guards and NO seed retries. Tone + pace are still in the prompt;
    # nothing is measured or re-rolled, so every unit uses the same seed.
    if no_guard:
        wav = np.asarray(
            model.generate(
                text=prompt,
                reference_wav_path=str(ref_path),
                cfg_value=CFG_VALUE,
                inference_timesteps=INFERENCE_TIMESTEPS,
                denoise=DENOISE_REFERENCE,
                seed=SEED,
            ),
            dtype=np.float32,
        )
        wc = _word_count(sentence)
        dur = len(wav) / sample_rate
        spw = dur / wc if wc else 0.0
        print(f"    no-guard: seed {SEED}  dur={dur:.2f}s spw={spw:.2f} words={wc}  "
              f'"{sentence[:36]}"', file=sys.stderr)
        _guard_log(f"seed={SEED} result=no-guard dur={dur:.2f}s spw={spw:.2f} words={wc} text={sentence!r}")
        if SENTENCE_CACHE_ENABLED and not _CACHE_BYPASS and cache_path is not None:
            _write_cache(cache_path, wav, sample_rate)
        return wav, []

    limit = expected_max_seconds(sentence)
    min_limit = expected_min_seconds(sentence) if DURATION_MIN_GUARD_ENABLED else 0.0
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
    best_measure = ""
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
        dur_under = max(0.0, min_limit - dur) if DURATION_MIN_GUARD_ENABLED else 0.0

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
        if DURATION_MIN_GUARD_ENABLED and min_limit > 0 and dur < min_limit:
            reasons.append(f"fast {dur:.1f}s<{min_limit:.1f}s")
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

        # Falling-ending guard: a declarative (period-ended) sentence should not
        # land on a rising, question-like pitch. Measure the tail slope and reject
        # a clearly-rising take so another seed is tried (kept: the most falling).
        end_slope = 0.0
        check_ending = FALLING_ENDING_GUARD_ENABLED and _should_end_falling(sentence)
        if check_ending:
            end_slope = _end_pitch_slope(wav, sample_rate)
            if end_slope > ENDING_RISE_SLOPE_MAX:
                reasons.append(f"rising {end_slope:+.1f}")

        # Intra-sentence dialogue checks. With ASR on, Whisper measures the
        # quoted span's pace AND verifies intelligibility (mumble/hallucination);
        # otherwise the silence-split heuristic estimates only the pace.
        quote_spw = None
        asr_ratio = None
        _qt = _split_quote_tail(sentence)
        if _qt is not None and _word_count(_qt[0]) >= QUOTE_MIN_WORDS:
            _qw = _word_count(_qt[0])
            if ASR_VERIFY_ENABLED:
                asr_ratio, quote_spw = _asr_quote_check(wav, sample_rate, sentence, _qw)
                if asr_ratio is not None and asr_ratio < ASR_MATCH_MIN_RATIO:
                    reasons.append(f"asr-mismatch {asr_ratio:.2f}<{ASR_MATCH_MIN_RATIO}")
                if quote_spw is not None and quote_spw < QUOTE_MIN_WORDS_PER_SECOND:
                    reasons.append(f"quote-fast {quote_spw:.2f}<{QUOTE_MIN_WORDS_PER_SECOND}")
            elif QUOTE_PACE_GUARD_ENABLED:
                quote_spw = _quote_speaking_rate(wav, sample_rate, _qw, _word_count(sentence))
                if quote_spw is not None and quote_spw < QUOTE_MIN_WORDS_PER_SECOND:
                    reasons.append(f"quote-fast {quote_spw:.2f}<{QUOTE_MIN_WORDS_PER_SECOND}")

        # Per-seed measurement (logged for EVERY candidate, incl. the accepted
        # one) so pace can be inspected/calibrated, not just pitch.
        wc = _word_count(sentence)
        spw = dur / wc if wc else 0.0
        measure = (f"dur={dur:.2f}s spw={spw:.2f} pitch={semi:.1f}st "
                   f"words={wc} limit={limit:.1f}s min={min_limit:.1f}s")
        if check_ending:
            measure += f" end={end_slope:+.1f}"
        if quote_spw is not None:
            measure += f" qspw={quote_spw:.2f}"
        if asr_ratio is not None:
            measure += f" asr={asr_ratio:.2f}"

        if not reasons:
            print(f"    guard: seed {SEED + i} OK  {measure}  \"{sentence[:36]}\"",
                  file=sys.stderr)
            _guard_log(f"seed={SEED + i} result=OK {measure} text={sentence!r}")
            if SENTENCE_CACHE_ENABLED and not _CACHE_BYPASS and cache_path is not None:
                _write_cache(cache_path, wav, sample_rate)
            return wav, []

        score = (
            dur_over
            + dur_under
            + semi
            + clip_frac * 1000.0
            + (5.0 if onset_click else 0.0)
            + max(0.0, internal_sil - INTERNAL_SILENCE_MAX_S)
            + max(0.0, centroid_ratio - 1.0)
            + (max(0.0, QUOTE_MIN_WORDS_PER_SECOND - quote_spw) if quote_spw is not None else 0.0)
            + (max(0.0, ASR_MATCH_MIN_RATIO - asr_ratio) * 3.0 if asr_ratio is not None else 0.0)
            + ENDING_RISE_PENALTY_WEIGHT * max(0.0, end_slope - ENDING_RISE_SLOPE_MAX)
        )
        if score < best_score:
            best, best_score, best_reasons, best_measure = wav, score, reasons, measure
        print(
            f"    guard: seed {SEED + i} rejected ({', '.join(reasons)})  {measure}  "
            f'"{sentence[:36]}" - trying next',
            file=sys.stderr,
        )
        _guard_log(f"seed={SEED + i} result=reject:{','.join(reasons)} {measure} text={sentence!r}")
    print(
        f"    guard: kept best candidate (score {best_score:.2f}; "
        f"{', '.join(best_reasons)})  {best_measure}  after {n_seeds} seeds",
        file=sys.stderr,
    )
    _guard_log(f"seed=BEST result=kept:{','.join(best_reasons)} {best_measure} text={sentence!r}")
    if (SENTENCE_CACHE_ENABLED and not _CACHE_BYPASS
            and cache_path is not None and best is not None):
        _write_cache(cache_path, best, sample_rate)
    return best, best_reasons


def safe_name(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return cleaned or "voice"


# Speech verbs that mark a dialogue attribution tag (`said Sam.`, `he called.`).
_ATTR_SPEECH_VERBS = {
    "said", "asked", "called", "cried", "replied", "whispered", "shouted",
    "added", "answered", "murmured", "giggled", "laughed", "announced",
    "growled", "warned", "sighed", "yelled", "gasped", "begged", "chuckled",
    "exclaimed", "screamed", "snapped", "hissed", "muttered", "wondered",
    "smiled", "grinned", "nodded",
    # Added alongside the enriched DIALOGUE_EMOTION so their attribution tails
    # (e.g. `"Hooray!" they cheered.`) also merge back onto the quote instead of
    # generating as tiny, mumble-prone orphan clips.
    "cheered", "boomed", "whimpered", "squeaked", "teased", "groaned",
    # New speech verbs found in the tts_stories scan (kept in sync with
    # DIALOGUE_EMOTION) so their tails (e.g. `"Run!" the dog barked.`) merge too.
    "roared", "bellowed", "squealed", "breathed", "barked", "sneered",
    "chirped", "declared", "promised", "agreed", "admitted",
    # New speech verbs from the First100 scan.
    "explained", "purred",
    # New speech / vocalization verbs from the 2026-07 First100 + tts scan (kept
    # in sync with DIALOGUE_EMOTION so their tails merge onto the quote).
    "huffed", "yelped", "puffed", "beamed", "boasted", "scoffed",
    "commanded", "honked", "neighed", "brayed",
}


def _is_attribution_tail(chunk: str) -> bool:
    """True if `chunk` is a short dialogue attribution tag that belongs to the
    PRECEDING quote — e.g. `said Sam.`, `he called.`, `the voice asked.`,
    `Bell asked softly.`, `the little girl cried.`.

    The sentence splitter breaks after `!"`/`?"`, so `"Help!" he called.` becomes
    two chunks and the attribution generates as its own tiny (mumble-prone) clip.
    Such a tail is merged back onto the quote it belongs to (see
    `_merge_attribution_tails`). Only fires when the tail is short, quote-free,
    and a speech verb sits at the FIRST word (inversion `said the girl`), the
    LAST word (`the voice asked`), or second-to-last (trailing adverb
    `Bell asked softly`). Verb in the middle (`she answered the door`) is NOT an
    attribution and is left alone.
    """
    if '"' in chunk:
        return False
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", chunk)]
    if not (1 <= len(words) <= 6):
        return False
    if words[0] in _ATTR_SPEECH_VERBS:                 # "said Sam."
        return True
    if words[-1] in _ATTR_SPEECH_VERBS:                # "the voice asked."
        return True
    if len(words) >= 2 and words[-2] in _ATTR_SPEECH_VERBS:  # "Bell asked softly."
        return True
    return False


def _merge_attribution_tails(sentences: list[str]) -> list[str]:
    """Merge a short trailing attribution tag back onto its quote so `"Help!"`
    and `he called.` render as ONE clean chunk instead of a tiny orphan clip.
    Only merges when the previous chunk ends in a closing double-quote AND the
    merged chunk stays short (<= MERGE_MAX_WORDS): a long quote + tail would form
    an over-long utterance, so we leave those split."""
    out: list[str] = []
    for s in sentences:
        if (out and out[-1].rstrip().endswith('"') and _is_attribution_tail(s)
                and _word_count(out[-1]) + _word_count(s) <= MERGE_MAX_WORDS):
            out[-1] = out[-1].rstrip() + " " + s
        else:
            out.append(s)
    return out


def _split_quote_tail(sentence: str) -> tuple[str, str] | None:
    """Split `sentence` into (quote_part, attribution_tail) if it ends with a
    short attribution tag AFTER a closing double-quote, else None.

    e.g. '"But I am hungry," said Coco.' -> ('"But I am hungry,"', 'said Coco.')
    Used by the quote-pace guard to score the quoted span on its own.
    """
    idx = sentence.rfind('"')
    if idx == -1:
        return None
    tail = sentence[idx + 1:].strip()
    if not tail or not _is_attribution_tail(tail):
        return None
    return sentence[: idx + 1], tail


def _quote_speaking_rate(wav: np.ndarray, sample_rate: int,
                         quote_words: int, total_words: int) -> float | None:
    """Seconds-per-word of just the quoted span of a quote+attribution clip.

    Splits the audio at the internal pause nearest the expected quote/tail word
    boundary and measures the quoted span alone. Returns None when it can't
    confidently split (too short, no interior pause) — a deliberate false-negative
    so the guard never fires on a bad split.
    """
    n = wav.size
    if n == 0 or quote_words <= 0 or total_words <= 0:
        return None
    frame = max(1, int(0.02 * sample_rate))
    count = n // frame
    if count < 5:
        return None
    block = wav[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
    voiced = rms >= SILENCE_RMS_FLOOR
    idx = np.where(voiced)[0]
    if idx.size == 0:
        return None
    first, last = int(idx[0]), int(idx[-1])
    if last - first < 3:
        return None
    # Interior silent runs = candidate quote/tail boundaries.
    gaps: list[tuple[int, int]] = []
    i = first + 1
    while i < last:
        if not voiced[i]:
            j = i
            while j < last and not voiced[j]:
                j += 1
            gaps.append((i, j))
            i = j
        else:
            i += 1
    if not gaps:
        return None
    # Expected boundary position by word fraction; pick the closest interior gap.
    expected = first + (last - first) * (quote_words / total_words)
    boundary = min(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - expected))
    quote_dur = (boundary[0] - first) * frame / sample_rate
    if quote_dur <= 0.2:
        return None
    return quote_dur / quote_words


# --- ASR verification (Whisper) helpers ------------------------------------
_ASR_STATE: dict = {"proc": None, "model": None}


def _get_asr():
    """Lazily load the Whisper processor + model once. The audio array is fed
    straight to the model, so no FFmpeg/torchcodec decoding is involved."""
    if _ASR_STATE["model"] is None:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        proc = WhisperProcessor.from_pretrained(ASR_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        _ASR_STATE["proc"], _ASR_STATE["model"] = proc, model
    return _ASR_STATE["proc"], _ASR_STATE["model"]


def _asr_transcribe_words(wav: np.ndarray, sample_rate: int) -> tuple[str, list[tuple[str, float]]]:
    """Whisper transcript + [(word, start_time)] for a clip. ("", []) on failure."""
    import torch
    try:
        proc, model = _get_asr()
        a16 = (librosa.resample(wav.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
               if sample_rate != 16000 else wav.astype(np.float32))
        inp = proc(a16, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        dev = model.device
        with torch.no_grad():
            out = model.generate(
                inp.input_features.to(dev), attention_mask=inp.attention_mask.to(dev),
                language="en", task="transcribe",
                return_token_timestamps=True, return_dict_in_generate=True,
            )
        toks = out["sequences"][0].tolist()
        times = out["token_timestamps"][0].tolist()
    except Exception:
        return "", []
    # Group BPE tokens into words (a new word begins with a leading space).
    words: list[tuple[str, float]] = []
    cur, cur_t = "", 0.0
    for tid, t in zip(toks, times):
        piece = proc.tokenizer.decode([tid])
        if not piece or piece.startswith("<|"):
            continue
        if piece.startswith(" ") and cur:
            words.append((cur, cur_t))
            cur, cur_t = piece.strip(), float(t)
        else:
            if not cur:
                cur_t = float(t)
            cur += piece
    if cur:
        words.append((cur, cur_t))
    return " ".join(w for w, _ in words), words


def _asr_quote_check(wav: np.ndarray, sample_rate: int, sentence: str,
                     quote_words: int) -> tuple[float | None, float | None]:
    """ASR dialogue check. Returns (match_ratio, quote_spw):
    match_ratio = transcript-vs-text word similarity (0..1); quote_spw = the
    quoted span's seconds-per-word from word timings (None if unmeasurable)."""
    import difflib
    transcript, words = _asr_transcribe_words(wav, sample_rate)
    if not words:
        return None, None
    heard = re.findall(r"[a-z0-9']+", transcript.lower())
    want = re.findall(r"[a-z0-9']+", sentence.lower())
    ratio = difflib.SequenceMatcher(None, heard, want).ratio() if want else 1.0
    quote_spw = None
    if quote_words > 0 and len(words) > quote_words:
        # Span from the first word to the first attribution word (quote speech
        # up to the tail onset); slightly conservative (includes any pause).
        dur = words[quote_words][1] - words[0][1]
        if dur > 0.2:
            quote_spw = dur / quote_words
    return ratio, quote_spw


def split_into_paragraphs(text: str) -> list[list[str]]:
    """Return a list of paragraphs, each a list of sentence strings."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        # Dramatic-pause normalization: turn any typed run of "..." (3+ dots) into
        # a single-char ellipsis "…". This lets authors write plain "..." in the
        # story txt for an attention-grabbing pause (e.g. `"Rapunzel, Rapunzel...
        # let down your hair!"`) WITHOUT it splitting the sentence: a literal ". "
        # matches the splitter below and would slice a quote mid-way, but "…" is
        # not in [.!?] so it stays inside the one chunk and the model reads it as a
        # longer hold. (Em-dash "—" is already safe and needs no change.)
        block = re.sub(r"\.{3,}", "…", block)
        # Split after sentence-ending punctuation, allowing an optional closing
        # quote before the whitespace, so a sentence like `... ever." Clover...`
        # is split into two (otherwise it glues to the next sentence and gets no
        # inter-sentence pause). The closing quote is KEPT with its sentence (via
        # look-behind, not consumed) so per-paragraph quote-balance counts stay
        # correct — consuming it made `... hay?" Chicken...` look unbalanced.
        sentences = re.split(r"(?<=[.!?][\"'])\s+|(?<=[.!?])\s+", block)
        sentences = [s.strip() for s in sentences if s.strip()]
        # Re-attach short attribution tails (`"Help!"` + `he called.`) so a
        # quote and its tag generate as one chunk (no tiny mumble-prone clip).
        sentences = _merge_attribution_tails(sentences)
        if sentences:
            paragraphs.append(sentences)
    return paragraphs


def _output_dir_for_label(label: str) -> Path:
    """Age-group subfolder for a story's audio (the cache/review stay shared).

    e.g. age2-3_nar_story_011 -> OUTPUT_BASE/age2-3, age4-5_story_251 ->
    OUTPUT_BASE/age4-5. Keeps different age groups from colliding in one folder.
    """
    for prefix in ("age2-3", "age4-5", "age6-7"):
        if label.startswith(prefix):
            return OUTPUT_BASE / prefix
    if label.startswith("batch"):
        return OUTPUT_BASE / "batches"
    return OUTPUT_BASE / "misc"


# --- Per-voice tuning profiles ---------------------------------------------
# Each reference voice reads at its OWN pace / pitch character, so its knobs are
# NOT shared with other voices (user rule 2026-07). Applied LAST in
# apply_age_profile — after the age profile AND the lively overrides — so a voice's
# own tuning wins. Only list keys that differ; a voice with no entry here just
# uses the age/lively result. Keyed by the voice TAG (voice_tag_for()).
VOICE_PROFILES: dict[str, dict] = {
    "jabari": {
        # Guard-reports (batch01_story_003, batch05_story_042) show jabari reads
        # FAST (~0.40 spw) and the pace floor can't slow her (thrash -> kept:fast).
        # User choice: NO time-stretch. So LOWER the floor to stop the wasteful
        # re-rolls (accept her natural first take, much faster generation), and
        # push the pace via a STRONGER "very slow" hint instead. NOTE: "very slow"
        # was retired for the shared/rooster setup (over-slowed + fought the pitch
        # guard) — but this is jabari-ONLY and the floor is low so nothing fights
        # it; dial back if it over-slows.
        "MIN_WORDS_PER_SECOND": 0.24,
        "PACE_HINT": "very slow, unhurried storytelling pace, clear pauses",
        "NARRATION_TONE": "(very slow, warm expressive storytelling, natural rise and fall)",
    },
}


def _profile_for_label(label: str) -> dict:
    """Merged tuning values for a story: module defaults + its age-group overrides."""
    active = dict(_TUNABLE_DEFAULTS)
    for prefix, override in AGE_PROFILES.items():
        if label.startswith(prefix):
            active.update(override)
            break
    # --as6-7 A/B: force the age6-7 (fastest) profile onto ANY label for this run.
    if FORCE_AGE67:
        active.update(_AGE67_PROFILE)
    return active


def apply_age_profile(label: str, voice_tag: str | None = None) -> dict:
    """Set the per-age + per-voice tuning globals for `label`, returning the active
    values. Age profile + lively run first; the per-voice override (VOICE_PROFILES,
    keyed by `voice_tag`) is applied LAST so a voice's own tuning always wins.

    Called once per story (before its sentences are generated) so each age group
    uses its own pace/guards. All names reset from _TUNABLE_DEFAULTS first, so a
    profile that omits a key gets the default rather than the previous story's.
    """
    global PACE_HINT, PITCH_TOLERANCE_SEMITONES, DURATION_MIN_GUARD_ENABLED
    global MIN_WORDS_PER_SECOND, MIN_DURATION_BASE_S
    global WORDS_PER_SECOND, DURATION_BASE_S, MAX_CANDIDATE_SEEDS
    global QUOTE_PACE_GUARD_ENABLED, QUOTE_MIN_WORDS_PER_SECOND, QUOTE_MIN_WORDS
    global ASR_VERIFY_ENABLED, ASR_MATCH_MIN_RATIO
    global SENTENCE_PAUSE_S, PARAGRAPH_PAUSE_S
    global NARRATION_TONE
    active = _profile_for_label(label)
    # age6-7 pace A/B: --fast67 swaps variant A (the age4-5-clone default) for the
    # faster variant B. Done on the merged dict so BOTH the globals below and the
    # returned dict (console line + --guard-report header) show the fast values.
    if AGE67_PACE_FAST and label.startswith("age6-7"):
        active = dict(active)
        active["PACE_HINT"] = AGE67_PACE_HINT_FAST
        active["SENTENCE_PAUSE_S"] = AGE67_SENTENCE_PAUSE_FAST_S
        active["PARAGRAPH_PAUSE_S"] = AGE67_PARAGRAPH_PAUSE_FAST_S
    PACE_HINT = active["PACE_HINT"]
    _assert_pace_hint(PACE_HINT)  # a per-age profile can NEVER disable/blank it
    PITCH_TOLERANCE_SEMITONES = active["PITCH_TOLERANCE_SEMITONES"]
    DURATION_MIN_GUARD_ENABLED = active["DURATION_MIN_GUARD_ENABLED"]
    MIN_WORDS_PER_SECOND = active["MIN_WORDS_PER_SECOND"]
    MIN_DURATION_BASE_S = active["MIN_DURATION_BASE_S"]
    WORDS_PER_SECOND = active["WORDS_PER_SECOND"]
    DURATION_BASE_S = active["DURATION_BASE_S"]
    MAX_CANDIDATE_SEEDS = active["MAX_CANDIDATE_SEEDS"]
    QUOTE_PACE_GUARD_ENABLED = active["QUOTE_PACE_GUARD_ENABLED"]
    QUOTE_MIN_WORDS_PER_SECOND = active["QUOTE_MIN_WORDS_PER_SECOND"]
    QUOTE_MIN_WORDS = active["QUOTE_MIN_WORDS"]
    ASR_VERIFY_ENABLED = active["ASR_VERIFY_ENABLED"]
    ASR_MATCH_MIN_RATIO = active["ASR_MATCH_MIN_RATIO"]
    SENTENCE_PAUSE_S = active["SENTENCE_PAUSE_S"]
    PARAGRAPH_PAUSE_S = active["PARAGRAPH_PAUSE_S"]
    NARRATION_TONE = active["NARRATION_TONE"]
    if LIVELY_MODE:
        # Thin the re-roll-heavy guards so the natural, characterful first take
        # survives instead of being replaced by a flat "safe average" one.
        PITCH_TOLERANCE_SEMITONES = max(PITCH_TOLERANCE_SEMITONES, 7.0)  # gross drift only
        # Age-aware pace floor. The floor only REJECTS fast takes; the seed loop
        # then keeps the FIRST later take that passes, which can OVERSHOOT to
        # too-slow (e.g. a 0.34 take rejected, next seed lands 0.62 -> 4.3s drag).
        # Both ages now sit at ~0.30-0.28: age4-5's PACE_HINT is the FASTER
        # "natural, flowing, barely slow", so its reads land ~0.32-0.40 spw on
        # purpose; the old 0.33 floor then fought that faster pace (whole story
        # thrashed + kept:fast compromises on 9-11 word lines). 0.30 lets the
        # intended-faster takes pass while still catching the truly rushed (<0.30).
        # age2-3 is a touch slower still, so 0.28 keeps its natural take first-try
        # (no re-roll, no overshoot).
        MIN_WORDS_PER_SECOND = 0.30 if label.startswith(("age4-5", "age6-7", "batch")) else 0.28
        MIN_DURATION_BASE_S = 0.40    # with the coeff above -> effective floor catches only
                                      # the clearly-rushed tail; 0.40+ spw bulk passes first try.
        QUOTE_PACE_GUARD_ENABLED = False
        MAX_CANDIDATE_SEEDS = 6   # was 4; the pace floor now does real work, so give
                                  # rushed lines a few more tries to find a slower take.
                                  # Pitch is loose (<=7st) so extra seeds don't flatten register.
        # Pair the minimal guards with an EXPRESSIVE narration tone across ALL
        # ages (not just age4-5): lively = "minimal guards + strong rise-and-fall".
        # This overrides age2-3's gentler tone only during a --lively run; normal
        # runs keep each age's own tone.
        NARRATION_TONE = "(warm, expressive storytelling, natural rise and fall)"
        # Keep the RETURNED profile dict in sync with these lively overrides so
        # the console line + --guard-report header show the ACTUAL values used,
        # not the pre-lively profile (otherwise the header misleadingly prints
        # e.g. pitch<=4.0/wps-min=0.30 while generation really used 7.0/0.20).
        active.update({
            "PITCH_TOLERANCE_SEMITONES": PITCH_TOLERANCE_SEMITONES,
            "MIN_WORDS_PER_SECOND": MIN_WORDS_PER_SECOND,
            "MIN_DURATION_BASE_S": MIN_DURATION_BASE_S,
            "QUOTE_PACE_GUARD_ENABLED": QUOTE_PACE_GUARD_ENABLED,
            "MAX_CANDIDATE_SEEDS": MAX_CANDIDATE_SEEDS,
            "NARRATION_TONE": NARRATION_TONE,
        })
    # Per-voice overrides (applied LAST so a voice's own tuning wins over the age
    # profile AND the lively overrides). Voices never share knobs (VOICE_PROFILES).
    vprof = VOICE_PROFILES.get(voice_tag or "", {})
    if vprof:
        if "PACE_HINT" in vprof:
            PACE_HINT = vprof["PACE_HINT"]
            _assert_pace_hint(PACE_HINT)
        if "NARRATION_TONE" in vprof:
            NARRATION_TONE = vprof["NARRATION_TONE"]
        if "PITCH_TOLERANCE_SEMITONES" in vprof:
            PITCH_TOLERANCE_SEMITONES = vprof["PITCH_TOLERANCE_SEMITONES"]
        if "MIN_WORDS_PER_SECOND" in vprof:
            MIN_WORDS_PER_SECOND = vprof["MIN_WORDS_PER_SECOND"]
        if "MIN_DURATION_BASE_S" in vprof:
            MIN_DURATION_BASE_S = vprof["MIN_DURATION_BASE_S"]
        if "MAX_CANDIDATE_SEEDS" in vprof:
            MAX_CANDIDATE_SEEDS = vprof["MAX_CANDIDATE_SEEDS"]
        if "SENTENCE_PAUSE_S" in vprof:
            SENTENCE_PAUSE_S = vprof["SENTENCE_PAUSE_S"]
        if "PARAGRAPH_PAUSE_S" in vprof:
            PARAGRAPH_PAUSE_S = vprof["PARAGRAPH_PAUSE_S"]
        active.update(vprof)
    return active


def main() -> int:
    global _CACHE_BYPASS, _CACHE_REFRESH
    global LOUDNESS_NORMALIZE_ENABLED, TRIM_EDGES_ENABLED
    global AUDIO_QUALITY_GUARD_ENABLED, ONSET_CLICK_ENABLED, CENTROID_GUARD_ENABLED
    global TONE_ENABLED
    global SPEED
    global CFG_VALUE, INFERENCE_TIMESTEPS, SEED
    global GUARD_LOG_PATH
    global LIVELY_MODE
    global AGE67_PACE_FAST, FORCE_AGE67
    global READ_PARAGRAPH_MODE
    dur_report = False
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Separate flags (--...) from positional filter/override tokens.
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    positional = [a for a in args if not a.startswith("--")]

    # --raw: A/B diagnostic. Turn OFF everything added on top of "generate each
    # sentence + stitch with silence": loudness normalize, edge trim, and the
    # extended defect guards (onset-click / spectral-centroid / internal-silence).
    # Basic duration/pitch guards stay on. Use to check if the new post-processing
    # is what degraded the audio: `... age2-3_rep_story_001 --raw --no-cache`.
    if "--raw" in flags:
        LOUDNESS_NORMALIZE_ENABLED = False
        TRIM_EDGES_ENABLED = False
        AUDIO_QUALITY_GUARD_ENABLED = False
        ONSET_CLICK_ENABLED = False
        CENTROID_GUARD_ENABLED = False
        print("RAW mode: loudness-norm, edge-trim, and extended guards OFF",
              file=sys.stderr)

    # Lively is the DEFAULT (LIVELY_MODE=True at module level): it is the
    # validated production config, so no flag is needed. It thins the re-roll-heavy
    # guards to a minimal safety net so the natural, characterful FIRST take
    # survives instead of being re-rolled into a flat "safe average". The per-age
    # pitch/min-wps/seed overrides happen in apply_age_profile(); the two
    # non-profile globals (loudness-norm, centroid guard) are turned off here.
    #
    # --safe: the inverse escape hatch. Restores the FULL guard stack
    # (loudness-norm + centroid + quote-pace + tight per-age pitch). Nothing is
    # deleted — this is the A/B baseline for re-validating a future change. Pair
    # either mode with --no-cache (they share cache keys otherwise). --lively is
    # still accepted as a harmless no-op (lively is already the default).
    if "--safe" in flags:
        LIVELY_MODE = False
        print("SAFE mode: full guards ON (loudness-norm + centroid + quote-pace + tight pitch) "
              "- stable, less expressive (A/B baseline)",
              file=sys.stderr)
    else:
        LIVELY_MODE = True
        LOUDNESS_NORMALIZE_ENABLED = False   # restore per-sentence volume dynamics/emphasis
        CENTROID_GUARD_ENABLED = False        # stop rejecting characterful timbres
        print("LIVELY mode (default): loose pitch (<=7st), no loudness-norm/centroid/quote-pace guard, "
              "age-aware pace floor (age4-5 0.30 / others 0.28), seeds 6 - expressive first take kept "
              "(use --safe for the full-guard baseline)",
              file=sys.stderr)

    # --first=N: only generate the first N sentences of each story. Fast A/B
    # testing so you don't wait for the whole story: `... --first=5 --no-cache`.
    first_n: int | None = None
    for f in flags:
        if f.startswith("--first="):
            try:
                first_n = max(1, int(f.split("=", 1)[1]))
            except ValueError:
                print(f"invalid --first value: {f}", file=sys.stderr)
                return 1
    if first_n is not None:
        print(f"first-{first_n} mode: only the first {first_n} sentence(s)",
              file=sys.stderr)

    # --fast67: age6-7 pace A/B. Swap the age6-7 default (variant A = age4-5
    # clone) for the faster variant B (conversational pace + shorter pauses) and
    # tag the output `_fast` so it sits beside the variant-A take. Pair with
    # --no-cache (PACE_HINT is in the cache key).
    if "--fast67" in flags:
        AGE67_PACE_FAST = True
        print("FAST67 mode: age6-7 uses the faster pace (variant B), output tagged _fast",
              file=sys.stderr)

    # --as6-7: force the age6-7 (fastest) profile onto ANY story this run, tagged
    # _as67, so you can A/B a story's default pace vs the fast pace on the same text.
    if "--as6-7" in flags:
        FORCE_AGE67 = True
        print("AS6-7 mode: forcing the age6-7 (fast) profile, output tagged _as67",
              file=sys.stderr)

    # --paragraph: read WHOLE paragraphs as one utterance (natural flow across
    # sentences) instead of one-sentence-at-a-time, for age4-5 + age6-7 + tts
    # batch stories. Paragraphs are still stitched with the paragraph pause;
    # output is tagged _para so it sits beside the per-sentence take. Paragraph
    # reads use a tighter cfg (PARAGRAPH_CFG_VALUE=2.6) for the most faithful
    # timbre over the longer utterance (an explicit --cfg= still overrides).
    # Pair with --no-cache.
    if "--paragraph" in flags:
        READ_PARAGRAPH_MODE = True
        print("PARAGRAPH mode: age4-5 + age6-7 + tts batch read whole-paragraph "
              "in ONE take (fixed seed, NO guards), cfg=2.6, output tagged _para",
              file=sys.stderr)

    # --notone: drop the VoxCPM style tag prefix entirely (plain sentence prompt).
    # A/B against the tone-prefixed default to see if the tag hurts articulation.
    if "--notone" in flags:
        TONE_ENABLED = False
        print("NOTONE mode: no style-tag prefix on sentences", file=sys.stderr)

    # --asr: enable Whisper transcript+pace verification on dialogue lines for
    # this run (overrides the default; per-age profiles can still turn it off).
    if "--asr" in flags:
        _TUNABLE_DEFAULTS["ASR_VERIFY_ENABLED"] = True
        print("ASR mode: Whisper transcript+pace verification ON for dialogue",
              file=sys.stderr)

    # --dur-report: print a per-sentence duration table (word count / dur /
    # sec-per-word / current limit) to calibrate the DURATION guard. Logging only.
    dur_report_path = OUTPUT_BASE / "dur_report.txt"
    if "--dur-report" in flags:
        dur_report = True
        dur_report_path.write_text(
            "# per-sentence duration report\n"
            f"# WORDS_PER_SECOND={WORDS_PER_SECOND} DURATION_BASE_S={DURATION_BASE_S} "
            f"DURATION_MIN_LIMIT_S={DURATION_MIN_LIMIT_S}\n",
            encoding="utf-8",
        )
        print(f"dur-report: writing table to {dur_report_path}", file=sys.stderr)

    # --guard-report: write ONE report PER STORY (so a problem story can be
    # analysed in isolation) into OUTPUT_BASE/guard_reports/guard_report_<label>.txt.
    # The file is (re)created per story inside the generation loop below; here we
    # just record the mode and make sure the folder exists.
    guard_report_enabled = "--guard-report" in flags
    guard_reports_dir = OUTPUT_BASE / "guard_reports"
    if guard_report_enabled:
        guard_reports_dir.mkdir(parents=True, exist_ok=True)
        print(f"guard-report: per-story tables -> {guard_reports_dir}", file=sys.stderr)

    # --speed=X: pitch-preserving time-stretch per sentence (X<1 slower).
    for f in flags:
        if f.startswith("--speed="):
            try:
                SPEED = float(f.split("=", 1)[1])
            except ValueError:
                print(f"invalid --speed value: {f}", file=sys.stderr)
                return 1
            if not (0.5 <= SPEED <= 1.5):
                print(f"--speed out of range (0.5-1.5): {SPEED}", file=sys.stderr)
                return 1
    if SPEED != 1.0:
        print(f"speed mode: {SPEED}x (pitch-preserving time-stretch)", file=sys.stderr)

    # --timesteps=N / --cfg=X: generation-quality overrides. Higher timesteps
    # give cleaner onsets and less sampling noise (10 is fast but rough; try
    # 16-32). cfg tunes how tightly the model follows the reference/prompt.
    for f in flags:
        if f.startswith("--timesteps="):
            try:
                INFERENCE_TIMESTEPS = int(f.split("=", 1)[1])
            except ValueError:
                print(f"invalid --timesteps value: {f}", file=sys.stderr)
                return 1
        elif f.startswith("--cfg="):
            try:
                CFG_VALUE = float(f.split("=", 1)[1])
            except ValueError:
                print(f"invalid --cfg value: {f}", file=sys.stderr)
                return 1
        elif f.startswith("--seed="):
            try:
                SEED = int(f.split("=", 1)[1])
            except ValueError:
                print(f"invalid --seed value: {f}", file=sys.stderr)
                return 1
    # Paragraph mode uses a tighter cfg (PARAGRAPH_CFG_VALUE) for a more faithful
    # cloned timbre over the longer whole-paragraph utterance, unless the user
    # explicitly passed --cfg=.
    if READ_PARAGRAPH_MODE and not any(f.startswith("--cfg=") for f in flags):
        CFG_VALUE = PARAGRAPH_CFG_VALUE
    print(f"gen params: timesteps={INFERENCE_TIMESTEPS} cfg={CFG_VALUE} seed={SEED}", file=sys.stderr)

    if "--no-cache" in flags:
        _CACHE_BYPASS = True
        print("cache: bypassed for this run (--no-cache)", file=sys.stderr)
    if "--refresh-cache" in flags:
        _CACHE_REFRESH = True
        print("cache: refresh - regenerate these jobs and OVERWRITE their cache "
              "(others untouched) (--refresh-cache)", file=sys.stderr)
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
        # Search ALL discovered production stories (not just the JOBS sample set)
        # so any story — incl. age6-7 / batch — can be generated ad hoc by label.
        matched = [job for job in _discovered if story_key in job[0].lower()]
        if not matched:
            print(f"No story matches '{story_key}' among {len(_discovered)} discovered stories.",
                  file=sys.stderr)
            return 1
        # Tag the output with the voice so it doesn't overwrite the default take.
        for label, txt_path, _old_voice in matched:
            jobs.append((label, txt_path, ref))
            seen.add(label)

    # Plain filter jobs (story or voice substring). Searches ALL discovered
    # production stories so age6-7 / batch labels are targetable from the CLI.
    for w in plain_filters:
        for job in _discovered:
            if job[0] in seen:
                continue
            if w in job[0].lower() or w in str(job[2]).lower():
                jobs.append(job)
                seen.add(job[0])

    if not overrides and not plain_filters:
        if "--all" in flags:
            # FULL production run: EVERY discovered story (all ages + all tts),
            # not the curated JOBS sample. No positional -> production names (no _cmd).
            jobs = list(_discovered)
            print(f"--all: FULL run over all {len(jobs)} discovered stories "
                  f"(production names)", file=sys.stderr)
        else:
            jobs = JOBS
    elif not jobs:
        print(f"No jobs match {positional} among {len(_discovered)} discovered stories.",
              file=sys.stderr)
        return 1

    # CLI-invoked runs (any positional argument) tag their output with _cmd to
    # distinguish them from the default full-batch generation. --prod overrides that
    # so a targeted re-read of edited stories overwrites the PRODUCTION names.
    cli_run = bool(positional)
    prod_names = "--prod" in flags

    print("Loading VoxCPM model (once)...", file=sys.stderr)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=DENOISE_REFERENCE)
    sample_rate = model.tts_model.sample_rate

    failures = []
    review_entries: list[dict] = []
    for i, (label, txt_path, voice_name) in enumerate(jobs, start=1):
        ref_path = REFERENCE_DIR / voice_name
        vtag = voice_tag_for(voice_name)
        # Every output name carries the story label AND the voice; CLI runs add _cmd.
        stem = f"{label}_{vtag}"
        if cli_run and not prod_names:
            stem += "_cmd"
        # age6-7 fast-pace take (variant B) gets a _fast tag so it lands next to
        # the variant-A take instead of overwriting it.
        if AGE67_PACE_FAST and label.startswith("age6-7"):
            stem += "_fast"
        if FORCE_AGE67:
            stem += "_as67"
        # Whole-paragraph read (age4-5 + age6-7 + tts batch) gets a _para tag so
        # it lands beside the per-sentence production take.
        if READ_PARAGRAPH_MODE and label.startswith(("age4-5", "age6-7", "batch")):
            stem += "_para"
        if first_n is not None:
            stem += f"_first{first_n}"
        out_dir = _output_dir_for_label(label)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{stem}.wav"
        print(f"[{i}/{len(jobs)}] {label}  <- {txt_path.name}  voice={voice_name[:32]}")
        # Per-age + per-voice tuning: set this story's pace/guards before generating.
        prof = apply_age_profile(label, vtag)
        print(f"  profile: pace={prof['PACE_HINT']!r} pitch<={prof['PITCH_TOLERANCE_SEMITONES']}st "
              f"min-guard={prof['DURATION_MIN_GUARD_ENABLED']} "
              f"wps={prof['WORDS_PER_SECOND']} seeds={prof['MAX_CANDIDATE_SEEDS']} "
              f"pause={prof['SENTENCE_PAUSE_S']}/{prof['PARAGRAPH_PAUSE_S']}s",
              file=sys.stderr)
        # Per-age inter-clip silence, built AFTER apply_age_profile so each story
        # uses its own pauses (age2-3: 0.7s between sentences, 1.2s between
        # paragraphs; other ages keep the 0.45/0.80 defaults).
        sentence_gap = np.zeros(int(SENTENCE_PAUSE_S * sample_rate), dtype=np.float32)
        paragraph_gap = np.zeros(int(PARAGRAPH_PAUSE_S * sample_rate), dtype=np.float32)
        # Per-story guard report (one file each) so a problem story is isolated,
        # nested by age group to mirror the audio layout
        # (guard_reports/age2-3/..., guard_reports/age4-5/..., etc.).
        if guard_report_enabled:
            gr_dir = guard_reports_dir / out_dir.name
            gr_dir.mkdir(parents=True, exist_ok=True)
            GUARD_LOG_PATH = gr_dir / f"guard_report_{safe_name(label)}.txt"
            GUARD_LOG_PATH.write_text(
                "# per-seed guard diagnostics (every candidate, incl. the accepted/kept one)\n"
                "# fields: seed / result / dur(s) / spw(sec-per-word) / pitch(st) / words / limit(s) / text\n"
                f"# story: {label}  profile: pitch<={prof['PITCH_TOLERANCE_SEMITONES']}st "
                f"pace={prof['PACE_HINT']!r} "
                f"pause={prof['SENTENCE_PAUSE_S']}/{prof['PARAGRAPH_PAUSE_S']}s "
                f"wps-min={prof['MIN_WORDS_PER_SECOND']}\n",
                encoding="utf-8",
            )

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

        # --first=N: keep only the first N sentences (across paragraphs).
        if first_n is not None:
            trimmed: list[list[str]] = []
            remaining = first_n
            for para in paragraphs:
                if remaining <= 0:
                    break
                take = para[:remaining]
                if take:
                    trimmed.append(take)
                    remaining -= len(take)
            paragraphs = trimmed

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
            # Pass 1 — generate every unit (cache-aware) + trim dead air. Units are
            # single sentences by default; in --paragraph mode (age4-5 + age6-7 +
            # tts batch) each WHOLE paragraph is one unit so the model reads it in
            # one natural take. Paragraph units use the narration tone (narration_only).
            use_paragraph = READ_PARAGRAPH_MODE and label.startswith(("age4-5", "age6-7", "batch"))
            if use_paragraph:
                units = [(p_index, 0, " ".join(sentences))
                         for p_index, sentences in enumerate(paragraphs)]
            else:
                units = [(p_index, s_index, sentence)
                         for p_index, sentences in enumerate(paragraphs)
                         for s_index, sentence in enumerate(sentences)]

            items: list[dict] = []
            for p_index, s_index, sentence in units:
                wav, s_issues = generate_sentence(
                    model, sentence, ref_path, sample_rate, vtag,
                    narration_only=use_paragraph,
                    no_guard=use_paragraph,
                )
                wav = np.asarray(wav, dtype=np.float32)
                if SPEED != 1.0 and wav.size:
                    wav = np.asarray(
                        librosa.effects.time_stretch(wav, rate=SPEED),
                        dtype=np.float32,
                    )
                if TRIM_EDGES_ENABLED:
                    wav = _trim_edges(wav, sample_rate)
                if dur_report:
                    wc = _word_count(sentence)
                    dur = wav.size / sample_rate
                    lim = expected_max_seconds(sentence)
                    spw = dur / wc if wc else 0.0
                    flag = " <-- OVER" if dur > lim else ""
                    row = (
                        f"[dur] {label} p{p_index}.s{s_index} "
                        f"words={wc:2d} dur={dur:5.2f}s spw={spw:4.2f} "
                        f"limit={lim:4.2f}s{flag}  {sentence[:42]!r}"
                    )
                    print("  " + row, file=sys.stderr)
                    with dur_report_path.open("a", encoding="utf-8") as _f:
                        _f.write(row + "\n")
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
