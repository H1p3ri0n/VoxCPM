#!/usr/bin/env python3
"""Build a per-story INTRO clip: music jingle + spoken story TITLE.

Each intro = a Suno jingle (ref/intro1.wav or ref/intro2.wav, chosen per story by
story-number parity so it's reproducible, order-independent and ~50/50: odd ->
intro1, even -> intro2) with the story TITLE read by the ROOSTER voice. The title ENTERS during the jingle's tail fade-out (a slight
overlap) so the two blend instead of hard-cutting. Output is a STANDALONE clip
(out/first100/intros/<label>_intro.wav) — the story's finished audio is never
touched.

Titles are looked up from two READ-ONLY mapping files (built elsewhere):
  <FIRST100>/titles.json        -> age2-3 / age4-5 / age6-7 stories
  <TTS_STORIES>/titles.json     -> batch stories
Both are {"<label>": "Human Title"} and are merged. A label with no (or blank)
title is SKIPPED with a warning.

Reuses generate_with_pauses (imported as `g`) for the model, the reference dir,
the peak-normalize target and story discovery — same bootstrap as ab_pause.py.

Run from the repo root.

  # Cinderella audition (no mapping file needed) — try each jingle:
  uv run python scripts/make_intros.py batch01_story_001 --title "Cinderella" --jingle 1
  uv run python scripts/make_intros.py batch01_story_001 --title "Cinderella" --jingle 2

  # Coverage report (which discovered stories have / lack a title):
  uv run python scripts/make_intros.py --list

  # Batch: every story that has a title in the mapping files:
  uv run python scripts/make_intros.py

  # A subset by label substring:
  uv run python scripts/make_intros.py batch01 age4-5_story_251
"""

import argparse
import json
import os
import re
import sys

import librosa
import numpy as np
import soundfile as sf

# Make the repo-root module importable when run as scripts/make_intros.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_with_pauses as g  # noqa: E402

# --- configuration (tune these) --------------------------------------------
# The two jingles. Index 1/2 map to the --jingle CLI values.
JINGLE_PATHS = {
    1: g.REFERENCE_DIR / "intro1-final.wav",
    2: g.REFERENCE_DIR / "intro2-final.wav",
}

# Voice that reads the title (user choice: rooster).
TITLE_VOICE = g.VOICE_ROOSTER

# Style tag prepended to the title. narration-style, read at a natural pace.
# The pitch cues push the model to END LOW (declarative) instead of rising like a
# question. The trailing period (_punctuate_title) plus the SEED SWEEP in
# generate_title are what actually guarantee a falling ending; the tone reinforces.
TITLE_TONE = "(clear, warm, welcoming voice, calm confident statement, gentle falling intonation, settling down to a low final note)"

# Title read speed. 1.0 = normal (no time-stretch) — user wanted a natural read
# (slower sounded weird). <1 slows via a gentle pitch-preserving stretch if ever
# needed. Override per-run with --speed.
TITLE_SPEED = 1.0

TITLE_CFG = g.CFG_VALUE          # 2.5 — clean articulation + stable rooster timbre
TITLE_TIMESTEPS = 24             # a touch cleaner than the 10 used in batch reads

# A short title can END ON A RISING pitch or be read too quickly. Generate several
# seeds and keep a candidate that clears the pace floor, preferring the one whose
# terminal pitch slope descends the most (see generate_title / _end_slope).
TITLE_SEEDS = [g.SEED + offset for offset in range(16)]  # 45..60

# Listener-selected title seeds, keyed by exact story label. A label listed here
# renders ONLY that seed (pace/slope are still measured and printed, but the
# sweep never replaces it), so an approved title read survives re-runs. `--seed`
# on the command line still wins, for auditioning a different take.
TITLE_SEED_OVERRIDES: dict[str, int] = {
    "batch07_story_062": 54,
    "batch02_story_018": 47,
}
# Pace is measured after trimming generated edge silence. Candidates below this
# seconds-per-word floor are rejected and the next seed is tried. Keep this
# conservative: it catches distinctly rushed titles without dragging natural ones.
TITLE_MIN_SECONDS_PER_WORD = 0.30
TITLE_PACE_TRIM_TOP_DB = 30.0
# Stop the sweep early once a candidate's end-slope (Hz per f0 frame) is at least
# this negative (clearly falling). More negative = stricter. -0.5 avoids
# exhausting all seeds for clearly falling endings while rejecting weak dips.
FALLING_SLOPE_EARLY_STOP = -0.5
# Acceptance gate: after the sweep, if the best candidate's end-slope is still
# ABOVE this (i.e. rising), the title is SKIPPED and logged for later batch
# reprocessing. Questions ending in "?" are exempt (they are meant to rise).
# 0.0 = must not rise; raise it to tolerate near-flat endings.
FALLING_SLOPE_ACCEPT = 0.0

# Intro shaping.
# How early the title enters before the jingle ends (also the jingle's fade length).
# Bigger = the name starts EARLIER over the music. Per-jingle: intro1 is longer /
# has a longer lead-in, so its title starts earlier than intro2's.
OVERLAP_S = {1: 2.0, 2: 1.5}
DEFAULT_OVERLAP_S = 1.5  # fallback if a jingle idx isn't in OVERLAP_S
LEAD_SILENCE_S = 0.15  # tiny silence before the jingle starts
# Trailing pad after the title. The main post-intro gap (~1.0s) is added by the
# APP at compile time; this 0.5s is a cushion so the ending never feels clipped.
TRAIL_SILENCE_S = 0.5
# Sidechain-style ducking: while the title plays, the jingle dips to DUCK_LEVEL so
# the name stays clear; BEFORE and AFTER the title it plays at FULL volume, so the
# jingle's tail (e.g. intro2's shooting-star flourish) rings out instead of being
# faded away. A short END_FADE_S at the very end avoids a hard cut.
DUCK_LEVEL = 0.35        # jingle volume (0..1) underneath the title
DUCK_RAMP_S = 0.30       # smooth ramp in / out of the duck
# Gentle raised-cosine fade over the very end of the jingle so it doesn't sound
# cut. Per-jingle. intro2 uses intro2-5.wav whose ENDING (the shooting star) is
# intentionally boosted louder; its fade is a middle ground (0.7s) so the star
# still rings but the ending tapers off gently instead of vanishing abruptly.
END_FADE_S = {1: 0.40, 2: 1.20}
DEFAULT_END_FADE_S = 0.40
# Loudness-match the intro TITLE voice to the STORY narration so there's no volume
# jump into the story body. Measured narration speech RMS ~0.22 (-13 dBFS): the
# title voice is scaled to this RMS, and the intro is then only PEAK-CAPPED (not
# renormalized to a fixed peak) so that matched loudness survives the mix.
NARRATION_VOICE_RMS = 0.22
PEAK_CEILING = 0.97

OUTPUT_DIR = g.OUTPUT_BASE / "intros"
# Titles that never landed on a falling ending (seeds exhausted) are logged here,
# one JSON object per line, for later batch reprocessing.
RISING_LOG = OUTPUT_DIR / "rising_titles.jsonl"
# Titles that exhaust all seeds while still too fast are kept separate so they can
# be targeted for another pass without confusing them with pitch failures.
FAST_LOG = OUTPUT_DIR / "fast_titles.jsonl"

# READ-ONLY title mapping files (created in another project; never written here).
# Both are keyed by the SAME bare id (story_001 exists in each), and values are
# {"en": ..., "zh": ...}. They are kept SEPARATE and routed by label prefix
# (batch* -> tts, everything else -> First100); we read the English title.
FIRST100_TITLES = g.FIRST100 / "titles.json"
TTS_TITLES = g.TTS_STORIES / "titles.json"


# --- helpers ---------------------------------------------------------------
def _read_title_map(path) -> dict[str, str]:
    """Read a {story_id: {"en","zh"} | "title"} file -> {story_id: english title}."""
    if not path.exists():
        print(f"  note: title map not found (skipped): {path}", file=sys.stderr)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARN: could not read {path}: {exc}", file=sys.stderr)
        return {}
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for sid, val in data.items():
            if isinstance(val, dict):
                title = str(val.get("en", "")).strip()
            elif isinstance(val, str):
                title = val.strip()
            else:
                title = ""
            if title:
                out[str(sid)] = title
    return out


def load_titles() -> tuple[dict[str, str], dict[str, str]]:
    """Return (first100_map, tts_map), each keyed by bare 'story_<num>' -> en title.

    The two source files use the SAME bare keys (story_001 exists in both), so they
    are kept SEPARATE and routed by label prefix (see title_for), never merged.
    """
    return _read_title_map(FIRST100_TITLES), _read_title_map(TTS_TITLES)


_STORY_ID_RE = re.compile(r"(story_\d+)")


def _story_id(label: str) -> str | None:
    """Extract the bare 'story_<num>' from a discovered label.

    e.g. age4-5_story_251 -> story_251, batch01_story_001 -> story_001,
    age2-3_nar_story_011 -> story_011.
    """
    m = _STORY_ID_RE.search(label)
    return m.group(1) if m else None


def title_for(label: str, first100: dict[str, str], tts: dict[str, str]) -> str:
    """Look up a label's English title, routed by source (batch -> tts, else First100)."""
    sid = _story_id(label)
    if not sid:
        return ""
    src = tts if label.startswith("batch") else first100
    return src.get(sid, "").strip()


def pick_jingle(label: str) -> int:
    """Deterministic 1/2 pick by STORY-NUMBER PARITY: odd -> intro1, even -> intro2.

    Reproducible and order-independent (the number is fixed per story), and evenly
    split (~50/50) even though the numbering has gaps. Falls back to intro1 if no
    story number is found.
    """
    sid = _story_id(label)
    if not sid:
        return 1
    return 1 if int(sid.split("_")[1]) % 2 == 1 else 2


def load_jingle(idx: int, sample_rate: int) -> np.ndarray:
    """Load a jingle as float32 mono at the model sample rate.

    The jingle files (intro1-final / intro2-final) are already loudness-matched
    (baked to a common RMS with a peak cap), so no runtime normalization is done.
    """
    path = JINGLE_PATHS[idx]
    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(wav, dtype=np.float32)


def _punctuate_title(title: str) -> str:
    """Force a declarative (falling) ending.

    A bare title with no sentence-ending punctuation (e.g. "Cinderella") reads with
    an uncertain / rising pitch. Appending a period makes it a statement so the
    model ENDS LOW (falling intonation). Titles that are genuinely questions
    ("Who Is the Strawberry Thief?") keep their "?" and stay rising, which is
    correct for them; "!" titles keep their emphasis.
    """
    t = title.strip()
    if t and t[-1] not in ".!?…":
        t += "."
    return t


def _end_slope(wav: np.ndarray, sample_rate: int) -> float:
    """Linear slope (Hz per frame) of the pitch over the title's ENDING.

    Negative = the pitch falls at the end (declarative); positive = it rises
    (question-like). Used to pick the most falling seed. Returns 0.0 if pitch
    can't be tracked (treated as neutral).
    """
    if wav.size < int(0.2 * sample_rate):
        return 0.0
    try:
        f0, _voiced, _prob = librosa.pyin(
            wav, fmin=80.0, fmax=400.0, sr=sample_rate
        )
    except Exception:
        return 0.0
    f0 = f0[np.isfinite(f0)]
    if f0.size < 4:
        return 0.0
    # The ending = the last ~half of the voiced frames.
    tail = f0[-max(4, f0.size // 2):]
    x = np.arange(tail.size, dtype=np.float64)
    slope = float(np.polyfit(x, tail, 1)[0])
    return slope


def _title_pace(wav: np.ndarray, sample_rate: int, title: str,
                speed: float = 1.0) -> tuple[float, int, float]:
    """Return (effective duration, word count, seconds per word) for title speech."""
    try:
        speech, _idx = librosa.effects.trim(
            wav.astype(np.float32), top_db=TITLE_PACE_TRIM_TOP_DB
        )
    except Exception:
        speech = wav
    if speech.size == 0:
        speech = wav
    words = max(1, len(re.findall(r"[A-Za-z0-9']+", title)))
    # librosa time_stretch(rate<1) lengthens by 1/rate; account for a requested
    # --speed during candidate selection without stretching every candidate.
    effective_duration = len(speech) / sample_rate / speed
    return effective_duration, words, effective_duration / words


def generate_title(model, sample_rate: int, title: str, speed: float,
                   force_seed: int | None = None):
    """Render the title, rejecting rushed seeds and preferring a falling ending.

    A bare/short title can end rising or be rushed. Sweep TITLE_SEEDS, discard
    candidates below TITLE_MIN_SECONDS_PER_WORD, then keep the acceptable candidate
    whose terminal pitch slope descends the most. If every candidate is too fast,
    return the slowest one for diagnostics; make_one() will skip writing it.

    `force_seed` renders that ONE listener-chosen seed instead of sweeping; the
    pace/slope are still measured and reported.

    Returns (wav, best_slope, best_seed, tried, duration, words, seconds_per_word).
    """
    ref = g.REFERENCE_DIR / TITLE_VOICE
    if not ref.exists():
        raise FileNotFoundError(f"title reference voice missing: {ref}")
    prompt = f"{TITLE_TONE} {_punctuate_title(title)}"
    seeds = [force_seed] if force_seed is not None else TITLE_SEEDS

    best_wav: np.ndarray | None = None
    best_slope = float("inf")
    best_seed = seeds[0]
    best_duration = 0.0
    best_words = 1
    best_spw = 0.0
    slowest: tuple[float, np.ndarray, float, int, float, int] | None = None
    tried = 0
    for seed in seeds:
        wav = np.asarray(
            model.generate(
                text=prompt,
                reference_wav_path=str(ref),
                cfg_value=TITLE_CFG,
                inference_timesteps=TITLE_TIMESTEPS,
                denoise=False,
                seed=seed,
            ),
            dtype=np.float32,
        )
        tried += 1
        slope = _end_slope(wav, sample_rate)
        duration, words, spw = _title_pace(wav, sample_rate, title, speed)
        fast = spw < TITLE_MIN_SECONDS_PER_WORD
        print(
            f"    title seed={seed} dur={duration:.2f}s words={words} "
            f"spw={spw:.2f} {'FAST' if fast else 'pace-OK'} "
            f"end={slope:+.2f}",
            file=sys.stderr,
        )
        if slowest is None or spw > slowest[0]:
            slowest = (spw, wav, slope, seed, duration, words)
        if not fast and slope < best_slope:
            best_slope, best_wav, best_seed = slope, wav, seed
            best_duration, best_words, best_spw = duration, words, spw
        if not fast and slope <= FALLING_SLOPE_EARLY_STOP:
            break

    if best_wav is None and slowest is not None:
        best_spw, best_wav, best_slope, best_seed, best_duration, best_words = slowest
    wav = best_wav if best_wav is not None else np.zeros(0, dtype=np.float32)
    print(
        f"    title end-pitch slope={best_slope:+.2f} Hz/frame "
        f"({'falling' if best_slope < 0 else 'rising'})  "
        f"pace={best_spw:.2f}s/word seed={best_seed} "
        f"(tried {tried}/{len(seeds)})",
        file=sys.stderr,
    )
    if speed != 1.0 and wav.size:
        wav = np.asarray(librosa.effects.time_stretch(wav, rate=speed), dtype=np.float32)
    # Loudness-match the title voice to the story narration (measured speech RMS
    # ~0.22) so the intro title and the story body play at the same volume. A
    # scalar gain doesn't affect the pitch slope measured above.
    rms = float(np.sqrt(np.mean(wav ** 2))) if wav.size else 0.0
    if rms > 1e-6:
        wav = (wav * (NARRATION_VOICE_RMS / rms)).astype(np.float32)
    return wav, best_slope, best_seed, tried, best_duration, best_words, best_spw


def make_one(model, sample_rate, label, title, jingle_idx, speed,
             force_write: bool = False, force_seed: int | None = None) -> dict | None:
    """Generate and write one intro clip.

    Returns None on success. If the title never lands on a falling ending after all
    seeds (and isn't a question), it is SKIPPED and a failure record dict is
    returned instead — unless force_write=True (e.g. a --title audition), which
    writes anyway with a warning.

    `force_seed` is a listener-chosen seed: it is written even if the guards
    dislike it, since the choice was made by ear.
    """
    title_wav, slope, _seed, tried, duration, words, spw = generate_title(
        model, sample_rate, title, speed, force_seed
    )
    is_question = _punctuate_title(title).endswith("?")
    rising = (not is_question) and slope > FALLING_SLOPE_ACCEPT
    fast = spw < TITLE_MIN_SECONDS_PER_WORD
    if (rising or fast) and not force_write and force_seed is None:
        reasons = []
        if fast:
            reasons.append(f"FAST ({spw:.2f}<{TITLE_MIN_SECONDS_PER_WORD:.2f}s/word)")
        if rising:
            reasons.append(f"RISING (slope={slope:+.2f})")
        print(
            f"  SKIP {label}: {'; '.join(reasons)} after {tried} seed(s) -> logged",
            file=sys.stderr,
        )
        return {
            "label": label,
            "title": title,
            "fast": fast,
            "duration": round(duration, 3),
            "words": words,
            "seconds_per_word": round(spw, 3),
            "end_slope": round(slope, 3),
            "rising": rising,
            "seeds_tried": tried,
            "jingle": jingle_idx,
        }

    jingle = load_jingle(jingle_idx, sample_rate)
    overlap_s = OVERLAP_S.get(jingle_idx, DEFAULT_OVERLAP_S)
    intro = build_intro(jingle, title_wav, sample_rate, overlap_s, jingle_idx)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{label}_intro.wav"
    sf.write(str(out_path), intro, sample_rate)
    warnings = []
    if fast:
        warnings.append("still fast")
    if rising:
        warnings.append("still rising")
    warn = f"  [WARNING: {', '.join(warnings)}, written anyway]" if warnings else ""
    print(
        f"    saved {out_path.name}  intro{jingle_idx}  "
        f'title="{title}"  ({len(intro) / sample_rate:.2f}s){warn}'
    )
    return None


def build_intro(jingle: np.ndarray, title: np.ndarray, sample_rate: int,
                overlap_s: float, jingle_idx: int) -> np.ndarray:
    """Mix jingle + title with sidechain ducking.

    Layout (samples):
      [lead silence][jingle .......................]
                           [title .........]
      The jingle plays at FULL volume, dipped to DUCK_LEVEL only WHILE the title
      speaks (with short ramps), then recovered to full so the jingle's tail rings
      out. A per-jingle END_FADE_S at the very end winds it down so it doesn't
      sound cut. The summed result is peak-normalized.
    """
    lead = int(LEAD_SILENCE_S * sample_rate)
    trail = int(TRAIL_SILENCE_S * sample_rate)
    overlap = int(overlap_s * sample_rate)
    ramp = int(DUCK_RAMP_S * sample_rate)
    jlen = len(jingle)
    tlen = len(title)

    title_start = lead + max(0, jlen - overlap)

    # Gain envelope over the jingle: 1.0 everywhere, dipped to DUCK_LEVEL for the
    # span the title occupies (positions relative to the jingle's own start).
    env = np.ones(jlen, dtype=np.float32)
    d0 = max(0, min(jlen, title_start - lead))          # title start within jingle
    d1 = max(0, min(jlen, title_start + tlen - lead))    # title end within jingle
    r_in = max(0, d0 - ramp)
    if d0 > r_in:
        env[r_in:d0] = np.linspace(1.0, DUCK_LEVEL, d0 - r_in, dtype=np.float32)
    env[d0:d1] = DUCK_LEVEL
    r_out = min(jlen, d1 + ramp)
    if r_out > d1:
        env[d1:r_out] = np.linspace(DUCK_LEVEL, 1.0, r_out - d1, dtype=np.float32)
    # Gentle raised-cosine fade over the very end so the jingle never hard-cuts.
    fade_n = min(int(END_FADE_S.get(jingle_idx, DEFAULT_END_FADE_S) * sample_rate), jlen)
    if fade_n > 0:
        t = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        env[-fade_n:] *= 0.5 * (1.0 + np.cos(np.pi * t))

    ducked = jingle * env

    total = max(lead + jlen, title_start + tlen) + trail
    out = np.zeros(total, dtype=np.float32)
    out[lead:lead + jlen] += ducked
    out[title_start:title_start + tlen] += title

    # Peak-CAP only (not peak-normalize): preserves the title voice's
    # narration-matched loudness; scale down only if the mix would clip.
    peak = g._peak(out)
    if peak > PEAK_CEILING:
        out = (out * (PEAK_CEILING / peak)).astype(np.float32)
    return out


def report_coverage(first100: dict[str, str], tts: dict[str, str]) -> None:
    """Print which discovered stories have / lack a title."""
    jobs = g._discover_jobs()
    labels = [label for label, _txt, _voice in jobs]
    have = [l for l in labels if title_for(l, first100, tts)]
    missing = [l for l in labels if not title_for(l, first100, tts)]
    print(f"Discovered {len(labels)} stories: {len(have)} with title, {len(missing)} missing.")
    if missing:
        print("Missing titles:")
        for l in missing:
            print(f"  {l}")


# --- CLI -------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-story intro clips (jingle + title).")
    parser.add_argument("labels", nargs="*", help="label substrings to filter (default: all mapped)")
    parser.add_argument("--title", help="inline title for a single label (skips the mapping files)")
    parser.add_argument("--jingle", type=int, choices=(1, 2), help="force jingle 1 or 2 (default: hashed)")
    parser.add_argument("--speed", type=float, default=TITLE_SPEED,
                        help=f"title time-stretch, <1 slower (default {TITLE_SPEED})")
    parser.add_argument("--seed", type=int,
                        help="render this listener-chosen seed instead of sweeping "
                             "(see scripts/title_ab.py)")
    parser.add_argument("--list", action="store_true", help="print title coverage and exit")
    args = parser.parse_args()

    first100_titles, tts_titles = load_titles()

    if args.list:
        report_coverage(first100_titles, tts_titles)
        return 0

    print("Loading VoxCPM model (once)...", file=sys.stderr)
    model = g.VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    sample_rate = model.tts_model.sample_rate

    # Inline single-title audition (Cinderella smoke test): needs exactly one label.
    if args.title is not None:
        if len(args.labels) != 1:
            print("ERROR: --title requires exactly one label, e.g. "
                  '`make_intros.py batch01_story_001 --title "Cinderella"`', file=sys.stderr)
            return 2
        label = args.labels[0]
        jingle_idx = args.jingle or pick_jingle(label)
        make_one(model, sample_rate, label, args.title, jingle_idx, args.speed,
                 force_write=True,
                 force_seed=args.seed or TITLE_SEED_OVERRIDES.get(label))
        print(f"\nDone. Intro in {OUTPUT_DIR}")
        return 0

    # Batch: every DISCOVERED story that has a title, optionally filtered by the
    # positional substrings. Labels come from generate_with_pauses._discover_jobs()
    # (prefixed, e.g. age4-5_story_251 / batch01_story_001); titles are looked up by
    # the bare story id, routed by prefix (batch* -> tts map, else First100 map).
    jobs = g._discover_jobs()
    targets = [label for label, _txt, _voice in jobs]
    if args.labels:
        targets = [l for l in targets if any(sub in l for sub in args.labels)]

    if not targets:
        print("No labels to render. Use --list to see coverage, or --title for a one-off.",
              file=sys.stderr)
        return 1

    made = 0
    skipped: list[dict] = []
    for label in targets:
        title = title_for(label, first100_titles, tts_titles)
        if not title:
            print(f"  SKIP {label}: no title in mapping files", file=sys.stderr)
            continue
        jingle_idx = args.jingle or pick_jingle(label)
        rec = make_one(model, sample_rate, label, title, jingle_idx, args.speed,
                       force_seed=args.seed or TITLE_SEED_OVERRIDES.get(label))
        if rec is None:
            made += 1
        else:
            skipped.append(rec)

    print(f"\nDone. {made} intro(s) in {OUTPUT_DIR}")
    if skipped:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rising = [rec for rec in skipped if rec["rising"]]
        fast = [rec for rec in skipped if rec["fast"]]
        if rising:
            with open(RISING_LOG, "w", encoding="utf-8") as fh:
                for rec in rising:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if fast:
            with open(FAST_LOG, "w", encoding="utf-8") as fh:
                for rec in fast:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"{len(skipped)} title(s) exhausted all seeds and were SKIPPED "
            f"(not overwritten): {len(fast)} fast, {len(rising)} rising.\n"
            f"  Fast queue: {FAST_LOG}\n  Rising queue: {RISING_LOG}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
