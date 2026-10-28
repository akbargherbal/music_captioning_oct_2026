"""Pure-Python ABC parsing and QC for SheetSage2 scores.

No torch, no GPU, no network. Used by ``scripts/abc_pack.py`` (pack + QC),
``scripts/abc_batch.py`` (per-track gate) and ``scripts/build_abc_dataset.py``.

The ABC dialect is the one SheetSage2 emits (``notation_sheetsage2.py``):

    X:1 / T: / M:4/4 / L:1/32 / Q:1/4=101
    V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    V: Ins clef=treble name="Ins Melody" snm="Inst."
    K:Em
    V: Vocal
    "Em"z32|"C"e2f2...|
    V: Ins
    Z|e2f2...|

Chords are quoted tokens and appear only on the Vocal voice. ``--melody-only``
does not merely omit them: chord changes can split a held note into tied
tokens, so a chord-stripped full score is *musically* equivalent but not always
*textually* equal to a real melody-only run. That is measured, not assumed.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

VOICE_DECL_RE = re.compile(r"^V:\s*([^\s]+)")
HEADER_RE = re.compile(r"^([A-Za-z]):\s*(.*)$")
CHORD_TOKEN_RE = re.compile(r'"[^"\n]*"')
TEMPO_RE = re.compile(r"Q:\s*(?:\d+/\d+)?\s*=?\s*(\d+(?:\.\d+)?)")
NOTE_RE = re.compile(
    r"(?P<acc>\^{1,2}|_{1,2}|=)?"
    r"(?P<letter>[A-Ga-g])"
    r"(?P<octave>[,']*)"
    r"(?P<duration>\d*(?:/\d+)?)"
    r"(?P<tie>-?)"
)
REST_RE = re.compile(r"[zZ]\d*(?:/\d+)?-?")
INLINE_DIRECTIVE_RE = re.compile(r"\[[A-Za-z]:[^\]]*\]")

_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
VOICE_NAMES = ("Vocal", "Ins")
# E3-A5 in MIDI note numbers.
VOCAL_RANGE = (52, 81)


@dataclass
class Note:
    midi: int
    units: int
    text: str
    gap_before: int = 0


@dataclass
class ParsedAbc:
    text: str
    voices: dict[str, list[Note]] = field(default_factory=dict)
    meters: list[str] = field(default_factory=list)
    tempos: list[float] = field(default_factory=list)
    key: str = ""
    measures: int = 0
    chord_symbols: list[str] = field(default_factory=list)
    structure_labels: list[str] = field(default_factory=list)

    @property
    def notes_vocal(self) -> int:
        return len(self.voices.get("Vocal", []))

    @property
    def notes_ins(self) -> int:
        return len(self.voices.get("Ins", []))

    @property
    def has_chords(self) -> bool:
        return bool(self.chord_symbols)

    @property
    def tempo_bpm(self) -> float | None:
        return self.tempos[0] if self.tempos else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_title(title: str) -> str:
    """Title key for duplicate detection (``(Music)`` is not a new title)."""
    base = re.sub(r"\s*[\(\[]\s*music\s*[\)\]]\s*$", "", title.strip(),
                  flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", base).strip().lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _duration_units(text: str) -> int:
    """ABC duration text -> integer units of the file's ``L:`` value.

    Empty means one unit. ``4`` means four. Fractions are floored, which is
    enough for phrase/rest segmentation (never for re-serialization).
    """
    if not text:
        return 1
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            n = int(num) if num else 1
            d = int(den) if den else 2
            return max(1, n // d)
        except ValueError:
            return 1
    try:
        return max(1, int(text))
    except ValueError:
        return 1


def note_to_midi(acc: str | None, letter: str, octave_marks: str) -> int:
    """ABC note -> MIDI number (C4 = 60, ``c`` = C5, ``,`` lowers octave)."""
    octave = 5 if letter.islower() else 4
    octave += octave_marks.count("'") - octave_marks.count(",")
    midi = 12 * (octave + 1) + _SEMITONE[letter.upper()]
    if acc:
        if acc.startswith("^"):
            midi += len(acc)
        elif acc.startswith("_"):
            midi -= len(acc)
    return midi


def strip_chords(abc: str) -> str:
    """Remove quoted chord symbols from music lines only.

    Header voice declarations contain ``name="..."`` and must survive, so
    lines beginning with ``V:`` (and other header fields) are left alone.
    """
    out: list[str] = []
    for line in abc.splitlines():
        stripped = line.lstrip()
        is_field = bool(HEADER_RE.match(stripped)) or stripped.startswith("%")
        if is_field:
            out.append(line)
            continue
        cleaned = CHORD_TOKEN_RE.sub("", line)
        cleaned = re.sub(r"  +", " ", cleaned)
        out.append(cleaned)
    return "\n".join(out) + ("\n" if abc.endswith("\n") else "")


def parse_abc(abc: str) -> ParsedAbc:
    parsed = ParsedAbc(text=abc)
    current = ""
    for line in abc.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("%"):
            label = stripped.lstrip("%").strip()
            if label:
                parsed.structure_labels.append(label)
            continue

        voice = VOICE_DECL_RE.match(stripped)
        if voice:
            current = voice.group(1)
            parsed.voices.setdefault(current, [])
            continue

        header = HEADER_RE.match(stripped)
        if header:
            field, value = header.group(1).upper(), header.group(2).strip()
            if field == "M":
                parsed.meters.append(value)
            elif field == "K":
                parsed.key = value
            elif field == "Q":
                m = TEMPO_RE.search(stripped)
                if m:
                    parsed.tempos.append(float(m.group(1)))
            continue

        if not current:
            continue
        music = INLINE_DIRECTIVE_RE.sub("", line)
        chords = CHORD_TOKEN_RE.findall(music)
        parsed.chord_symbols.extend(c.strip('"') for c in chords)
        music = CHORD_TOKEN_RE.sub("", music)
        notes = parsed.voices.setdefault(current, [])
        rest_units = 0
        for match in re.finditer(
                rf"{REST_RE.pattern}|{NOTE_RE.pattern}", music):
            token = match.group(0)
            if token[:1] in ("z", "Z"):
                rest_units += _duration_units(
                    token[1:].rstrip("-"))
                continue
            m = NOTE_RE.fullmatch(token)
            if not m:
                continue
            units = _duration_units(m.group("duration"))
            notes.append(Note(
                midi=note_to_midi(m.group("acc"), m.group("letter"),
                                  m.group("octave")),
                units=units,
                text=token,
                gap_before=rest_units,
            ))
            rest_units = 0
    bars = list(_voice_bar_counts(abc).values())
    parsed.measures = max(bars) if bars else 0
    return parsed


def _voice_bar_counts(abc: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    current = ""
    for line in abc.splitlines():
        stripped = line.strip()
        voice = VOICE_DECL_RE.match(stripped)
        if voice:
            current = voice.group(1)
            counts.setdefault(current, 0)
            continue
        if not current or not stripped or stripped.startswith(("%", "X:", "T:", "M:", "L:", "Q:", "K:")):
            continue
        if HEADER_RE.match(stripped):
            continue
        counts[current] = counts.get(current, 0) + line.count("|")
    return counts


def _lab_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(line.split("\t"))
    return rows


def _phrase_medians(notes: list[Note], rest_gap_units: int = 8) -> list[list[int]]:
    """Split a voice into phrases at long rests.

    ``L:1/32`` is the SheetSage2 default, so 8 units is a quarter-note rest.
    """
    phrases: list[list[int]] = []
    current: list[int] = []
    for note in notes:
        if current and note.gap_before >= rest_gap_units:
            phrases.append(current)
            current = []
        current.append(note.midi)
    if current:
        phrases.append(current)
    return phrases


def _octave_suspect(notes: list[Note]) -> bool:
    """Heuristic for the documented phrase-level ±12 SheetSage2 failure.

    Never corrects anything. Flags when a >=10-note phrase sits a whole octave
    away from the track-wide median (>=10 notes). This is deliberately
    conservative; a human still listens.
    """
    if len(notes) < 20:
        return False
    global_median = statistics.median(n.midi for n in notes)
    for phrase in _phrase_medians(notes):
        if len(phrase) < 10:
            continue
        if abs(abs(statistics.median(phrase) - global_median) - 12) <= 0.5:
            return True
    return False


def _last_beat_seconds(raw_dir: Path) -> float | None:
    rows = _lab_rows(raw_dir / "beat.lab")
    last = None
    for row in rows:
        try:
            last = float(row[0])
        except (ValueError, IndexError):
            continue
    return last


def run_abc2midi(abc: str, timeout: int = 120) -> dict:
    """Parse gate: exit 0 and no 'not complete' diagnostics."""
    if shutil.which("abc2midi") is None:
        return {"exit": None, "ok": False,
                "detail": "abc2midi not installed (apt-get install abcmidi)"}
    with tempfile.TemporaryDirectory(prefix="abc2midi-") as tmp:
        src = Path(tmp) / "score.abc"
        src.write_text(abc, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["abc2midi", str(src), "-o", str(Path(tmp) / "out.mid")],
                capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"exit": None, "ok": False, "detail": str(exc)}
    output = f"{proc.stdout}\n{proc.stderr}"
    incomplete = "not complete" in output.lower()
    ok = proc.returncode == 0 and not incomplete
    detail = "" if ok else (output.strip()[-500:] or f"exit {proc.returncode}")
    return {"exit": proc.returncode, "ok": ok, "detail": detail}


def qc_track(raw_dir: Path, duration_s: float | None, *,
             file: str = "", abc: str | None = None) -> dict:
    """Run the §9 QC gate for one transcript directory.

    Returns a JSON-serializable dict. ``flagged != failed``: flags travel with
    the row, and nothing is repaired.
    """
    raw_dir = Path(raw_dir)
    if abc is None:
        abc_path = raw_dir / "score.abc"
        abc = abc_path.read_text(encoding="utf-8") if abc_path.is_file() else ""
    parsed = parse_abc(abc) if abc else ParsedAbc(text="")
    flags: list[str] = []

    voices = {name: len(parsed.voices.get(name, [])) for name in VOICE_NAMES}
    if not abc:
        flags.append("no_abc")
    for name in VOICE_NAMES:
        if name not in parsed.voices:
            flags.append(f"voice_missing:{name}")

    instrumental = ("(music)" in file.lower()) and parsed.notes_vocal < 50
    if parsed.notes_vocal < 50 and not instrumental:
        flags.append("vocal_sparse")

    last_beat = _last_beat_seconds(raw_dir)
    if last_beat is not None and duration_s:
        coverage = last_beat / duration_s
    elif parsed.measures and parsed.tempo_bpm and parsed.meters:
        # Fallback only: bars * meter * beat duration (not tempo-change aware).
        try:
            num, den = (int(x) for x in parsed.meters[-1].split("/")[:2])
            coverage = parsed.measures * (num / den) * (60.0 / parsed.tempo_bpm) / duration_s
        except (ValueError, ZeroDivisionError):
            coverage = None
    else:
        coverage = None
    if coverage is not None and coverage < 0.80:
        flags.append("coverage_low")

    vocal_notes = parsed.voices.get("Vocal", [])
    vocal_median = (statistics.median(n.midi for n in vocal_notes)
                    if vocal_notes else None)
    in_range = (vocal_median is not None
                and VOCAL_RANGE[0] <= vocal_median <= VOCAL_RANGE[1])
    if vocal_notes and not in_range:
        flags.append("vocal_range")

    key_rows = _lab_rows(raw_dir / "key.lab")
    chord_rows = _lab_rows(raw_dir / "chord.lab")
    if not key_rows or not chord_rows:
        flags.append("key_chord_missing")

    tempo = parsed.tempo_bpm
    if tempo is None or not 40 <= tempo <= 200:
        flags.append("tempo_out_of_range")

    octave = "suspect" if _octave_suspect(vocal_notes) else "ok"
    if octave == "suspect":
        flags.append("octave_suspect")

    midi_gate = run_abc2midi(abc) if abc else {"exit": None, "ok": False,
                                              "detail": "no abc"}
    if not midi_gate["ok"]:
        flags.append("parse_fail")

    return {
        "parse": "ok" if midi_gate["ok"] else "fail",
        "parse_exit": midi_gate["exit"],
        "parse_detail": midi_gate["detail"],
        "voices": voices,
        "notes_vocal": parsed.notes_vocal,
        "notes_ins": parsed.notes_ins,
        "vocal_median_midi": vocal_median,
        "vocal_range": "ok" if in_range else "out",
        "octave": octave,
        "coverage": round(coverage, 3) if coverage is not None else None,
        "key": parsed.key,
        "tempo_bpm": tempo,
        "meters": parsed.meters,
        "sections": len(parsed.structure_labels),
        "has_chords": parsed.has_chords,
        "chord_symbols": len(parsed.chord_symbols),
        "measures": parsed.measures,
        "instrumental": instrumental,
        "flags": sorted(set(flags)),
    }
