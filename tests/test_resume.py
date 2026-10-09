"""Pure-Python tests for the resume/manifest helpers. No GPU, network, or model."""

import json
from pathlib import Path

from src.resume import (
    album_key,
    album_output_path,
    atomic_write_json,
    atomic_write_text,
    group_tracks,
    is_done,
    load_manifest,
    parse_gs_track,
    pending_albums,
)

ROOT = "gs://bucket/DISCOGRAPHY"


def test_album_key_and_output_path(tmp_path: Path):
    assert album_key("ELISSA", "Album") == "ELISSA/Album"
    assert album_output_path(tmp_path, "ELISSA", "Album") == tmp_path / "ELISSA" / "Album.json"


def test_parse_gs_track_valid():
    uri = f"{ROOT}/ELISSA/Elissa - 2008 Ayami Beek/01. Ayami Bik.mp3"
    info = parse_gs_track(uri, ROOT)
    assert info is not None
    assert info["artist"] == "ELISSA"
    assert info["album"] == "Elissa - 2008 Ayami Beek"
    assert info["filename"] == "01. Ayami Bik.mp3"
    assert info["source_dir"] == f"{ROOT}/ELISSA/Elissa - 2008 Ayami Beek"


def test_parse_gs_track_rejects():
    assert parse_gs_track("gs://other/x/y/z.mp3", ROOT) is None
    assert parse_gs_track(f"{ROOT}/ELISSA/Album/cover.jpg", ROOT) is None
    assert parse_gs_track(f"{ROOT}/ELISSA/track.mp3", ROOT) is None


def test_group_tracks_groups_and_sorts():
    uris = [
        f"{ROOT}/ELISSA/Album B/02. Two.mp3",
        f"{ROOT}/ELISSA/Album B/01. One.mp3",
        f"{ROOT}/ELISSA/Album B/Covers/x.jpg",
        f"{ROOT}/FADL/Album A/01. Song.mp3",
    ]
    albums = group_tracks(uris, ROOT)
    assert set(albums) == {"ELISSA/Album B", "FADL/Album A"}
    assert [t["filename"] for t in albums["ELISSA/Album B"]["tracks"]] == [
        "01. One.mp3", "02. Two.mp3",
    ]


def test_atomic_write_text_creates_and_replaces(tmp_path: Path):
    target = tmp_path / "a" / "b.json"
    atomic_write_text(target, "first")
    assert target.read_text() == "first"
    atomic_write_text(target, "second")
    assert target.read_text() == "second"
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_json_roundtrip(tmp_path: Path):
    target = tmp_path / "m.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_load_manifest_missing_and_corrupt(tmp_path: Path):
    assert load_manifest(tmp_path / "nope.json")["albums"] == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_manifest(bad)["albums"] == {}


def test_is_done_requires_file(tmp_path: Path):
    outputs = tmp_path / "outputs"
    manifest = {"albums": {"ELISSA/A": {"path": "ELISSA/A.json"}}}
    assert is_done("ELISSA/A", manifest, outputs) is False
    (outputs / "ELISSA").mkdir(parents=True)
    (outputs / "ELISSA" / "A.json").write_text("{}")
    assert is_done("ELISSA/A", manifest, outputs) is True
    assert is_done("ELISSA/missing", manifest, outputs) is False


def test_pending_albums_and_force(tmp_path: Path):
    outputs = tmp_path / "outputs"
    (outputs / "ELISSA").mkdir(parents=True)
    (outputs / "ELISSA" / "A.json").write_text("{}")
    albums = {
        "ELISSA/A": {"artist": "ELISSA", "album": "A", "tracks": []},
        "ELISSA/B": {"artist": "ELISSA", "album": "B", "tracks": []},
    }
    manifest = {"albums": {"ELISSA/A": {"path": "ELISSA/A.json"}}}
    assert pending_albums(albums, manifest, outputs) == ["ELISSA/B"]
    assert pending_albums(albums, manifest, outputs, force=True) == ["ELISSA/A", "ELISSA/B"]
