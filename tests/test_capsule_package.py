import os
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from capability_capsule.config import Settings
from capability_capsule.manifest import (
    ArtifactProvenance,
    CapsuleManifest,
    CreatedBy,
    SourceType,
)
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.storage import save_index


def _inputs(tmp_path: Path, *, budget: int = 1_000_000) -> tuple[Path, Settings, CapsuleManifest]:
    settings = Settings.model_validate({"capsule": {"size_budget_bytes": budget}})
    index = tmp_path / "index.npz"
    chunk = TextChunk(
        relative_path="docs/guide.md",
        source_type=SourceType.REPO,
        chunk_index=0,
        start_char=0,
        end_char=5,
        text="guide",
    )
    save_index(index, (chunk,), ((1.0, 0.0),), embedding_model="nomic-embed-text")
    build_id = uuid4()
    manifest = CapsuleManifest(
        capsule_build_id=build_id,
        task="continue offline development",
        size_budget_bytes=budget,
        offline_duration_hours=settings.capsule.offline_duration_hours,
        generation_model=settings.ollama.generation_model,
        embedding_model=settings.ollama.embedding_model,
        artifacts=(
            ArtifactProvenance(
                source_type=SourceType.REPO,
                source="docs/guide.md",
                created_by=CreatedBy.ORIGINAL,
                capsule_build_id=build_id,
            ),
        ),
    )
    return index, settings, manifest


def test_write_capsule_contains_valid_snapshot(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["write_capsule"])
    index, settings, manifest = _inputs(tmp_path)
    output = tmp_path / "offline capsule.zip"
    index_before = index.read_bytes()
    result = module.write_capsule(index, output, manifest, settings)
    assert result.output_path == output.resolve()
    assert result.capsule_build_id == manifest.capsule_build_id
    assert result.artifact_count == 1
    assert result.size_bytes == output.stat().st_size
    assert 0 < result.size_bytes <= settings.capsule.size_budget_bytes
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["manifest.json", "settings.json", "index.npz"]
        loaded_manifest = CapsuleManifest.model_validate_json(archive.read("manifest.json"))
        loaded_settings = Settings.model_validate_json(archive.read("settings.json"))
        assert loaded_manifest == manifest
        assert loaded_settings.model_dump(mode="json") == settings.model_dump(mode="json")
        assert archive.read("index.npz") == index_before
        assert all(not info.is_dir() for info in archive.infolist())
    assert index.read_bytes() == index_before


def test_write_capsule_refuses_existing_output(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["write_capsule"])
    index, settings, manifest = _inputs(tmp_path)
    output = tmp_path / "capsule.zip"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        module.write_capsule(index, output, manifest, settings)
    assert output.read_bytes() == b"existing"


def test_write_capsule_enforces_total_budget_and_cleans_up(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["write_capsule"])
    index, settings, manifest = _inputs(tmp_path, budget=1)
    output = tmp_path / "capsule.zip"
    with pytest.raises(ValueError):
        module.write_capsule(index, output, manifest, settings)
    assert not output.exists()
    assert not list(tmp_path.glob(".capsule-package-*"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_model", "other-model"),
        ("embedding_model", "other-embed"),
        ("size_budget_bytes", 999_999),
        ("offline_duration_hours", 3.0),
    ],
)
def test_write_capsule_rejects_manifest_settings_mismatch(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["write_capsule"])
    index, settings, manifest = _inputs(tmp_path)
    mismatched = manifest.model_copy(update={field: value})
    output = tmp_path / "capsule.zip"
    with pytest.raises(ValueError):
        module.write_capsule(index, output, mismatched, settings)
    assert not output.exists()


def test_write_capsule_validates_index_before_output(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["write_capsule"])
    _, settings, manifest = _inputs(tmp_path)
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not an index")
    output = tmp_path / "capsule.zip"
    with pytest.raises((OSError, ValueError)):
        module.write_capsule(broken, output, manifest, settings)
    assert not output.exists()


def test_write_capsule_does_not_overwrite_racing_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["write_capsule"])
    index, settings, manifest = _inputs(tmp_path)
    output = tmp_path / "capsule.zip"
    real_link = os.link

    def racing_link(source: Path, destination: Path) -> None:
        Path(destination).write_bytes(b"concurrent")
        real_link(source, destination)

    monkeypatch.setattr(module.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        module.write_capsule(index, output, manifest, settings)
    assert output.read_bytes() == b"concurrent"
    assert not list(tmp_path.glob(".capsule-package-*"))


def _valid_capsule(tmp_path: Path) -> tuple[Any, Path, Settings, CapsuleManifest]:
    module = __import__("capability_capsule.packager.capsule", fromlist=["write_capsule"])
    index, settings, manifest = _inputs(tmp_path)
    path = tmp_path / "capsule.zip"
    module.write_capsule(index, path, manifest, settings)
    return module, path, settings, manifest


def _rewrite_capsule(
    source: Path,
    destination: Path,
    *,
    replace: dict[str, bytes] | None = None,
    omit: set[str] | None = None,
    extra: tuple[str, bytes] | None = None,
) -> None:
    replace = replace or {}
    omit = omit or set()
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist() if name not in omit}
    members.update(replace)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for name in ("manifest.json", "settings.json", "index.npz"):
            if name in members:
                archive.writestr(name, members[name])
        if extra is not None:
            archive.writestr(*extra)


def test_inspect_capsule_validates_and_returns_metadata(tmp_path: Path) -> None:
    module, path, settings, manifest = _valid_capsule(tmp_path)
    before = path.read_bytes()
    inspected = module.inspect_capsule(path)
    assert inspected.path == path.resolve()
    assert inspected.manifest == manifest
    assert inspected.settings.model_dump(mode="json") == settings.model_dump(mode="json")
    assert inspected.index_size_bytes > 0
    assert inspected.package_size_bytes == path.stat().st_size
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("omit", "extra"),
    [
        ({"manifest.json"}, None),
        ({"settings.json"}, None),
        ({"index.npz"}, None),
        (set(), ("unexpected.txt", b"extra")),
        (set(), ("../outside.txt", b"unsafe")),
    ],
)
def test_inspect_capsule_rejects_wrong_members(
    tmp_path: Path,
    omit: set[str],
    extra: tuple[str, bytes] | None,
) -> None:
    module, path, _, _ = _valid_capsule(tmp_path)
    changed = tmp_path / "changed.zip"
    _rewrite_capsule(path, changed, omit=omit, extra=extra)
    with pytest.raises(ValueError):
        module.inspect_capsule(changed)


@pytest.mark.parametrize(
    ("member", "content"),
    [
        ("manifest.json", b"not json"),
        ("settings.json", b"[]"),
        ("index.npz", b"not an index"),
    ],
)
def test_inspect_capsule_rejects_invalid_content(
    tmp_path: Path,
    member: str,
    content: bytes,
) -> None:
    module, path, _, _ = _valid_capsule(tmp_path)
    changed = tmp_path / "changed.zip"
    _rewrite_capsule(path, changed, replace={member: content})
    with pytest.raises((OSError, ValueError)):
        module.inspect_capsule(changed)


def test_inspect_capsule_rejects_snapshot_mismatch(tmp_path: Path) -> None:
    module, path, _, _ = _valid_capsule(tmp_path)
    with zipfile.ZipFile(path) as archive:
        settings = Settings.model_validate_json(archive.read("settings.json"))
    settings.ollama.generation_model = "different-model"
    changed = tmp_path / "changed.zip"
    _rewrite_capsule(
        path,
        changed,
        replace={"settings.json": settings.model_dump_json().encode()},
    )
    with pytest.raises(ValueError):
        module.inspect_capsule(changed)


def test_inspect_capsule_enforces_declared_package_budget(tmp_path: Path) -> None:
    module, path, _, manifest = _valid_capsule(tmp_path)
    tiny_settings = Settings.model_validate({"capsule": {"size_budget_bytes": 1}})
    tiny_manifest = manifest.model_copy(update={"size_budget_bytes": 1})
    changed = tmp_path / "changed.zip"
    _rewrite_capsule(
        path,
        changed,
        replace={
            "manifest.json": tiny_manifest.model_dump_json().encode(),
            "settings.json": tiny_settings.model_dump_json().encode(),
        },
    )
    with pytest.raises(ValueError):
        module.inspect_capsule(changed)


def test_inspect_capsule_rejects_missing_and_non_zip_files(tmp_path: Path) -> None:
    module = __import__("capability_capsule.packager.capsule", fromlist=["inspect_capsule"])
    with pytest.raises(FileNotFoundError):
        module.inspect_capsule(tmp_path / "missing.zip")
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not a zip archive")
    with pytest.raises(ValueError):
        module.inspect_capsule(invalid)


def test_inspect_capsule_rejects_duplicate_members(tmp_path: Path) -> None:
    module, path, _, _ = _valid_capsule(tmp_path)
    changed = tmp_path / "duplicate.zip"
    changed.write_bytes(path.read_bytes())
    with pytest.warns(UserWarning), zipfile.ZipFile(changed, "a") as archive:
        archive.writestr("manifest.json", b"{}")
    with pytest.raises(ValueError):
        module.inspect_capsule(changed)
