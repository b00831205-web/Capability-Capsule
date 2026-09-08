"""Create a portable capsule archive from a validated vector index."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import httpx
from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.config import Settings
from capability_capsule.manifest import (
    ArtifactProvenance,
    CapsuleManifest,
    CreatedBy,
    SourceType,
)
from capability_capsule.packager.build import build_index
from capability_capsule.rag.storage import load_index

_EXPECTED_MEMBERS = ("manifest.json", "settings.json", "index.npz")

_MAX_METADATA_BYTES = 1_000_000


class CapsulePackageResult(BaseModel):
    """Summary of a completed capsule packaging operation"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Path
    capsule_build_id: UUID
    artifact_count: int = Field(ge=0)
    size_bytes: int = Field(gt=0)


class CapsuleInspection(BaseModel):
    """Validated metadata from an existing capsule archive"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    manifest: CapsuleManifest
    settings: Settings
    index_size_bytes: int = Field(gt=0)
    package_size_bytes: int = Field(gt=0)


class CapsuleBuildResult(BaseModel):
    """Summary of a repository-to-capsule build"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Path
    manifest: CapsuleManifest
    document_count: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    vector_dimensions: int = Field(gt=0)
    size_bytes: int = Field(gt=0)


def _validate_manifest_settings(manifest: CapsuleManifest, settings: Settings) -> None:
    """Ensure the manifest describes the configuration being packaged"""

    if manifest.generation_model != settings.ollama.generation_model:
        raise ValueError("Manifest generation model does not match settings")

    if manifest.embedding_model != settings.ollama.embedding_model:
        raise ValueError("Manifest embedding model does not match settings")

    if manifest.size_budget_bytes != settings.capsule.size_budget_bytes:
        raise ValueError("Manifest size budget does not match settings")

    if manifest.offline_duration_hours != settings.capsule.offline_duration_hours:
        raise ValueError("Manifest offline duration does not match settings")


def write_capsule(
    index_path: Path, output_path: Path, manifest: CapsuleManifest, settings: Settings
) -> CapsulePackageResult:
    """Validate and package an index, manifest, and settings snapshot."""

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)

    destination = output_path.resolve()

    if not destination.parent.is_dir():
        raise NotADirectoryError(destination.parent)

    index = index_path.resolve(strict=True)

    if index_path.is_symlink() or not index.is_file():
        raise FileNotFoundError(index_path)

    _validate_manifest_settings(manifest, settings)

    load_index(index, expected_embedding_model=settings.ollama.embedding_model)

    with TemporaryDirectory(
        prefix=".capsule-package-",
        dir=destination.parent,
    ) as temporary_directory:
        staged_archive = Path(temporary_directory) / "capsule.zip"

        with ZipFile(staged_archive, mode="x", compression=ZIP_STORED, allowZip64=True) as archive:
            archive.writestr(
                "manifest.json",
                manifest.model_dump_json(indent=2),
            )
            archive.writestr("settings.json", settings.model_dump_json(indent=2))
            archive.write(index, arcname="index.npz")

        actual_size = staged_archive.stat().st_size
        budget = settings.capsule.size_budget_bytes

        if actual_size > budget:
            raise ValueError(f"Capsule size {actual_size} bytes exceeds budget {budget} bytes")

        os.link(staged_archive, destination)

    return CapsulePackageResult(
        output_path=destination,
        capsule_build_id=manifest.capsule_build_id,
        artifact_count=len(manifest.artifacts),
        size_bytes=destination.stat().st_size,
    )


def inspect_capsule(path: Path) -> CapsuleInspection:
    """Validate a capsule archive without extracting it into the workspace"""

    if path.is_symlink():
        raise ValueError("Capsule path must not be a symbolic link")

    source = path.resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(path)

    package_size = source.stat().st_size

    try:
        with ZipFile(source, mode="r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]

            if len(names) != len(_EXPECTED_MEMBERS) or set(names) != set(_EXPECTED_MEMBERS):
                raise ValueError("Capsule must contain exactly the expected members")
            if any(member.is_dir() for member in members):
                raise ValueError("Capsule members must all be files")

            members_by_name = {member.filename: member for member in members}

            for metadata_name in {"manifest.json", "settings.json"}:
                if members_by_name[metadata_name].file_size > _MAX_METADATA_BYTES:
                    raise ValueError(f"{metadata_name} exceeds the metadata size limit")

            manifest = CapsuleManifest.model_validate_json(archive.read("manifest.json"))
            settings = Settings.model_validate_json(archive.read("settings.json"))

            _validate_manifest_settings(manifest, settings)

            budget = settings.capsule.size_budget_bytes
            unpacked_size = sum(member.file_size for member in members)

            if package_size > budget or unpacked_size > budget:
                raise ValueError("Capsule exceeds its declared size budget")

            index_info = members_by_name["index.npz"]

            if index_info.file_size <= 0:
                raise ValueError("Capsule index must not be empty")

            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Capsule member failed CRC validation: {bad_member}")
            index_bytes = archive.read("index.npz")

    except BadZipFile as error:
        raise ValueError("Capsule is not a valida ZIP archive") from error

    with TemporaryDirectory(prefix=".capsule-inspect-") as temporary_directory:
        temporary_index = Path(temporary_directory) / "index.npz"
        temporary_index.write_bytes(index_bytes)

        load_index(temporary_index, expected_embedding_model=settings.ollama.embedding_model)

    return CapsuleInspection(
        path=source,
        manifest=manifest,
        settings=settings,
        index_size_bytes=index_info.file_size,
        package_size_bytes=package_size,
    )


def build_capsule(
    root: Path,
    output_path: Path,
    task: str,
    settings: Settings,
    *,
    chunk_size_chars: int = 1_000,
    overlap_chars: int = 200,
    batch_size: int = 32,
    max_file_size_bytes: int = 1_000_000,
    transport: httpx.BaseTransport | None = None,
) -> CapsuleBuildResult:
    """Build an index and package it with provenance and settings"""

    if not task.strip():
        raise ValueError("Task must not be blank")

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)

    destination = output_path.resolve()

    if not destination.parent.is_dir():
        raise NotADirectoryError(destination.parent)

    build_id = uuid4()

    with TemporaryDirectory(
        prefix=".capsule-build-",
        dir=destination.parent,
    ) as temporary_directory:
        temporary_index = Path(temporary_directory) / "index.npz"

        index_result = build_index(
            root,
            temporary_index,
            settings,
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars,
            batch_size=batch_size,
            max_file_size_bytes=max_file_size_bytes,
            transport=transport,
        )

        index = load_index(
            temporary_index,
            expected_embedding_model=settings.ollama.embedding_model,
        )

        source_keys: list[tuple[SourceType, str]] = []
        seen_sources: set[tuple[SourceType, str]] = set()

        for chunk in index.chunks:
            key = (
                chunk.source_type,
                chunk.relative_path,
            )

            if key not in seen_sources:
                seen_sources.add(key)
                source_keys.append(key)

        artifacts = tuple(
            ArtifactProvenance(
                source_type=source_type,
                source=source,
                created_by=(
                    CreatedBy.CLOUD_TEACHER
                    if source_type is SourceType.PREFETCHED
                    else CreatedBy.ORIGINAL
                ),
                capsule_build_id=build_id,
            )
            for source_type, source in source_keys
        )

        manifest = CapsuleManifest(
            capsule_build_id=build_id,
            task=task,
            size_budget_bytes=settings.capsule.size_budget_bytes,
            offline_duration_hours=(settings.capsule.offline_duration_hours),
            generation_model=settings.ollama.generation_model,
            embedding_model=settings.ollama.embedding_model,
            artifacts=artifacts,
        )

        package_result = write_capsule(
            temporary_index,
            destination,
            manifest,
            settings,
        )

    return CapsuleBuildResult(
        output_path=package_result.output_path,
        manifest=manifest,
        document_count=index_result.document_count,
        chunk_count=index_result.chunk_count,
        vector_dimensions=index_result.vector_dimensions,
        size_bytes=package_result.size_bytes,
    )
