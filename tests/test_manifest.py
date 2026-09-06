from uuid import uuid4

import pytest

from capability_capsule.manifest import (
    ArtifactProvenance,
    CapsuleManifest,
    CreatedBy,
    SourceType,
)


def test_manifest_accepts_matching_artifact() -> None:
    build_id = uuid4()
    artifact = ArtifactProvenance(
        source_type=SourceType.REPO,
        source="src/example.py",
        created_by=CreatedBy.ORIGINAL,
        capsule_build_id=build_id,
    )

    manifest = CapsuleManifest(
        capsule_build_id=build_id,
        task="Continue offline repository development",
        size_budget_bytes=1_073_741_824,
        offline_duration_hours=12,
        generation_model="qwen3.5:4b",
        embedding_model="nomic-embed-text",
        artifacts=(artifact,),
    )

    assert manifest.schema_version == "0.1"
    assert manifest.artifacts == (artifact,)
    assert manifest.created_at.utcoffset() is not None


def test_manifest_rejects_artifact_from_another_build() -> None:
    artifact = ArtifactProvenance(
        source_type=SourceType.LOCAL_DOCUMENT,
        source="notes/design.docx",
        created_by=CreatedBy.ORIGINAL,
        capsule_build_id=uuid4(),
    )

    with pytest.raises(ValueError):
        CapsuleManifest(
            capsule_build_id=uuid4(),
            task="Continue offline repository development",
            size_budget_bytes=1_073_741_824,
            offline_duration_hours=12,
            generation_model="qwen3.5:4b",
            embedding_model="nomic-embed-text",
            artifacts=(artifact,),
        )
