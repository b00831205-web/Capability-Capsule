"""Training-data export and minimal adapter training support."""

from capability_capsule.training.lora import (
    BackendCheckpoint,
    LoraTrainingConfig,
    LoraTrainingResult,
    SavedAdapterCheckpoint,
    TrainerProcessEvent,
    TrainingMetric,
    reload_lora_adapter,
    run_lora_training,
)
from capability_capsule.training.sft import (
    SFTArtifact,
    SFTExample,
    SFTExportManifest,
    encode_sft_trajectory,
    export_sft_dataset,
    load_sft_examples,
    SFTChatContract,
)
from capability_capsule.training.transformers_peft import (
    TransformersPeftAdapterLoader,
    TransformersPeftBackend,
)

__all__ = [
    "BackendCheckpoint",
    "LoraTrainingConfig",
    "LoraTrainingResult",
    "SFTArtifact",
    "SFTExample",
    "SFTExportManifest",
    "SavedAdapterCheckpoint",
    "TrainerProcessEvent",
    "TrainingMetric",
    "TransformersPeftAdapterLoader",
    "TransformersPeftBackend",
    "encode_sft_trajectory",
    "export_sft_dataset",
    "load_sft_examples",
    "reload_lora_adapter",
    "run_lora_training",
    "SFTChatContract"
]