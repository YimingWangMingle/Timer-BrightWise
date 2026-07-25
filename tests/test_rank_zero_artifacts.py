import torch
from torch.utils.data import Dataset

from tsfm.config import TimerConfig
from tsfm.model import TimerModel
from tsfm.train_config import TrainingConfig
from tsfm.trainer import run_training


class OneWindow(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index):
        values = torch.arange(12, dtype=torch.float32)
        return {
            "context": values[:8],
            "target": values[4:],
            "mean": torch.zeros(1),
            "scale": torch.ones(1),
            "source_id": "synthetic",
            "record_id": "synthetic/0",
            "sample_index": index,
            "window_start": 0,
        }


def test_non_main_rank_writes_no_checkpoints_or_reports(tmp_path) -> None:
    model = TimerModel(
        TimerConfig(
            input_token_len=4,
            output_token_len=4,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
        )
    )
    config = TrainingConfig(
        total_steps=1,
        warmup_steps=0,
        peak_lr=1e-3,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=False,
        context_patches=2,
    )

    report = run_training(
        model=model,
        dataset=OneWindow(),
        training_config=config,
        output_dir=tmp_path,
        manifest_checksum="e" * 64,
        config_snapshots={"model": {}, "data": {}, "training": config.to_dict()},
        device=torch.device("cpu"),
        is_main_process=False,
    )

    assert report.checkpoint_path is None
    assert not list(tmp_path.glob("*.pt"))
    assert not (tmp_path / "training-report.json").exists()
