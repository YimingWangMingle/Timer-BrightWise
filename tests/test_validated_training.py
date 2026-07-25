import json

import torch
from torch.utils.data import Dataset

from tsfm.config import TimerConfig
from tsfm.model import TimerModel
from tsfm.train_config import TrainingConfig
from tsfm.validated_training import run_validated_training


class Windows(Dataset):
    def __len__(self): return 100
    def __getitem__(self, index):
        values = torch.arange(12, dtype=torch.float32) + index * 0.01
        context, target = values[:8], values[4:]
        mean = context.mean().reshape(1); scale = torch.sqrt(context.var(unbiased=False)+1e-5).reshape(1)
        return {"context":(context-mean)/scale,"target":(target-mean)/scale,"mean":mean,"scale":scale,"source_id":"synthetic","record_id":f"s/{index}","sample_index":index,"window_start":0}


def test_validated_training_records_both_views_and_validation_best(tmp_path) -> None:
    model = TimerModel(TimerConfig(input_token_len=4,output_token_len=4,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=8))
    config = TrainingConfig(total_steps=1,warmup_steps=0,peak_lr=1e-3,micro_batch_size=1,gradient_accumulation_steps=1,validation_interval=1,checkpoint_interval=1,num_workers=1,prefetch_factor=1,pin_memory=False,context_patches=2)
    report = run_validated_training(model=model,train_dataset=Windows(),validation_datasets={"val_heldout":Windows(),"val_temporal":Windows()},training_config=config,output_dir=tmp_path,manifest_checksum="f"*64,config_snapshots={"model":{},"data":{},"training":config.to_dict()},device=torch.device("cpu"),validation_batches=1)
    lines = [json.loads(line) for line in (tmp_path/"validation-report.jsonl").read_text().splitlines()]
    assert report.checkpoint_path.is_file()
    assert (tmp_path/"best.pt").is_file()
    assert set(lines[0]["views"]) == {"val_heldout","val_temporal"}
