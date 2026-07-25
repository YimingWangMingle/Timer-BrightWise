from __future__ import annotations
import argparse,json,sys
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Dataset
PROJECT_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT_ROOT/"src"))
from tsfm.config import TimerConfig
from tsfm.distributed import distributed_context_from_environment,wrap_model
from tsfm.model import TimerModel
from tsfm.runtime import build_s3_dataset
from tsfm.s3.sampling import CounterSampler
from tsfm.train_config import TrainingConfig
from tsfm.trainer import bounded_batch_probe,run_training,seed_everything
from tsfm.validated_training import run_validated_training

class FixedItemDataset(Dataset):
    def __init__(self,item):self.item=item
    def __len__(self):return 2**31
    def __getitem__(self,index):return self.item

def parse_args():
    p=argparse.ArgumentParser(description="Timer-style production trainer");p.add_argument("action",choices=("probe","overfit-one-batch","run","resume-check"))
    for name in ("model-config","training-config","manifest","output-dir","resume"):p.add_argument(f"--{name}",type=Path)
    p.add_argument("--steps",type=int);p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto");return p.parse_args()

def main():
    a=parse_args();missing=[n for n in ("model_config","training_config","manifest","output_dir") if getattr(a,n) is None]
    if missing:raise ValueError(f"missing required arguments: {', '.join(missing)}")
    c=distributed_context_from_environment();mc=TimerConfig.from_json(a.model_config);tc=TrainingConfig.from_json(a.training_config);seed_everything(tc.seed)
    device=torch.device("cuda",c.local_rank) if c.enabled else torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device))
    model=wrap_model(TimerModel(mc).to(device),c)
    train,digest=build_s3_dataset(a.manifest,split="train",seed=tc.seed,patch_length=mc.input_token_len,context_patches=tc.context_patches)
    snapshots={"model":asdict(mc),"data":{"manifest_checksum":digest},"training":tc.to_dict()}
    if a.action=="probe":
        loader=DataLoader(train,batch_size=tc.micro_batch_size,sampler=CounterSampler(0,c.rank,c.world_size),num_workers=tc.num_workers,pin_memory=tc.pin_memory,prefetch_factor=tc.prefetch_factor)
        result=bounded_batch_probe(model,next(iter(loader)),device)
        if c.is_main_process:print(json.dumps(result))
        return 0
    if a.action=="run":
        heldout,heldout_digest=build_s3_dataset(a.manifest,split="val_heldout",seed=tc.seed,patch_length=mc.input_token_len,context_patches=tc.context_patches)
        temporal,temporal_digest=build_s3_dataset(a.manifest,split="val_temporal",seed=tc.seed,patch_length=mc.input_token_len,context_patches=tc.context_patches)
        if {digest,heldout_digest,temporal_digest}!={digest}:raise ValueError("validation manifests differ from training manifest")
        report=run_validated_training(model=model,train_dataset=train,validation_datasets={"val_heldout":heldout,"val_temporal":temporal},training_config=tc,output_dir=a.output_dir,manifest_checksum=digest,config_snapshots=snapshots,device=device,resume=a.resume,rank=c.rank,world_size=c.world_size,is_main_process=c.is_main_process)
    else:
        dataset=train;target=a.steps
        if a.action=="overfit-one-batch":dataset=FixedItemDataset(train[c.rank]);target=min(100,tc.total_steps)
        if a.action=="resume-check":
            if a.resume is None or a.steps is None:raise ValueError("resume-check requires --resume and --steps")
            payload=torch.load(a.resume,map_location="cpu",weights_only=True);target=int(payload["training_state"]["global_step"])+a.steps
        report=run_training(model=model,dataset=dataset,training_config=tc,output_dir=a.output_dir,manifest_checksum=digest,config_snapshots=snapshots,device=device,resume=a.resume,total_steps_override=target,rank=c.rank,world_size=c.world_size,is_main_process=c.is_main_process)
    if c.is_main_process:print(json.dumps({"losses":report.losses,"checkpoint":str(report.checkpoint_path),"consumed_samples":report.consumed_samples}))
    return 0
if __name__=="__main__":raise SystemExit(main())
