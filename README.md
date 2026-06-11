# Attention_Transformer_Reproduction

This repository contains a from-scratch PyTorch implementation of the original Transformer (Vaswani et al., 2017), focused on rigorous reproduction of the paper's results on WMT 2014.

## Training Modes

- Single process (existing path): `python train.py`
- Single-node multi-GPU DDP: `torchrun --standalone --nproc_per_node=4 train.py distributed.enabled=true training.device=cuda`
- Base EN-DE profile with DDP: `torchrun --standalone --nproc_per_node=4 train.py --config-name base_en_de distributed.enabled=true training.device=cuda`

## Resume From Checkpoints

- Auto-resume latest checkpoint is enabled by default: `training.auto_resume_latest=true`
- Explicit resume path: `training.resume_from=checkpoints/base_en_de_v1/step_050000.pt`

## DDP Preflight On Single GPU / Local Machine

- DDP path with one process (useful before running on 2-8 GPUs):
	- `torchrun --standalone --nproc_per_node=1 train.py --config-name smoke distributed.enabled=true training.device=cuda`
- Multi-process logic check without multi-GPU (CPU + Gloo):
	- `torchrun --standalone --nproc_per_node=2 train.py --config-name smoke distributed.enabled=true distributed.backend=gloo training.device=cpu`
