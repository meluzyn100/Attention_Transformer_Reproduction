# Attention_Transformer_Reproduction

This repository contains a from-scratch PyTorch implementation of the original Transformer (Vaswani et al., 2017), focused on rigorous reproduction of the paper's results on WMT 2014.



#ideas to add in future:
- add mlflow logging to the trainer and make it possible to run mlflow ui from the Makefile
- add smoke test config and Makefile target to run it
- add bucketed sampling to the dataloader to speed up training and make it more stable
- add gradient clipping to the trainer
- add learning rate scheduler with warmup to the trainer
- add support for mixed precision training to the trainer
- add support for resuming training from checkpoints to the trainer
- add support for logging training and validation metrics to mlflow in the trainer
- add support for saving model checkpoints to mlflow in the trainer
- add support for early stopping based on validation loss to the trainer
- add support for training on multiple GPUs to the trainer
- add support for training on TPUs to the trainer
- add support for distributed training to the trainer
- add support for training on multiple nodes to the trainer
- add support for training on multiple nodes with multiple GPUs to the trainer
- torch compile the model and trainer for faster training