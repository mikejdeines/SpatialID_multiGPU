#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 10:40
# @Author  : zhangchao
# @File    : trainer.py
# @Software: PyCharm

import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from spatialid import SpatialModel, KDLoss, DNNModel, MultiCEFocalLoss, DNNDataset

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Use SLURM_LOCALID if available, otherwise fallback to rank
    local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

class Base:
    def __init__(self, device):
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None
        self.criterion = None
        if device > -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

    def set_model(self, *args, **kwargs):
        pass

    def set_optimizer(self, *args, **kwargs):
        pass

    @staticmethod
    def load_checkpoint(path, map_location=None):
        assert os.path.exists(path)
        checkpoint = torch.load(path, map_location=map_location)
        assert isinstance(checkpoint, dict)
        assert 'model' in checkpoint.keys()
        print(f"The checkpoint was saved: {checkpoint.keys()}")
        return checkpoint

    def save_checkpoint(self, path: str, state: dict = None):
        base_info = {'model': self.model}
        if state is not None:
            base_info.update(state)
        torch.save(base_info, path)

class SpatialTrainer(Base):
    def __init__(self, input_dim, num_classes, lr, weight_decay, device, use_ddp=False, rank=0, world_size=1):
        super(SpatialTrainer, self).__init__(device=device)
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.set_model(input_dim, num_classes)
        self.set_optimizer(lr=lr, weight_decay=weight_decay)

    def set_model(self, input_dim, num_classes):
        gae_dim, dae_dim, feat_dim = [32, 8], [100, 20], 64
        model = SpatialModel(input_dim, num_classes, gae_dim, dae_dim, feat_dim).to(self.device)
        if self.use_ddp:
            # Use SLURM_LOCALID if available, otherwise fallback to self.rank
            local_rank = int(os.environ.get("SLURM_LOCALID", self.rank))
            torch.cuda.set_device(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            self.model = model
        self.criterion = KDLoss(1)

    def set_optimizer(self, lr=0.01, weight_decay=0.0001, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=.95)

    def train(self, data, epochs, w_cls, w_dae, w_gae):
        self.model.train()
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            data = data.to(self.device, non_blocking=True)
            inputs, targets = data.x, data.y
            edge_index = data.edge_index
            edge_weight = data.edge_weight

            outputs, dae_loss, gae_loss = self.model(inputs, edge_index, edge_weight)
            loss = w_cls * self.criterion(outputs, targets) + w_dae * dae_loss + w_gae * gae_loss
            train_loss = loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total = targets.size(0)
            predictions = outputs.argmax(1)
            correct = predictions.eq(targets.argmax(1)).sum().item()
            self.scheduler.step()
            process_time = time.time() - start_time
            accuracy = correct / total * 100.0
            if (not self.use_ddp) or (self.rank == 0):
                print('\r  [Epoch %3d] Loss: %.5f, Time: %.2f s, Psuedo-Acc: %.2f%%'
                      % (epoch, train_loss, process_time, accuracy), flush=True, end="")

    @torch.no_grad()
    def infer(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            inputs = data.x
            edge_index = data.edge_index
            edge_weight = data.edge_weight
            outputs, _, _ = self.model(inputs, edge_index, edge_weight)
            predictions = outputs.argmax(1)
        predictions = predictions.detach().cpu().numpy()
        return predictions

class DnnTrainer(Base):
    def __init__(self, input_dims, label_names, device=0, lr=3e-4, weight_decay=1e-6, gamma=2, alpha=.25, reduction="mean", save_path="./dnn.bgi", use_ddp=False, rank=0, world_size=1):
        super(DnnTrainer, self).__init__(device=device)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.input_dims = input_dims
        self.n_types = len(label_names)
        self.label_names = label_names
        self.save_path = save_path
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.set_model(input_dims, hidden_dims=1024, output_dims=self.n_types, gamma=gamma, alpha=alpha, reduction=reduction)
        self.set_optimizer(lr=lr, weight_decay=weight_decay)

    def set_model(self, input_dims, hidden_dims, output_dims, gamma, alpha, reduction, **kwargs):
        model = DNNModel(input_dims, hidden_dims, output_dims).to(self.device)
        if self.use_ddp:
            # Use SLURM_LOCALID if available, otherwise fallback to self.rank
            local_rank = int(os.environ.get("SLURM_LOCALID", self.rank))
            torch.cuda.set_device(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        else:
            self.model = model
        self.criterion = MultiCEFocalLoss(class_num=self.n_types, gamma=gamma, alpha=alpha, reduction=reduction)

    def set_optimizer(self, lr=3e-4, weight_decay=1e-6, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, data, ann_key, marker_genes=None, batch_size=4096, epochs=200):
        dataset = DNNDataset(adata=data, ann_key=ann_key, marker_genes=marker_genes)
        if self.use_ddp:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, sampler=sampler, num_workers=16)
        else:
            loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=16)

        self.model.train()
        best_loss = np.inf
        for epoch in range(epochs):
            if self.use_ddp:
                loader.sampler.set_epoch(epoch)
            epoch_acc = []
            epoch_loss = []
            for idx, data in enumerate(loader):
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.long().to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, targets)
                train_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total = targets.size(0)
                prediction = output.argmax(1)
                correct = prediction.eq(targets).sum().item()

                accuracy = correct / total * 100.
                epoch_acc.append(accuracy)
                epoch_loss.append(train_loss)
            if (not self.use_ddp) or (self.rank == 0):
                print(f"\r  [{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Epoch: {epoch + 1:3d} "
                      f"Loss: {np.mean(epoch_loss):.5f}, "
                      f"acc: {np.mean(epoch_acc):.2f}%",
                      flush=True,
                      end="")
            if np.mean(epoch_loss) < best_loss:
                best_loss = np.mean(epoch_loss)
                state = {
                    'marker_genes': marker_genes,
                    'batch_size': batch_size,
                    'label_names': self.label_names
                }
                if (not self.use_ddp) or (self.rank == 0):
                    self.save_checkpoint(self.save_path, state)
        if (not self.use_ddp) or (self.rank == 0):
            print("\n validation model: ")
            checkpoint = self.load_checkpoint(self.save_path, map_location=self.device)
            with torch.no_grad():
                best_model = checkpoint["model"]
                best_model.to(self.device)
                best_model.eval()
                val_acc = []
                for idx, data in enumerate(loader):
                    inputs, targets = data
                    inputs = inputs.to(self.device)
                    targets = targets.long().to(self.device)
                    outputs = best_model(inputs)

                    total = targets.size(0)
                    prediction = outputs.argmax(1)
                    correct = prediction.eq(targets).sum().item()
                    accuracy = correct / total * 100.
                    val_acc.append(accuracy)
                print(f"\n  [{time.strftime('%Y-%m-%d %H:%M:%S')} total accuracy: {np.mean(val_acc):.2f}%]")

# Entrypoint example for running training with DDP or single-GPU fallback
def run_spatial_trainer_ddp(rank, world_size, trainer_args, trainer_kwargs, train_args):
    setup(rank, world_size)
    trainer = SpatialTrainer(*trainer_args, use_ddp=True, rank=rank, world_size=world_size, **trainer_kwargs)
    trainer.train(*train_args)
    cleanup()

def run_dnn_trainer_ddp(rank, world_size, trainer_args, trainer_kwargs, train_args):
    setup(rank, world_size)
    trainer = DnnTrainer(*trainer_args, use_ddp=True, rank=rank, world_size=world_size, **trainer_kwargs)
    trainer.train(*train_args)
    cleanup()

def launch_training(trainer_cls, trainer_args, trainer_kwargs, train_args):
    world_size = torch.cuda.device_count()
    if world_size > 1:
        if trainer_cls is SpatialTrainer:
            mp.spawn(run_spatial_trainer_ddp, args=(world_size, trainer_args, trainer_kwargs, train_args),
                     nprocs=world_size, join=True)
        elif trainer_cls is DnnTrainer:
            mp.spawn(run_dnn_trainer_ddp, args=(world_size, trainer_args, trainer_kwargs, train_args),
                     nprocs=world_size, join=True)
    else:
        trainer = trainer_cls(*trainer_args, use_ddp=False, rank=0, world_size=1, **trainer_kwargs)
        trainer.train(*train_args)
