import os
import time
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from spatialid import SpatialModel, KDLoss, DNNModel, MultiCEFocalLoss, DNNDataset

def setup_distributed(world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    rank = int(os.environ.get("SLURM_PROCID", 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def cleanup_distributed():
    dist.destroy_process_group()

class DnnTrainer:
    def __init__(self, input_dims, label_names, device=0, lr=3e-4, weight_decay=1e-6, gamma=2, alpha=.25, reduction="mean", save_path="./dnn.bgi", use_ddp=False, rank=0, world_size=1):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.input_dims = input_dims
        self.n_types = len(label_names)
        self.label_names = label_names
        self.save_path = save_path
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.device = torch.device(f"cuda:{device}" if device > -1 and torch.cuda.is_available() else "cpu")

        self.set_model(input_dims, hidden_dims=1024, output_dims=self.n_types, gamma=gamma, alpha=alpha, reduction=reduction)
        self.set_optimizer(lr=lr, weight_decay=weight_decay)

    def set_model(self, input_dims, hidden_dims, output_dims, gamma, alpha, reduction, **kwargs):
        model = DNNModel(input_dims, hidden_dims, output_dims).to(self.device)
        if self.use_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device.index])
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
                    torch.save({'model': self.model.module if self.use_ddp else self.model, **state}, self.save_path)
        if (not self.use_ddp) or (self.rank == 0):
            print("\n validation model: ")
            checkpoint = torch.load(self.save_path, map_location=self.device)
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

if __name__ == "__main__":
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    use_ddp = world_size > 1
    rank = 0
    local_rank = 0
    if use_ddp:
        rank, local_rank = setup_distributed(world_size)
    # Prepare your args: input_dims, label_names, etc. from your main script or config
    # Example:
    # trainer = DnnTrainer(input_dims, label_names, device=local_rank, use_ddp=use_ddp, rank=rank, world_size=world_size, ...)
    # trainer.train(data, ann_key, marker_genes, batch_size, epochs)
    if use_ddp:
        cleanup_distributed()
