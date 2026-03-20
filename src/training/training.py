import os
from dataclasses import dataclass, field
from typing import Callable
from itertools import chain
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from ..data.data import LossConfig, UncapInstance, UncapGeneratorConfig, ConstantConfig, NormalConfig, UniformConfig
from ..models.processors import Reasoner
from ..data.prepare_data import GenerateDataset
from ..utils.collate_fn import collate_fn

class Trainer:
    #class that handles all the training of the Reasoner model
    def __init__(self, model, optimizer, loss_config: LossConfig, lr_scheduler=None, device="cuda", checkpoint_dir="checkpoints"):
        #save the variabels
        self.model = model
        self.optimizer = optimizer
        self.loss_config = loss_config
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        #initiate dict for collecting the training history
        self.history = {
            "train": [], #use list for average over different train laoders
            "val": {}, #dict to store val_loss for different val loaders
        }
        
        self.best_val_loss = float("inf")

    #method for saving model checkpoints
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "val_loss": val_loss,
            "history": self.history,
            "loss_config": self.loss_config.weights,
        }, path)

    #method for laoding checkpoints
    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.lr_scheduler and ckpt["lr_scheduler_state_dict"]:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
        return ckpt["epoch"]
    
    #load only the model weights for inference, only want model parameters, not training state
    @staticmethod
    def load_model(model, path: str, device="cuda"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
        return model
    
    #run one epoch
    def _run_epoch(self, loader, weights: dict, train: bool) -> dict:
        self.model.train(train)
        totals = {k: 0.0 for k in weights}
        n_batches = 0

        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(self.device)
                losses = self.model.teacher_forcing(batch)
                total = sum(weights[k] * losses[k] for k in weights)

                if train:
                    self.optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                for k in weights:
                    totals[k] += losses[k].item()
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}
    
    #get weighted loss
    def _weighted_total(self, losses: dict, weights: dict) -> float:
        return sum(weights.get(k, 0) * losses.get(k, 0) for k in weights)
    
    #main training loop
    def fit(self, train_loaders: list, val_loaders: dict, n_epochs: int, verbose: bool = True):

        #init val history keys
        for key in val_loaders:
            if key not in self.history["val"]:
                self.history["val"][key] = []

        pbar = tqdm(range(1, n_epochs + 1), desc="Training", disable=not verbose)
        for epoch in pbar:
            weights = self.loss_config.get_weights(epoch)
            self.model.tf_prob = max((self.loss_config.tf_end - epoch) / max(self.loss_config.tf_end, 1.0), 0.0)

            #iterate over all training sizes
            all_train_losses = []
            for loader in train_loaders:
                losses = self._run_epoch(loader, weights, train=True)
                all_train_losses.append(losses)

            #average the training loss
            train_avg = {k: sum(l[k] for l in all_train_losses) / len(all_train_losses) for k in weights}
            train_avg["_weights"] = weights.copy()
            self.history["train"].append(train_avg)

            #iterate over all validation sizes
            val_parts = []
            for key, loader in val_loaders.items():
                vl = self._run_epoch(loader, weights, train=False)
                self.history["val"][key].append(vl)
                val_parts.append(vl)
            
            #average the val loss
            val_avg = {k: sum(v[k] for v in val_parts) / len(val_parts) for k in weights}

            #checkpoint if valid
            val_total = self._weighted_total(val_avg, weights)
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best_model.pt"), epoch, val_total
                )

            #do a lr scheduler step
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_total)
                else:
                    self.lr_scheduler.step()

            #print progress
            if verbose:
                train_total = self._weighted_total(train_avg, weights)
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix_str(
                    f"lr={lr:.1e} "
                    f"t_loss={train_total:.3f} v_loss={val_total:.3f} "
                    f"t_opt={train_avg['optimum_diff']:.3f} "
                    f"v_opt={val_avg['optimum_diff']:.3f} "
                    f"t_dual={train_avg['dual_diff']:.3f} "
                    f"v_dual={val_avg['dual_diff']:.3f}"
                )

        #save final checkpoint
        self.save_checkpoint(
            os.path.join(self.checkpoint_dir, "last_model.pt"), epoch, val_total
        )

        print(f"\nBest val loss: {self.best_val_loss:.4f}")
        return self.history
    
def generate_training_validation_datasets(
        train_configs: list,
        val_configs: list,
        batch_size: int,
):
    train_datasets = []
    val_datasets = []

    train_path = Path(__file__).resolve().parent.parent / "datasets" / "training"

    #create the generator configs
    for cfg in train_configs:
        gen_cfg = UncapGeneratorConfig(
            n_cli=cfg["n_cli"],
            n_fac=cfg["n_fac"],
            seed=33,
            demand_config=ConstantConfig(1), #training only on the constand demand, otherwise traces are wrong
            coords_config=NormalConfig(mean=0.5, std=0.5),
        )

        dataset = GenerateDataset(
            generator_config=gen_cfg,
            n_samples=cfg["n_samples"],
            cache_dir= train_path / f"train_{cfg["n_cli"]}_{cfg["n_fac"]}"
        )
        dataset.prepare_data()
        #split, not actually necessary if seperate valdiation datasets are created but keep for now
        n_train = int(0.75 * len(dataset))
        n_val = len(dataset) - n_train
        t, v = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_datasets.append(t)
        val_datasets.append(v)
        print(f"  {cfg['n_fac']}x{cfg['n_cli']}: {n_train} train / {n_val} val")

    #generate the train loaders (one per size)
    train_loaders = [
        DataLoader(dataset, batch_size=min(batch_size, cfg["n_samples"]), shuffle=True, collate_fn=collate_fn) for dataset in train_datasets
    ]

    #create additional purely valdiation datasets/loaders
    val_extra_loaders = {}
    for cfg in val_configs:
        gen_cfg = UncapGeneratorConfig(
            n_cli=cfg["n_fac"], n_cli=["n_cli"], seed=7777,
            demand_config=ConstantConfig(1),
            facility_cost_config=UniformConfig(0, 1),
            coords_config=NormalConfig(mean=0.5, std=0.5),
        )
        dataset = GenerateDataset(gen_cfg, 
                                  n_samples=cfg["n_samples"],
                                  cache_dir=train_path / f"val_{cfg["n_cli"]}_{cfg["n_fac"]}"
        )
        dataset.prepare_data()
        loader = DataLoader(dataset, batch_size=min(batch_size, cfg["n_samples"]), shuffle=False, collate_fn=collate_fn)
        key = f"{cfg["n_cli"]}x{cfg["n_fac"]}"
        val_extra_loaders[key] = loader
        print(f"val scale {key}: {len(dataset)} samples")

    #combine train-split validation with the extra val loaders
    val_loaders_by_scale = {}
    for i, cfg in enumerate(train_configs):
        key = f"{cfg["n_cli"]}x{cfg["n_fac"]}"
        val_loaders_by_scale[key] = DataLoader(
            val_datasets[i], batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
    val_loaders_by_scale.update(val_extra_loaders)
