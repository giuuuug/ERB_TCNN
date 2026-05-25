# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

"""Base trainer classes with all the usual Pytorch training boilerplate."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
import copy

class BaseTrainer:
    """
    Base trainer class with all the checkpointing, logging etc., but none of the actual
    training or evaluation logic.
    Does not support distributed training.

    Notes
    -----
    Trainers for actual models should subclass this class and implement the
    _run_train_batch, _run_validation_batch, _run_train_epoch and _run_validation_epoch methods.
    The _run_train_batch method should take a batch as input and retun training loss
    The _run_validation_batch method should take a batch as input and return validation metrics
    The _run_train_epoch and _run_validation_epoch methods should take an epoch number as input
    and return training loss and validation metrics over the epoch respectively.
    For examples, look at trainers in other python modules in the same folder.
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_data: DataLoader,
                 valid_data: DataLoader,
                 device: str = "cuda:0",
                 save_every: int = 20,
                 ckpt_path: str = "checkpoints/",
                 logs_path: str = "training_logs.csv",
                 snapshot_path: str = "snapshot.pth",
                 device_memory_fraction: float = 0.5,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        '''
        Parameters
        ----------
        model, torch.nn.Module : Torch model to train
        optimizer, torch.optim.Optimizer : Torch optimizer object to use during training.
        train_data, Dataloader : Dataloader for the training data
        valid_data, Dataloader : Dataloader for the validation data.
        device, str or torch.device : torch device to use for training
        save_every, int : A training snapshot and a model checkpoint 
            will be saved every save_every epochs.
        ckpt_path, str or Path : Path to the folder where model checkpoints should be saved
        logs_path, str or Path : Filepath to where the logs .csv file should be saved.
        snapshot_path, str or Path : Filepath to where the training snapshot file should be saved.
        device_memory_fraction, float : Maximum fraction of device's memory to use
        '''
        if device != "cpu":
            torch.cuda.set_per_process_memory_fraction(device_memory_fraction, device)

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_data
        self.valid_data = valid_data
        self.device = device
        self.save_every = save_every
        self.ckpt_path = Path(ckpt_path)
        self.logs_path = Path(logs_path)
        self.snapshot_path = Path(snapshot_path)
        # Set by subclasses at the end of each validation epoch; used by ReduceLROnPlateau
        self.last_reference_metric_value = None

        # Make sure to update the best model in the inheriting class
        self.best_model = copy.deepcopy(self.model)

        # Check that checkpointing folder exists, if not create it
        # Skips creation instead of overwriting if folder already exists
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        # Starting epoch, gets updated if loading snapshot
        self.starting_epoch = 0
        # Array that stores training and validation metrics.
        # One row of the array = metrics for one epoch.
        # Note that the header of the metrics dataframe should be defined by the inheriting class
        # at initialization.
        self.metrics_array = None
        # Define self.header in the inheriting class. 
        # It is a list of the metric names in the order they are returned by _run_train_epoch and _run_validation_epoch.
        self.header = None
        # Load snapshot on instanciation if it exists
        if self.snapshot_path.exists():
            print(f"Snapshot found. Loading snapshot at {self.snapshot_path}")
            self.load_snapshot()

    def save_checkpoint(self, epoch, ckpt_name="checkpoint.pth"):
        '''
        Saves a model checkpoint (i.e. the state dict of the Module in self.model)

        Inputs
        ------
        epoch, int : Epoch number, only used for printing messages to user
        ckpt_name, str : Checkpoint filename

        Outputs
        -------
        None
        '''
        torch.save(self.model.state_dict(), self.ckpt_path / ckpt_name)
        print(f"Model checkpoint at epoch {epoch} saved at {self.ckpt_path / ckpt_name}")
    
    def save_snapshot(self, epoch):
        '''
        Saves a training snapshot containing the model state dict, the current epoch and 
        training/validation metrics. These snapshots are used for warm restarts.

        Inputs
        ------
        epoch : Current epoch number

        Outputs
        -------
        None
        '''
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPT_STATE": self.optimizer.state_dict(),
            "EPOCH": epoch,
            "METRICS_ARRAY": self.metrics_array,
            "BEST_MODEL_STATE": self.best_model.state_dict()
        }
        if self.scheduler is not None:
            snapshot["SCHEDULER_STATE"] = self.scheduler.state_dict()
        torch.save(snapshot, self.snapshot_path)

    def load_snapshot(self):
        '''
        Loads the training snapshot located at self.snapshot_path

        Inputs
        ------
        None

        Outputs
        -------
        None
        '''
        snapshot = torch.load(self.snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPT_STATE"])
        self.starting_epoch = snapshot["EPOCH"]
        self.metrics_array = snapshot["METRICS_ARRAY"]
        self.best_model.load_state_dict(snapshot["BEST_MODEL_STATE"])
        if self.scheduler is not None and "SCHEDULER_STATE" in snapshot:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        print("Snapshot loaded successfully.")
        
    def _step_scheduler(self):
        if self.scheduler is None:
            return
        old_lr = self.optimizer.param_groups[0]['lr']
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.last_reference_metric_value is not None:
                self.scheduler.step(self.last_reference_metric_value)
        else:
            self.scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[INFO] LR changed: {old_lr:.2e} -> {new_lr:.2e}")

    def _run_train_batch(self, batch):
        raise NotImplementedError(self.missing_method_str("_run_train_batch"))
    def _run_validation_batch(self, batch):
        raise NotImplementedError(self.missing_method_str("_run_validation_batch"))
    def _run_train_epoch(self, epoch):
        raise NotImplementedError(self.missing_method_str("_run_train_epoch"))
    def _run_validation_epoch(self, epoch):
        raise NotImplementedError(self.missing_method_str("_run_validation_epoch"))
    
    def _log_metrics(self):
        '''
        Saves computed metrics in a .csv file at self.logs_path

        Inputs
        ------
        None

        Outputs
        -------
        None
        '''
        # I might add more to this later, for now simply save the metrics in a .csv
        metrics_df = pd.DataFrame(self.metrics_array, columns=self.header)
        metrics_df.to_csv(self.logs_path, header=True, index=False)

    def train(self, n_epochs):
        '''
        Main training loop.

        Inputs
        ------
        n_epochs, int : Number of epochs to train for

        Outputs
        -------
        None
        '''
        # If metrics array doesn't exist, initialize it.
        # If there is a snapshot, it should have been loaded in __init__
        # self.header should be defined in the __init__ of the inheriting class
        if self.metrics_array is None:
            self.metrics_array = np.zeros((n_epochs, len(self.header)))
        elif n_epochs > len(self.metrics_array):
            # If starting from snapshot and doing more epochs than in initial run, 
            # add rows to self.metrics array
            self.metrics_array = np.append(self.metrics_array,
                                np.zeros((n_epochs - len(self.metrics_array), len(self.header))), axis=0)
        
        for epoch in range(self.starting_epoch, n_epochs):
            self._run_train_epoch(epoch)
            stop = self._run_validation_epoch(epoch)
            self._step_scheduler()
            if (epoch + 1) % self.save_every == 0:
                self.save_snapshot(epoch=epoch + 1)
                self.save_checkpoint(epoch + 1, ckpt_name=f"checkpoint_epoch_{epoch + 1}")
            if stop:
                print(f"[INFO] Early stopping triggered at epoch {epoch + 1}. Stopping training.")
                break
        self._log_metrics()
        return self.model, self.best_model

    @staticmethod
    def missing_method_str(m_name):
        return (f"No {m_name} method implemented. BaseTrainer should not be instantiated \n"
            "You should instead define your own Trainer class that inherits from BaseTrainer and \n"
            "implements the _run_train_batch, _run_validation_batch, _run_train_epoch and \n" 
            "_run_validation_epoch methods.")