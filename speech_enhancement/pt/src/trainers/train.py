# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import torch
from pathlib import Path
import matplotlib.pyplot as plt
from .spec import MagSpecTrainer
from .myspec import MyMagSpecTrainer
from speech_enhancement.pt.src.trainers.myspec_stream import MyMagSpecTrainer_Stream
from speech_enhancement.pt.src.utils import plot_training_metrics
from common.utils import log_to_file

import pandas as pd
from onnxruntime import InferenceSession


class SETrainerWrapper:
    '''Wrapper class for the pre-exisiting speech enhancement trainers.
       Handles trained model ONNX export & figure plotting
    '''
    def __init__(self, cfg, model, dataloaders):
        '''Initialize the trainer wrapper and underlying trainer.

        Parameters
        ----------
        cfg, object : User configuration including `training` and `preprocessing` sections.
        model, torch.nn.Module : Torch model to train.
        dataloaders, dict[str, torch.utils.data.DataLoader] : Contains `train_dl` and `valid_dl`.

        Notes
        -----
        Initializes optimizer, log and checkpoint paths, optional snapshot loading, and
        constructs the `MagSpecTrainer` with regularization and preprocessing arguments.
        '''
        self.cfg = cfg
        self.model = model
        self.train_dl = dataloaders["train_dl"]
        self.valid_dl = dataloaders["valid_dl"]


        # Initialize optimizer
        self.optimizer = getattr(torch.optim, cfg.training.optimizer)(
            params=model.parameters(), **cfg.training.optimizer_arguments)

        # Initialize LR scheduler (optional)
        self.scheduler = None
        lr_scheduler_cfg = getattr(cfg.training, "lr_scheduler", None)
        if lr_scheduler_cfg is not None:
            scheduler_name = getattr(lr_scheduler_cfg, "scheduler", None)
            if scheduler_name:
                scheduler_args = getattr(lr_scheduler_cfg, "scheduler_arguments", {}) or {}
                self.scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
                    self.optimizer, **scheduler_args)
                print(f"\n[INFO] LR scheduler: {scheduler_name} with args {scheduler_args}\n")
        
        # Gather preprocessing args that need to be passed to trainer
        preproc_args = {"sampling_rate":cfg.preprocessing.sample_rate,
                        "frame_length":cfg.preprocessing.win_length,
                        "hop_length":cfg.preprocessing.hop_length,
                        "n_fft":cfg.preprocessing.n_fft,
                        "center":cfg.preprocessing.center}

        ckpt_path = Path(cfg.output_dir, cfg.training.ckpt_path)
        self.logs_path = Path(cfg.output_dir, 'training_logs', cfg.training.logs_filename)
        
        # If user provides a training snapshot, use it.
        if cfg.training.snapshot_path is None:
            snapshot_path = Path(cfg.output_dir, 'training_logs', 'training_snapshot.pth')
        else:
            snapshot_path = cfg.training.snapshot_path
        
        # If snapshot file exists, it will be loaded automatically by the trainer
        # Log this to file
        if snapshot_path.exists():
            log_to_file(cfg.output_dir, f"Loaded training snapshot at {snapshot_path}")

        # Make dirs if necessary
        self.logs_path.parent.mkdir(parents=False, exist_ok=True)
        ckpt_path.mkdir(parents=False, exist_ok=True)

        if cfg.training.regularization is None:
            cfg.training.regularization = {}
        regularization_args = cfg.training.regularization

        if cfg.training.trainer_model  == "MyMagSpecTrainer":
            if cfg.operation_mode is not None and "stream" in cfg.operation_mode:
                print("\n[ASSERTION] Using MyMagSpecTrainer_Stream \n")
                self.trainer = MyMagSpecTrainer_Stream(model=model,
                                    optimizer=self.optimizer,
                                    train_data=self.train_dl,
                                    valid_data=self.valid_dl,
                                    loss=cfg.training.loss,
                                    batching_strat=cfg.training.batching_strategy,
                                    device=cfg.training.device,
                                    device_memory_fraction=cfg.general.gpu_memory_limit,
                                    save_every=cfg.training.save_every,
                                    ckpt_path=ckpt_path,
                                    logs_path=self.logs_path,
                                    snapshot_path=snapshot_path,
                                    early_stopping=cfg.training.early_stopping,
                                    early_stopping_patience=cfg.training.early_stopping_patience,
                                    reference_metric=cfg.training.reference_metric,
                                    loud_loss_weight=cfg.training.loud_loss_weight,
                                    si_snr_loss_weight=cfg.training.si_snr_loss_weight,
                                    scheduler=self.scheduler,
                                    **preproc_args,
                                    **regularization_args)
            else:
                print("\n[ASSERTION] Using MyMagSpecTrainer \n")
                self.trainer = MyMagSpecTrainer(model=model,
                                    optimizer=self.optimizer,
                                    train_data=self.train_dl,
                                    valid_data=self.valid_dl,
                                    loss=cfg.training.loss,
                                    batching_strat=cfg.training.batching_strategy,
                                    device=cfg.training.device,
                                    device_memory_fraction=cfg.general.gpu_memory_limit,
                                    save_every=cfg.training.save_every,
                                    ckpt_path=ckpt_path,
                                    logs_path=self.logs_path,
                                    snapshot_path=snapshot_path,
                                    early_stopping=cfg.training.early_stopping,
                                    early_stopping_patience=cfg.training.early_stopping_patience,
                                    reference_metric=cfg.training.reference_metric,
                                    loud_loss_weight=cfg.training.loud_loss_weight,
                                    si_snr_loss_weight=cfg.training.si_snr_loss_weight,
                                    scheduler=self.scheduler,
                                    **preproc_args,
                                    **regularization_args)
        else:
            print("\n[ASSERTION] Using MagSpecTrainer \n")
            self.trainer = MagSpecTrainer(model=model,
                                optimizer=self.optimizer,
                                train_data=self.train_dl,
                                valid_data=self.valid_dl,
                                loss=cfg.training.loss,
                                batching_strat=cfg.training.batching_strategy,
                                device=cfg.training.device,
                                device_memory_fraction=cfg.general.gpu_memory_limit,
                                save_every=cfg.training.save_every,
                                ckpt_path=ckpt_path,
                                logs_path=self.logs_path,
                                snapshot_path=snapshot_path,
                                early_stopping=cfg.training.early_stopping,
                                early_stopping_patience=cfg.training.early_stopping_patience,
                                reference_metric=cfg.training.reference_metric,
                                scheduler=self.scheduler,
                                **preproc_args,
                                **regularization_args)
        
    def _plot_figures(self):
        '''Plot and save the training metrics figure.

        Notes
        -----
        - Reads the metrics CSV at `self.logs_path` and creates a figure via `plot_training_metrics`.
        - Displays the figure when `cfg.general.display_figures` is True.
        - Saves the figure to `training_metrics.png` in the logs directory.
        '''
        metrics_df = pd.read_csv(self.logs_path)
        fig = plot_training_metrics(metrics_df=metrics_df, figsize=(12, 15))

        if self.cfg.general.display_figures:
            plt.show()
        plt.savefig(Path(self.logs_path.parent, "training_metrics.png"))
        plt.close(fig)

    def train(self):
        '''Run training and export trained and best models to ONNX.

        Returns
        -------
        tuple : (`onnx_model_session`, `best_onnx_model_session`)
            - `onnx_model_session`, onnxruntime.InferenceSession : Session for the trained ONNX model.
            - `best_onnx_model_session`, onnxruntime.InferenceSession : Session for the best ONNX model.

        Notes
        -----
        - Exports ONNX models with dynamic sequence length axis `seq_len`.
        - Assumes a single input/output and input shape `(batch, n_fft // 2 + 1, sequence_length)`.
        '''

        model, best_model = self.trainer.train(n_epochs=self.cfg.training.epochs)

        # Export to ONNX
        # Here we assume the model's input shape is (batch, n_fft // 2 + 1, sequence_length)
        # Might change this later and expose the input shape in cfg
        # We also assume it only has one input & output
        # NOTE : Change this when adding support for decomposed LSTM
        
        model.eval()
        model.to("cpu")
        is_stream = self.cfg.operation_mode is not None and "stream" in self.cfg.operation_mode
        
        if is_stream:
            # For streaming models we use T=1 and pass explicit states, and no dynamic axes for STM32 deployment
            dummy_tensor = torch.ones((1, self.cfg.preprocessing.n_fft // 2 + 1, 1))
            states = model.get_initial_states(batch_size=1, device="cpu")
            num_states = sum(1 for block in states for s in block)
            input_names = ["input"] + [f"state_in_{i}" for i in range(num_states)]
            output_names = ["output"] + [f"state_out_{i}" for i in range(num_states)]
            dynamic_axes = None
            inputs_tuple = (dummy_tensor, states)
        else:
            dummy_tensor = torch.ones((1, self.cfg.preprocessing.n_fft // 2 + 1, 10))
            input_names = ["input"]
            output_names = ["output"]
            dynamic_axes = {"input":{2:"seq_len"}, "output":{2:"seq_len"}}
            inputs_tuple = dummy_tensor

        onnx_model_path = Path(self.cfg.output_dir, self.cfg.general.saved_models_dir, 'trained_model.onnx')
        onnx_model_path.parent.mkdir(exist_ok=True)
        
        if dynamic_axes is not None:
            torch.onnx.export(model,
                            inputs_tuple,
                            onnx_model_path,
                            export_params=True,
                            opset_version=self.cfg.training.opset_version,
                            do_constant_folding=True,
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes
                            )
        else:
            torch.onnx.export(model,
                            inputs_tuple,
                            onnx_model_path,
                            export_params=True,
                            opset_version=self.cfg.training.opset_version,
                            do_constant_folding=True,
                            input_names=input_names,
                            output_names=output_names
                            )
        
        # Same with best model 
        
        best_model.eval()
        best_model.to("cpu")
        if is_stream:
            states = best_model.get_initial_states(batch_size=1, device="cpu")
            inputs_tuple = (dummy_tensor, states)

        best_onnx_model_path = Path(self.cfg.output_dir, self.cfg.general.saved_models_dir, 'best_trained_model.onnx')
        
        if dynamic_axes is not None:
            torch.onnx.export(best_model,
                            inputs_tuple,
                            best_onnx_model_path,
                            export_params=True,
                            opset_version=self.cfg.training.opset_version,
                            do_constant_folding=True,
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes
                            )
        else:
             torch.onnx.export(best_model,
                            inputs_tuple,
                            best_onnx_model_path,
                            export_params=True,
                            opset_version=self.cfg.training.opset_version,
                            do_constant_folding=True,
                            input_names=input_names,
                            output_names=output_names
                            )
        
        print("\n [INFO] Training complete\n"
              f"Trained model saved at {onnx_model_path}")
        
        # Plot figures
        self._plot_figures()

        onnx_model_session = InferenceSession(onnx_model_path)
        best_onnx_model_session = InferenceSession(best_onnx_model_path)
        return onnx_model_session, best_onnx_model_session

