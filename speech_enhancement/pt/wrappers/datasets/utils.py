# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
import copy
from speech_enhancement.pt.src.dataset_utils.utils import load_dataset_from_cfg
from speech_enhancement.pt.src.preprocessing import IdentityPipeline
import speech_enhancement.pt.src.preprocessing
from torch.utils.data import DataLoader

'''Helper functions to instantiate preproc pipelines'''


def get_dataloaders(cfg: DictConfig):
    '''
    Build dataloaders for training, validation, evaluation, and quantization.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration object containing `operation_mode`, `training`, `evaluation`, `quantization`,
        `dataset`, and `preprocessing` sections.

    Returns
    -------
    dict
        Dictionary with optional keys:
        - ``train_dl``: torch.utils.data.DataLoader for training data or ``None``.
        - ``valid_dl``: torch.utils.data.DataLoader for validation data (batch size 1) or ``None``.
        - ``eval_dl``: torch.utils.data.DataLoader for evaluation data (batch size 1) or ``None``.
        - ``quant_dl``: torch.utils.data.DataLoader for quantization calibration data (batch size 1) or ``None``.
    '''
    # We use the same code for Valentini or Custom DS, so put it here instead of each individual wrapper
    
    # Initialize preproc pipelines
    pipeline_args = copy.copy(cfg.preprocessing)
    pipeline_type = pipeline_args["pipeline_type"]
    del pipeline_args["pipeline_type"]
    
    # If we are asked to load the training set
    if cfg.training and cfg.operation_mode in ["training", "chain_tqe", "chain_tqeb", "chain_tqe_stream"]:

        input_pipeline = getattr(speech_enhancement.pt.src.preprocessing, pipeline_type)(
        magnitude=False, **pipeline_args)

        loss = cfg.training.loss
        if loss in ["spec_mse", "loud_loss", "loud_loss__si_snr"]:
            train_target_pipeline = input_pipeline
        elif loss in ['wave_mse', 'wave_sisnr', 'wave_snr']:
            train_target_pipeline = IdentityPipeline(peak_normalize=pipeline_args["peak_normalize"])
        else:
            raise ValueError("Invalid loss type. Should be one of 'spec_mse', 'loud_loss', 'loud_loss__si_snr',"
                            f" 'wave_mse', 'wave_sisnr', 'wave_snr', but was {loss}")

        valid_target_pipeline = IdentityPipeline(peak_normalize=pipeline_args["peak_normalize"])

        # Load training dataset
        print("[INFO] Loading training set")
        train_ds = load_dataset_from_cfg(cfg,
                                        set="train",
                                        n_clips=cfg.dataset.num_training_samples,
                                        val_split=cfg.dataset.num_validation_samples,
                                        input_pipeline=input_pipeline,
                                        target_pipeline=train_target_pipeline)
        
        train_dl = DataLoader(train_ds,
                            batch_size=cfg.training.batch_size,
                            num_workers=cfg.training.num_dataloader_workers,
                            shuffle=cfg.training.shuffle)

        # Load validation dataset
        print("[INFO] Loading validation set")
        valid_ds = load_dataset_from_cfg(cfg,
                                        set="valid",
                                        n_clips=cfg.dataset.num_validation_samples,
                                        val_split=cfg.dataset.num_validation_samples,
                                        input_pipeline=input_pipeline,
                                        target_pipeline=valid_target_pipeline)
        
        # Here, batch size is forced to 1 to avoid padding/trimming during validation
        valid_dl = DataLoader(valid_ds, batch_size=1)

    else:
        train_dl = None
        valid_dl = None
    
    # If we need to load eval set
    if cfg.evaluation and cfg.operation_mode in ["evaluation", "chain_eqe", "chain_tqe", "chain_tqeb", "chain_eqeb", "chain_tqe_stream"]:
        
        input_pipeline = getattr(speech_enhancement.pt.src.preprocessing, pipeline_type)(
        magnitude=False, **pipeline_args)

        target_pipeline = IdentityPipeline(peak_normalize=pipeline_args["peak_normalize"])


        # Load evaluation dataset
        print("[INFO] Loading eval set")
        eval_ds = load_dataset_from_cfg(cfg,
                                        set="test",
                                        n_clips=cfg.dataset.num_test_samples,
                                        input_pipeline=input_pipeline,
                                        target_pipeline=target_pipeline)
        
        eval_dl = DataLoader(eval_ds,
                            batch_size=1)
    else:
        eval_dl = None
    
    # If we need to load quant set
    if cfg.quantization and cfg.operation_mode in ["quantization", "chain_qd", "chain_qb", "chain_tqe", "chain_tqeb", "chain_eqe", "chain_eqeb", "chain_tqe_stream"]:
        input_pipeline = getattr(speech_enhancement.pt.src.preprocessing, pipeline_type)(
        magnitude=True, **pipeline_args)

        # Load quantisation dataset
        print("[INFO] Loading calibration set")
        quant_ds = load_dataset_from_cfg(cfg,
                                        set="train",
                                        n_clips=cfg.quantization.num_quantization_samples,
                                        val_split=0,
                                        input_pipeline=input_pipeline,
                                        target_pipeline=None,
                                        quantization=True)
        
        quant_dl = DataLoader(quant_ds,
                            batch_size=1)
    else:
        quant_dl = None

    dl_dict = {"train_dl":train_dl,
                "valid_dl":valid_dl,
                "eval_dl":eval_dl,
                "quant_dl":quant_dl}
    return dl_dict
