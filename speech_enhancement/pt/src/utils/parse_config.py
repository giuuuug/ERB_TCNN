# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from munch import DefaultMunch
from typing import List

from common.utils.cfg_utils import postprocess_config_dict, check_config_attributes, parse_tools_section, \
                      parse_benchmarking_section, parse_mlflow_section, parse_top_level, check_hardware_type

def mlflow_ini(cfg: DictConfig = None) -> None:
    """
    Initializes MLflow tracking with the given configuration.

    Args:
        cfg (dict): A dictionary containing the configuration parameters for MLflow tracking.

    Returns:
        None
    """
    mlflow.set_tracking_uri(cfg['mlflow']['uri'])
    experiment_name = cfg.general.project_name
    mlflow.set_experiment(experiment_name)
    run_name = HydraConfig.get().runtime.output_dir.split(os.sep)[-1]
    mlflow.set_tag("mlflow.runName", run_name)
    params = {"operation_mode": cfg.operation_mode}
    mlflow.log_params(params)
    mlflow.pytorch.autolog(log_models=False) # This is not going to log much

def _parse_general_section(cfg: DictConfig,
                          legal: List = None,
                          required: List = None) -> None:
    '''
    parses the general section of configuration file.
    args:
        cfg (DictConfig): configuration dictionary
        mode (str): operation mode
        mode_groups (str): operation mode group
        legal (List): UC specific usable attributes
        required (List): UC specific required attributes
        output_dir (str): output directory for the current run
    '''
    # Usage of the model_path attribute in training modes
    # is checked when parsing the 'training' section.
    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="general")

    # Set default values of missing optional attributes
    if not cfg.project_name:
        cfg.project_name = "<unnamed>"
    if not cfg.logs_dir:
        cfg.logs_dir = "logs"
    if not cfg.saved_models_dir:
        cfg.saved_models_dir = "saved_models"
    cfg.display_figures = cfg.display_figures if cfg.display_figures is not None else False

    # Check that GPU memory fraction is a float < 1 (==1 will throw a torch exception)

    assert (isinstance(cfg.gpu_memory_limit, float) and cfg.gpu_memory_limit) < 1, \
          "gpu_memory_limit must be a float < 1"
    

def _parse_dataset_section(cfg: DictConfig, mode: str = None, mode_groups: DictConfig = None) -> None:
    # cfg: dictionary containing the 'dataset' section of the configuration file

    legal = ["name", "dataset_name", "root_folder", "n_speakers", "file_extension", "num_training_samples",
             "num_validation_samples", "num_test_samples", "random_seed", "clean_train_files_path",
             "clean_valid_files_path", "clean_test_files_path", "noisy_train_files_path",
             "noisy_valid_files_path", "noisy_test_files_path", "shuffle"]
    one_or_more = ["name", "dataset_name"]
    required = ["file_extension", "random_seed"]
    if cfg.dataset_name == "valentini":
        required += ["root_folder", "n_speakers"]
    if str(cfg.dataset_name).lower() == "dns":
        required += ["clean_train_files_path", "noisy_train_files_path",
                     "clean_valid_files_path", "noisy_valid_files_path",
                     "clean_test_files_path", "noisy_test_files_path"]
    if mode in mode_groups.training and cfg.dataset_name == "custom":
        required += ["clean_train_files_path", "noisy_train_files_path"]
    elif mode in mode_groups.evaluation and cfg.dataset_name == "custom":
        required += ["noisy_test_files_path", "clean_test_files_path"]

    check_config_attributes(cfg, specs={"legal": legal, "all": required, "one_or_more": one_or_more},
                            section="dataset")

    # Check that the dataset directories exist
    dataset_audio_paths = [(cfg.root_folder, "root folder"),
                           (cfg.clean_train_files_path, "Clean training set files"),
                           (cfg.clean_valid_files_path, "Clean validation set files"),
                           (cfg.clean_test_files_path, "Clean test set files"),
                           (cfg.noisy_train_files_path, "Noisy training set files"),
                           (cfg.noisy_valid_files_path, "Noisy validation set files"),
                           (cfg.noisy_test_files_path, "Noisy test set files")]
    
    for path, name in dataset_audio_paths:
        if path and not Path(path).is_dir():
            raise FileNotFoundError(f"\nUnable to find the directory of {name}\n"
                                    f"Received path: {path}\n"
                                    "Please check the 'dataset' section of your configuration file.")

def _parse_model_section(cfg, mode, mode_groups):
    legal = ["model_type", "state_dict_path", "model_path", "model_name", "framework"]
    if mode in mode_groups.training:
        one_or_more = ["model_name", "model_type"]
        required = []
    elif mode in mode_groups.evaluation:
        one_or_more = ["model_path", "model_name", "model_type"]
        required = []
    else:
        required = ["model_path"]
        one_or_more = []

    check_config_attributes(cfg, specs={"legal": legal, "all": required, "one_or_more": one_or_more},
                            section="model")
    
    # Keep compatibility with old config files by allowing model.model_type to be used instead of
    # model.model_name
    if cfg.model_name is None:
        cfg.model_name = cfg.model_type



def _parse_training_section(cfg):
    legal = ["device", "epochs", "optimizer", "optimizer_arguments", "trainer_name",
           "loss", "batching_strategy", "num_dataloader_workers", "batch_size",
           "regularization", "save_every", "snapshot_path", "ckpt_path", "logs_filename",
           "opset_version", "reference_metric", "early_stopping", "early_stopping_patience", "dryrun", "trainer_model", "loud_loss_weight", "si_snr_loss_weight",
           "lr_scheduler"]
    required = ["device", "epochs", "optimizer", "optimizer_arguments",
               "loss", "batching_strategy", "batch_size", "trainer_name",
               "save_every", "opset_version", "reference_metric", "early_stopping", "trainer_model"]
    
    # Check that optimizer_arguments at least has the lr key present
    if "lr" not in cfg.optimizer_arguments:
        raise ValueError("training.optimizer_arguments dict must contain at least the 'lr' key")
    # Early stopping patience defaut value
    if cfg.early_stopping and not cfg.early_stopping_patience:
        cfg.early_stopping_patience = 20
    # If regularization params are given, check them
    if cfg.regularization:
        reg_legal = ["weight_clipping_max", "activation_regularization", "act_reg_layer_types",
                     "act_reg_threshold", "penalty_type"]
        check_config_attributes(cfg.regularization, specs={"legal": reg_legal}, section="training.regularization")

    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="training")

def _parse_quantization_section(cfg):
    legal = ["num_quantization_samples", "random_seed", "noisy_quantization_files_path",
             "static_sequence_length", "static_axis_name", "granularity", "onnx_quant_parameters", 
             "reduce_range", "onnx_extra_options", "quantizer"]
    required = ["static_sequence_length", "static_axis_name", "granularity", "onnx_quant_parameters",
                "reduce_range", "quantizer"]

    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="quantization")

def _parse_evaluation_section(cfg):
    legal = ["logs_path", "device", "fixed_sequence_length"]
    required = ["logs_path", "device"]

    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="evaluation")

def _parse_deployment_section(cfg):
    legal = ["frames_per_patch", "lookahead_frames", "output_silence_threshold",
             "c_project_path", "IDE", "verbosity", "hardware_setup", "build_conf"]
    legal_hw = ["serie", "board", "stlink_serial_number"]
    required_hw = ["serie", "board"]
    check_config_attributes(cfg, specs={"legal":legal, "all":legal}, section="deployment")
    check_config_attributes(cfg.hardware_setup, specs={"legal":legal_hw, "all":required_hw}, section="deployment")


def get_config(config_data: DictConfig) -> DefaultMunch:
    """
    Converts the configuration data, performs some checks and reformats
    some sections so that they are easier to use later on.

    Args:
        config_data (DictConfig): dictionary containing the entire configuration file.

    Returns:
        DefaultMunch: The configuration object.
    """

    config_dict = OmegaConf.to_container(config_data)
             
    # Restore booleans, numerical expressions and tuples
    # Expand environment variables
    postprocess_config_dict(config_dict)

    # Top level section parsing
    cfg = DefaultMunch.fromDict(config_dict)
    mode_groups = DefaultMunch.fromDict({
        "training": ["training", "chain_tqeb", "chain_tqe", "chain_tqe_stream"],
        "evaluation": ["evaluation", "chain_tqeb", "chain_tqe", "chain_eqe", "chain_eqeb", "chain_tqe_stream"],
        "quantization": ["quantization", "chain_tqeb", "chain_tqe", "chain_eqe",
                         "chain_qb", "chain_eqeb", "chain_qd", "chain_tqe_stream"],
        "benchmarking": ["benchmarking", "chain_tqeb", "chain_qb", "chain_eqeb"],
        "deployment": ["deployment", "chain_qd"],
        "compression": []
    })
    mode_choices = ["training", "evaluation", "prediction", "deployment", 
                    "quantization", "benchmarking", "chain_tqeb", "chain_tqe",
                    "chain_eqe", "chain_qb", "chain_eqeb", "chain_qd", "chain_tqe_stream"]
    
    legal = ["general", "operation_mode", "model", "model_specific", "dataset",
             "preprocessing", "training", "quantization", "evaluation", "tools",
             "evaluation", "benchmarking", "deployment", "mlflow", "hydra"]
    parse_top_level(cfg, 
                    mode_groups=mode_groups,
                    mode_choices=mode_choices,
                    legal=legal)
    print(f"[INFO] : Running `{cfg.operation_mode}` operation mode")

    # General section parsing
    if not cfg.general:
        cfg.general = DefaultMunch.fromDict({"project_name": "<unnamed>"})
    legal = ["project_name", "logs_dir", "saved_models_dir", "display_figures", "gpu_memory_limit"]
    required = []
    _parse_general_section(cfg.general, 
                          legal=legal,
                          required=required)

    # Select hardware_type from yaml information
    check_hardware_type(cfg,
                        mode_groups)

    # Dataset section parsing
    if cfg.operation_mode not in ["benchmarking", "deployment"]:
        if not cfg.dataset:
            cfg.dataset = DefaultMunch.fromDict({})
        _parse_dataset_section(cfg.dataset, 
                              mode=cfg.operation_mode, 
                              mode_groups=mode_groups)
    # Model section parsing
    _parse_model_section(cfg.model,
                         mode=cfg.operation_mode,
                         mode_groups=mode_groups)

    # Training section parsing
    if cfg.operation_mode in mode_groups.training:
        # We only have 1 trainer, add it to config in case it wasn't present in the YAML
        cfg.training.trainer_name = 'se_trainer'
        _parse_training_section(cfg.training)

    # Quantization section parsing
    if cfg.operation_mode in mode_groups.quantization:
        # Select the appropriate quantizer based on operation mode
        if cfg.operation_mode is not None and "stream" in cfg.operation_mode:
            cfg.quantization.quantizer = 'onnx_quantizer_stream'
        else:
            cfg.quantization.quantizer = 'onnx_quantizer'
        _parse_quantization_section(cfg.quantization)

    # Evaluation section parsing
    if cfg.operation_mode in mode_groups.evaluation:
        _parse_evaluation_section(cfg.evaluation)

    # Tools section parsing
#    if cfg.operation_mode in (mode_groups.benchmarking + mode_groups.deployment) \
#        or cfg.operation_mode == "evaluation" \
#        or cfg.operation_mode == "prediction":
    if (
        cfg.operation_mode in (mode_groups.benchmarking + mode_groups.deployment)
        or (
            cfg.operation_mode == "evaluation"
            and "evaluation" in cfg
            and "target" in cfg.evaluation
            and cfg.evaluation.target != "host"
        )
        or (
            cfg.operation_mode == "prediction"
            and "prediction" in cfg
            and "target" in cfg.prediction
            and cfg.prediction.target != "host"
        )
    ):
        parse_tools_section(cfg.tools, 
                            cfg.operation_mode,
                            cfg.hardware_type)

    # Benchmarking section parsing
    if cfg.operation_mode in mode_groups.benchmarking:
        parse_benchmarking_section(cfg.benchmarking)
        if cfg.hardware_type == "MPU" :
            if not (cfg.tools.stedgeai.on_cloud):
                print("Target selected for benchmark :", cfg.benchmarking.board)
                print("Offline benchmarking for MPU is not yet available please use online benchmarking")
                exit(1)

    # Deployment section parsing
    if cfg.operation_mode in mode_groups.deployment:
        _parse_deployment_section(cfg.deployment)
        
    # MLFlow section parsing
    parse_mlflow_section(cfg.mlflow)

    return cfg
