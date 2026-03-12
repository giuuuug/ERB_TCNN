# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import hydra
import argparse
import mlflow
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from api import get_model, get_dataloaders, get_trainer, get_quantizer, get_evaluator
from common.benchmarking.common_benchmark import _get_credentials
from common.utils import mlflow_ini, log_to_file
from common.evaluation import model_is_quantized
from common.benchmarking import benchmark
from speech_enhancement.pt.src.utils import get_config
from speech_enhancement.pt.src.deployment import deploy
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

def _process_mode(cfg):
    """
    Process the selected mode of operation.

    Args:
        cfg (DictConfig): The configuration object.

    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """
    mode = cfg.operation_mode
    mlflow.log_param("model_path", cfg.model.model_path)
    # Logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

     # Connect to STM32Cube.AI Developer Cloud if needed
    if cfg.tools and cfg.tools.stedgeai and cfg.tools.stedgeai.on_cloud:
        credentials = _get_credentials()

    model = get_model(cfg)
    

    if cfg.operation_mode not in ["benchmarking", "deployment"]:
        # Keep compatibility with old config files
        if cfg.dataset.name and not cfg.dataset.dataset_name:
            cfg.dataset.dataset_name = cfg.dataset.name
        dataloaders = get_dataloaders(cfg)

    if mode == "training":
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        print("[INFO] Training model")
        _, best_model_session = trainer.train()
        
    elif mode == "evaluation":
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        print("[INFO] Evaluating model")
        evaluator.evaluate()

    elif mode == "quantization":
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        print("[INFO] Quantizing model")
        quantizer.quantize()


    
    elif mode == "benchmarking":
        model_path_to_benchmark = Path(cfg.model.model_path)
        benchmark(cfg, model_path_to_benchmark=model_path_to_benchmark, credentials=credentials)

    elif mode == "deployment":
        deploy(cfg)
    
    elif mode == "chain_tqe":
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        print("[INFO] Training model")
        _, best_model_session = trainer.train()
        print("[INFO] Evaluating float model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=best_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        print("[INFO] Quantizing model")
        quantizer = get_quantizer(cfg=cfg, 
                                  model=best_model_session, 
                                  dataloaders=dataloaders)
        quantized_model_session, _ = quantizer.quantize()
        
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
    
    elif mode == "chain_tqe_stream":
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        print("[INFO] Training model")
        _, best_model_session = trainer.train()
        print("[INFO] Evaluating float model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=best_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        print("[INFO] Quantizing model")
        quantizer = get_quantizer(cfg=cfg, 
                                  model=best_model_session, 
                                  dataloaders=dataloaders)
        quantized_model_session, _ = quantizer.quantize()
        
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
     
    elif mode == "chain_eqe":
        if model_is_quantized(cfg.model.model_path):
            raise ValueError("Tried to run chain_eqe on a quantized ONNX model. \n"
                             "Chain_eqe can only be run on float ONNX models. \n"
                             "If you wish to evaluate a quantized ONNX model, use the 'evaluate' mode instead.")
        
        print("[INFO] Evaluating float model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        print("[INFO] Quantizing model")
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model_session, _ = quantizer.quantize()
        
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        
    elif mode == "chain_tqeb":

        print("[INFO] Training model")
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        _, best_model_session = trainer.train()
        print("[INFO] Evaluating float model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=best_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()

        print("[INFO] Quantizing model")
        quantizer = get_quantizer(cfg=cfg, 
                                  model=best_model_session, 
                                  dataloaders=dataloaders)
        quantized_model_session, quantized_static_model_session = quantizer.quantize()
        
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        quantized_static_model_path = quantized_static_model_session._model_path
        
        benchmark(cfg, model_path_to_benchmark=quantized_static_model_path,
                  credentials=credentials)
        
    elif mode == "chain_eqeb":
        if model_is_quantized(cfg.model.model_path):
            raise ValueError("Tried to run chain_eqe on a quantized ONNX model. \n"
                             "Chain_eqe can only be run on float ONNX models. \n"
                             "If you wish to evaluate a quantized ONNX model, use the 'evaluate' mode instead.")
        
        print("[INFO] Evaluating float model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        
        print("[INFO] Quantizing model")
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
            
        quantized_model_session, quantized_static_model_session = quantizer.quantize()
        
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model_session, 
                                  dataloaders=dataloaders)
        evaluator.evaluate()
        
        quantized_static_model_path = quantized_static_model_session._model_path
        
        benchmark(cfg, model_path_to_benchmark=quantized_static_model_path,
                  credentials=credentials)

    elif mode == "chain_qb":
        if model_is_quantized(cfg.model.model_path):
            raise ValueError("Tried to run chain_qb on a quantized ONNX model. \n"
                             "Chain_qb can only be run on float ONNX models.")

        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        print("[INFO] Quantizing model")
        quantized_model_session, quantized_static_model_session = quantizer.quantize()
        quantized_static_model_path = quantized_static_model_session._model_path
        
        benchmark(cfg, model_path_to_benchmark=quantized_static_model_path,
                  credentials=credentials)
        
    elif mode == "chain_qd":
        if model_is_quantized(cfg.model.model_path):
            raise ValueError("Tried to run chain_qd on a quantized ONNX model. \n"
                             "Chain_qd can only be run on float ONNX models.")
        
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        print("[INFO] Quantizing model")
        quantized_model_session, quantized_static_model_session = quantizer.quantize()
        quantized_static_model_path = quantized_static_model_session._model_path
        
        deploy(cfg, model_path_to_deploy=quantized_static_model_path)
        
    else:
        raise ValueError(f"Invalid operation mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(cfg.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        mlflow.log_param("stedgeai_core_version", cfg.tools.stedgeai.version)
        mlflow.log_param("target", cfg.benchmarking.board)
    
    # Logging the completion of the chain
    log_to_file(cfg.output_dir, f'operation finished: {mode}')

    # ClearML
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(cfg)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        None
    """

    cfg = _fw_agnostic_initializations(cfg)
    cfg = _pt_specific_initializations(cfg)
    _process_mode(cfg=cfg)

def _pt_specific_initializations(cfg: DictConfig = None) -> DictConfig:
    # Accomodate get_model() api
    cfg.use_case = "speech_enhancement"
    cfg.model.framework = "torch"
    return cfg

def _fw_agnostic_initializations(cfg: DictConfig = None) -> DictConfig:
    """
    Framework-agnostic initializations.

    This function performs initializations that are independent of the specific deep learning framework being used.
    It includes parsing the configuration file, setting up MLFlow, and initializing ClearML if a valid configuration
    file is found.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        DictConfig: Updated configuration object with initialized settings.
    """
    # Parse the configuration file and set the output directory
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir

    # Initialize MLFlow
    mlflow_ini(cfg)

    # Check if there's a valid ClearML configuration file and initialize ClearML
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='aed_modelzoo_task')
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)
    
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')

    # Add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()