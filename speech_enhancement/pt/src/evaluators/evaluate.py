# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from pathlib import Path
from speech_enhancement.pt.src.evaluators import MagSpecONNXEvaluator, MagSpecTorchEvaluator
from speech_enhancement.pt.src.evaluators.myspec_stream import MyMagSpecONNXEvaluator_Stream, MyMagSpecTorchEvaluator_Stream
from common.utils import log_to_file
from common.evaluation import model_is_quantized
import mlflow



class BaseEvaluatorWrapper:
    '''Base Wrapper class for both Torch & ONNX evaluators'''
    def __init__(self, cfg, model, dataloaders):
        '''Initialize the common evaluator wrapper 

        Parameters
        ----------
        cfg, object : User configuration object providing preprocessing and evaluation settings.
        model, object : Model-like object to evaluate; for Torch, an `nn.Module`; for ONNX, a session wrapper.
        dataloaders, dict[str, DataLoader] : Dictionary containing the evaluation dataloader under key "eval_dl".

        Notes
        -----
        Collects preprocessing arguments, prepares log directory, and sets metric names based on model type.
        '''

        self.cfg = cfg
        self.eval_dl = dataloaders["eval_dl"]
        self.model = model
        # Gather preprocessing args that need to be passed to evaluator
        self.preproc_args = {"sampling_rate":cfg.preprocessing.sample_rate,
                        "frame_length":cfg.preprocessing.win_length,
                        "hop_length":cfg.preprocessing.hop_length,
                        "n_fft":cfg.preprocessing.n_fft,
                        "center":cfg.preprocessing.center}
        if cfg.evaluation.logs_path is None:
            self.logs_path = Path(cfg.output_dir, "eval_logs/")
        else:
            self.logs_path = Path(cfg.output_dir, cfg.evaluation.logs_path) 

        # Make dirs if necessary
        self.logs_path.mkdir(parents=False, exist_ok=True)
        if self.model_type == "Quantized":
            self.metric_names = ["quant_pesq", "quant_stoi", "quant_snr", "quant_sisnr", "quant_mse"]
        else:
            self.metric_names = ["pesq", "stoi", "snr", "sisnr", "mse"]

    def _print_and_log_metrics(self):
        '''Print and persist the aggregated evaluation metrics.

        Notes
        -----
        Prints to stdout, logs to a file via `log_to_file`, and records metrics in MLflow using `mlflow.log_metrics`.
        '''
        print("[INFO] Average metrics on test set : ")
        for key in self.metrics_dict.keys():
            print(f"{key} : {self.metrics_dict[key]}")
            log_to_file(self.cfg.output_dir, f"{self.model_type} {key} : {self.metrics_dict[key]}")
        # Log in mlflow
        mlflow.log_metrics(self.metrics_dict)

class SETorchEvaluatorWrapper(BaseEvaluatorWrapper):
    '''Wrapper for Torch evaluators handling STFT-based magnitude spectrogram models.'''
    def __init__(self, cfg, model, dataloaders):
        '''Construct the Torch evaluator wrapper.

        Parameters
        ----------
        cfg, object : User configuration object including preprocessing and evaluation parameters.
        model, nn.Module : Torch model to evaluate.
        dataloaders, dict[str, DataLoader] : Dictionary containing the evaluation dataloader under key "eval_dl".

        Notes
        -----
        Instantiates a `MagSpecTorchEvaluator` with STFT-related preprocessing args and device settings.
        '''
        self.model_type = "Float"
        super().__init__(cfg, model, dataloaders)

        if cfg.operation_mode is not None and "stream" in cfg.operation_mode:
            self.evaluator = MyMagSpecTorchEvaluator_Stream(model=model,
                                eval_data=self.eval_dl,
                                logs_path=self.logs_path,
                                device=self.cfg.evaluation.device,
                                device_memory_fraction=self.cfg.general.gpu_memory_limit,
                                metric_names=self.metric_names,
                                **self.preproc_args
                                )
        else:
            self.evaluator = MagSpecTorchEvaluator(model=model,
                                eval_data=self.eval_dl,
                                logs_path=self.logs_path,
                                device=self.cfg.evaluation.device,
                                device_memory_fraction=self.cfg.general.gpu_memory_limit,
                                metric_names=self.metric_names,
                                **self.preproc_args
                                )

    def evaluate(self):
        '''Run evaluation with the Torch evaluator and log results.

        Returns
        -------
        tuple : (`metrics_dict`, `metrics_array`, `logs_path`)
            - `metrics_dict`, dict[str, float] : Average metrics over the evaluation set.
            - `metrics_array`, np.ndarray : Per-sample metrics with shape (num_samples, num_metrics).
            - `logs_path`, pathlib.Path : Directory containing saved evaluation artifacts.
        '''
        log_to_file(self.cfg.output_dir, f"{self.model_type} model evaluation on dataset {self.cfg.dataset.dataset_name}")
        self.metrics_dict, self.metrics_array = self.evaluator.evaluate()
        self._print_and_log_metrics()
        print("\n [INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {self.logs_path}")

        return self.metrics_dict, self.metrics_array, self.logs_path


class SEONNXEvaluatorWrapper(BaseEvaluatorWrapper):
    '''Wrapper for ONNX evaluators handling STFT-based magnitude spectrogram models.'''
    def __init__(self, cfg, model, dataloaders):
        '''Construct the ONNX evaluator wrapper.

        Parameters
        ----------
        cfg, object : User configuration object including preprocessing and evaluation parameters.
        model, object : ONNX session-like model wrapper used for inference; must expose `_model_path`.
        dataloaders, dict[str, DataLoader] : Dictionary containing the evaluation dataloader under key "eval_dl".
        '''
        # Some logging
        self.model_type = "Quantized" if model_is_quantized(model._model_path) else "Float"
        super().__init__(cfg, model, dataloaders)

        if cfg.operation_mode is not None and "stream" in cfg.operation_mode:
            self.evaluator = MyMagSpecONNXEvaluator_Stream(session=self.model,
                                    eval_data=self.eval_dl,
                                    logs_path=self.logs_path,
                                    fixed_sequence_length=self.cfg.evaluation.fixed_sequence_length,
                                    metric_names=self.metric_names,
                                    **self.preproc_args)
        else:
            self.evaluator = MagSpecONNXEvaluator(session=self.model,
                                    eval_data=self.eval_dl,
                                    logs_path=self.logs_path,
                                    fixed_sequence_length=self.cfg.evaluation.fixed_sequence_length,
                                    metric_names=self.metric_names,
                                    **self.preproc_args)

    def evaluate(self):
        '''Run evaluation with the ONNX evaluator and log results.

        Returns
        -------
        tuple : (`metrics_dict`, `metrics_array`, `logs_path`)
            - `metrics_dict`, dict[str, float] : Average metrics over the evaluation set.
            - `metrics_array`, np.ndarray : Per-sample metrics with shape (num_samples, num_metrics).
            - `logs_path`, pathlib.Path : Directory containing saved evaluation artifacts.
        '''
        log_to_file(self.cfg.output_dir, f"{self.model_type} model evaluation on dataset {self.cfg.dataset.dataset_name}")
        self.metrics_dict, self.metrics_array = self.evaluator.evaluate()
        self._print_and_log_metrics()

        print("\n [INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {self.logs_path}")

        return self.metrics_dict, self.metrics_array, self.logs_path
