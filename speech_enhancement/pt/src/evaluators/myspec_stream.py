# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

'''Evaluators for frequency domain models with STFT pre-processing'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
import numpy as np
from pathlib import Path
from pesq import pesq
from pystoi import stoi
from speech_enhancement.pt.src.metrics import si_snr, snr
from speech_enhancement.pt.src.evaluators import BaseTorchEvaluator, BaseONNXEvaluator


class MyMagSpecTorchEvaluator_Stream(BaseTorchEvaluator):
    '''Evaluator for torch LSTM frequency domain models with STFT preprocessing.'''
    def __init__(self,
                 model: nn.Module,
                 eval_data: DataLoader,
                 logs_path: Path,
                 frame_length: int,
                 hop_length: int,
                 n_fft: int,
                 center: bool, 
                 sampling_rate: int,
                 window: str = "hann",
                 device: str = "cuda:0",
                 device_memory_fraction: float = 0.5,
                 metric_names=["pesq", "stoi", "snr", "sisnr", "mse"],
                 model_checkpoint=None):

        '''
        Parameters
        ----------
        model, nn.Module : Torch model object to evaluate
        model_checkpoint, str or Path : Optional, Path to the state dict of the model to evaluate
        eval_data, Dataloader : Dataloader for the evaluation data.
            /!\ Batch size of the dataloader should be 1 /!\ 
            This is because we do not perform trimming or padding of audio clips during evaluation.
        logs_path, str or Path : Path to the directory where evaluation logs are to be saved
        frame_length, int : Length of STFT frames in samples before padding
        hop_length, int : Number of samples between successive STFT frames
        n_fft, int : Length of windowed signal after padding.
            Also number of Fourier coefficients used in inverse STFT.
        center, bool : if True the signal y is padded so that frames are centered.
        sampling_rate, int : Sampling rate of audio in eval_data
        window , str, np.ndarray or torch.Tensor : Window used for inverse STFT. 
            Use the same one as was used to compute the STFT.
            If string, passed to librosa.get_window. If array or Tensor, used as-is.
        device, str : Torch device on which to run the evaluation
        device_memory_fraction, float : Portion of memory used on device by the evaluator

        Notes
        -----
        This evaluator can be used to evaluate any model that takes magnitude spectrogram as input
        and outputs a mask applied to the complex spectrogram, not just LSTM models.
        The model's input and output shape should be (batch, n_fft // 2 + 1, sequence_length)
        e.g., for a model with n_fft=512, it should be (batch, 257, sequence_length)
        '''
        super().__init__(model=model,
                         model_checkpoint=model_checkpoint,
                         eval_data=eval_data,
                         metric_names=metric_names,
                         logs_path=logs_path,
                         device=device,
                         device_memory_fraction=device_memory_fraction)
        self.frame_length = frame_length
        self.hop_length=hop_length
        self.n_fft = n_fft
        self.center = center
        self.sampling_rate = sampling_rate
        self.window = window
        if type(self.window) not in [np.ndarray, torch.Tensor]:
            # If window is a string or a tuple, pass to librosa.filters.get_window
            # instead of trying to match it to one of the window functions
            # implemented in torch
            self.window = torch.Tensor(librosa.filters.get_window(self.window, Nx=self.frame_length))
        # And move window to device
        self.window = self.window.to(self.device)


    def _run_evaluation_step(self, batch):
        noisy_frames, clean_wave = batch
        noisy_frames = noisy_frames.to(self.device)

        # Convert noisy complex spectrogram to magnitude spectrogram
        noisy_frames_mag = torch.abs(noisy_frames)
        
        B, C, T = noisy_frames_mag.shape
        states = self.model.get_initial_states(batch_size=B, device=noisy_frames_mag.device)
        
        pred_weighted_mask, states = self.model(noisy_frames_mag, states)
        
        pred_frames = noisy_frames * pred_weighted_mask
        pred_wave = torch.istft(pred_frames, n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.frame_length, window=self.window, center=self.center)
        # Squeeze waves
        pred_wave, clean_wave = pred_wave.squeeze(), clean_wave.squeeze()
        # Trim clean wave to the length of predicted wave
        clean_wave = clean_wave[:pred_wave.shape[-1]]
        # Put everything back on CPU
        # Then compute metrics
        denoised, clean_source = pred_wave.to("cpu"), clean_wave.to("cpu")
        # Back to numpy
        denoised = denoised.numpy()
        clean_source = clean_source.numpy()
        
        if not self.center:
            # When center=False, ISTFT has artifacts at the boundaries due to lack of overlap-add
            # Discard the first and last frame_length samples to compute accurate metrics
            offset = self.frame_length
            if min(len(denoised), len(clean_source)) > 2 * offset:
                clean_source = clean_source[offset:-offset]
                denoised = denoised[offset:-offset]

        eval_mse = np.mean((denoised - clean_source) ** 2)

        eval_pesq = pesq(fs=self.sampling_rate,
                         ref=clean_source,
                         deg=denoised,
                         mode="wb")
        eval_stoi = stoi(x=clean_source,
                         y=denoised,
                         fs_sig=self.sampling_rate)
        eval_snr = snr(ref=clean_source,
                       deg=denoised)
        eval_si_snr = si_snr(ref=clean_source,
                             deg=denoised)
        return (eval_pesq, eval_stoi, eval_snr, eval_si_snr, eval_mse)
    
class MyMagSpecONNXEvaluator_Stream(BaseONNXEvaluator):
    def __init__(self,
                 eval_data: DataLoader,
                 logs_path: Path,
                 frame_length: int,
                 hop_length: int,
                 n_fft: int,
                 center: bool, 
                 sampling_rate: int,
                 window: str = "hann",
                 fixed_sequence_length: int = None,    # Trim input tensors in case you want to run a quantized ONNX with fixed input shape
                 metric_names=["pesq", "stoi", "snr", "sisnr", "mse"],
                 model_path: str = None,
                 session = None): 

        '''
        Parameters
        ----------
        model_path, str or Path : Path to the saved ONNX model to evaluate
        eval_data, Dataloader : Dataloader for the evaluation data.
            /!\ Batch size of the dataloader should be 1 /!\ 
        logs_path, str or Path : Path to the directory where evaluation logs are to be saved
        frame_length, int : Length of STFT frames in samples before padding
        hop_length, int : Number of samples between successive STFT frames
        n_fft, int : Length of windowed signal after padding.
            Also number of Fourier coefficients used in inverse STFT.
        center, bool : if True the signal y is padded so that frames are centered.
        sampling_rate, int : Sampling rate of audio in eval_data
        window , str, np.ndarray or torch.Tensor : Window used for inverse STFT. 
            Use the same one as was used to compute the STFT.
            If string, passed to librosa.get_window. If array or Tensor, used as-is.
        fixed_sequence_length, int or None : If int, will trim input sequenced to the specified length.
            Useful for evaluating ONNX models with a sequence length axis of fixed length.
        session, onnxruntime InferenceSession : Session corresponding to the model to evaluate.
            Mutually exclusive with model_path.

        Notes
        -----
        This evaluator can be used to evaluate any model that takes magnitude spectrogram as input
        and outputs a mask applied to the complex spectrogram, not just LSTM models.
        The model's input and output shape should be (batch, n_fft // 2 + 1, sequence_length)
        e.g., for a model with n_fft=512, it should be (batch, 257, sequence_length)
        '''
        super().__init__(model_path=model_path,
                         session=session,
                         eval_data=eval_data,
                         metric_names=metric_names,
                         logs_path=logs_path)
        self.frame_length = frame_length
        self.hop_length=hop_length
        self.n_fft = n_fft
        self.center = center
        self.sampling_rate = sampling_rate
        self.window = window
        if type(self.window) not in [np.ndarray, torch.Tensor]:
            # If window is a string or a tuple, pass to librosa.filters.get_window
            self.window = librosa.filters.get_window(self.window, Nx=self.frame_length)
        self.fixed_sequence_length = fixed_sequence_length

    def _run_evaluation_step(self, batch):
        noisy_frames, clean_wave = batch
        
        # Trim input specotrogram if necessary
        # We assume batch size is 1
        if self.fixed_sequence_length is not None:
            if noisy_frames.shape[-1] >= self.fixed_sequence_length:
                noisy_frames = noisy_frames[:, :, :self.fixed_sequence_length]
            else: # Pad a little
                print("Input shorter than prescribed fixed_sequence_length."
                      f"Added {self.fixed_sequence_length - noisy_frames.shape[-1]} pad frames")
                pad_lengths = [(0, 0)] * noisy_frames.ndim
                pad_lengths[-1] = (0, self.fixed_sequence_length - noisy_frames.shape[-1])
                noisy_frames = np.pad(noisy_frames, pad_width=pad_lengths, mode="constant")

        # Convert noisy complex spectrogram to magnitude spectrogram
        noisy_frames_mag = np.abs(noisy_frames)
        
        B, C, T = noisy_frames_mag.shape
        onnx_inputs = self.session.get_inputs()
        
        states = {}
        for inp in onnx_inputs[1:]:
            states[inp.name] = np.zeros(shape=inp.shape).astype(np.float32)
        
        pred_weighted_mask_list = []
        for t in range(T):
            input_frame = noisy_frames_mag[:, :, t:t+1].astype(np.float32)
            inputs_dict = {onnx_inputs[0].name: input_frame, **states}
            
            onnx_outs = self.session.run(None, inputs_dict)
            
            pred_weighted_mask_list.append(onnx_outs[0])
            for i, inp in enumerate(onnx_inputs[1:]):
                states[inp.name] = onnx_outs[i + 1]

        pred_weighted_mask = np.concatenate(pred_weighted_mask_list, axis=-1)
        
        pred_frames = noisy_frames * pred_weighted_mask
        pred_wave = librosa.istft(pred_frames, n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.frame_length, window=self.window, center=self.center)
        # Squeeze waves
        pred_wave, clean_wave = np.squeeze(pred_wave), np.squeeze(clean_wave)
        # Trim waves to the length of the shortest one
        wave_len = min(pred_wave.shape[-1], clean_wave.shape[-1])
        clean_source = clean_wave[:wave_len]
        denoised = pred_wave[:wave_len]
        # Then compute metrics

        if not self.center:
            # When center=False, ISTFT has artifacts at the boundaries due to lack of overlap-add
            # Discard the first and last frame_length samples to compute accurate metrics
            offset = self.frame_length
            if wave_len > 2 * offset:
                clean_source = clean_source[offset:-offset]
                denoised = denoised[offset:-offset]

        eval_mse = np.mean((denoised - clean_source) ** 2)

        try: # Sometimes the pesq packages fails to detect a speech signal for some reason
            eval_pesq = pesq(fs=self.sampling_rate,
                            ref=clean_source,
                            deg=denoised,
                            mode="wb")
        except Exception as e:
            print("PESQ package failed to detect speech signal")
            print(e)
            eval_pesq = 1
        eval_stoi = stoi(x=clean_source,
                         y=denoised,
                         fs_sig=self.sampling_rate)
        eval_snr = snr(ref=clean_source,
                       deg=denoised)
        eval_si_snr = si_snr(ref=clean_source,
                             deg=denoised)
        return (eval_pesq, eval_stoi, eval_snr, eval_si_snr, eval_mse)
        