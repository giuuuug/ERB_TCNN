'''Trainer for Models that take magnitude spectrograms as input and output masks
   that are applied to the complex spectrogram. Models are expected to have input shape
   (batch, frame_length, sequence_length), e.g. for n_fft=512 with 20 spectrogram frames
   (batch, 257, 20)
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import librosa
import numpy as np
from .base import BaseTrainer
from tqdm import tqdm
from torch.utils.data import default_collate
from pesq import pesq
from pystoi import stoi
from speech_enhancement.pt.src.metrics import si_snr, snr, SISNRLoss
from pathlib import Path

LOUD_SPL_TABLE = {
    20: 99.85,
    25: 93.94,
    31.5: 88.17,
    40: 82.63,
    50: 77.78,
    63: 73.08,
    80: 68.48,
    100: 64.37,
    125: 60.59,
    160: 56.7,
    200: 53.41,
    250: 50.4,
    315: 47.58,
    400: 44.98,
    500: 43.05,
    630: 41.34,
    800: 40.06,
    1000: 40.01,
    1250: 41.82,
    1600: 42.51,
    2000: 39.23,
    2500: 36.51,
    3150: 35.61,
    4000: 36.65,
    5000: 40.01,
    6300: 45.83,
    8000: 51.8,
    10000: 54.28,
    12500: 51.49,
}

class OutputHook(list):
    """ Hook to capture module outputs."""
    def __call__(self, module, input, output):
        self.append(output)

class MyMagSpecTrainer(BaseTrainer):
    '''Trainer class for the ERB-TCNN model.
       Model input and output shape should be (batch, n_fft // 2 + 1, sequence_length).
       Input audio clips are trimmed to the length of the shortest clip in the batch, 
       so this tends to work better with small batch sizes.
       Training metrics are training loss.
       Validation metrics are PESQ, STOI, SNR, SI-SNR and MSE between clean and denoised waveforms.
    '''
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_data: DataLoader,
                 valid_data: DataLoader,
                 frame_length: int,
                 hop_length: int,
                 n_fft: int,
                 center: bool, 
                 sampling_rate: int,
                 window: str = "hann",
                 loss: str = "loud_loss__si_snr",
                 loud_loss_weight: float = 0.005,
                 si_snr_loss_weight: float = 1.0,
                 batching_strat: str = "trim",
                 weight_clipping_max: float = None,
                 activation_regularization: float = None,
                 act_reg_layer_names: list[str] = None,
                 act_reg_layer_types: list = None,
                 act_reg_threshold: float = None,
                 penalty_type: str = "l2",
                 device: str = "cuda:0",
                 save_every: int = 20,
                 ckpt_path: str = "checkpoints/",
                 logs_path: str = "training_logs.csv",
                 snapshot_path: str = "snapshot.pth",
                 device_memory_fraction: float = 0.9,
                 early_stopping: bool = False,
                 reference_metric: str = "pesq",
                 early_stopping_patience: int = 20,
                 ):
        super().__init__(model=model,
                         optimizer=optimizer,
                         train_data=train_data,
                         valid_data=valid_data,
                         device=device,
                         save_every=save_every,
                         ckpt_path=ckpt_path,
                         logs_path=logs_path,
                         snapshot_path=snapshot_path,
                         device_memory_fraction=device_memory_fraction)
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.center = center
        self.loss = loss
        self.loud_loss_weight = loud_loss_weight
        self.si_snr_loss_weight = si_snr_loss_weight
        self.window = window
        self.weight_clipping_max = weight_clipping_max
        self.activation_regularization = activation_regularization
        self.penalty_type = penalty_type
        self.act_reg_layer_names = act_reg_layer_names
        self.act_reg_layer_types = act_reg_layer_types
        self.act_reg_threshold = act_reg_threshold

        allowed_losses = ["loud_loss", "loud_loss__si_snr"]
        assert self.loss in allowed_losses, f"self.loss must be one of {allowed_losses}, was {self.loss}"
        
        self.batching_strat = batching_strat
        assert self.batching_strat in ["trim", "pad"], f"self.batching_strat must be one of 'trim', 'pad', was {self.batching_strat}"
        
        self.sampling_rate = sampling_rate # Needed for PESQ computation

        # Initialize metrics header, and handle reference metric / early stopping params
        self.header = ["train_loss", "val_mse", "pesq", "stoi", "snr", "si-snr"]
        self.early_stopping = early_stopping
        self.reference_metric= reference_metric
        self.early_stopping_patience = early_stopping_patience
        assert self.reference_metric in self.header, f"Reference metric unavailable, must be one of{self.header}, was {reference_metric}"
        self.best_metric = np.inf if self.reference_metric in ["train_loss", "val_mse"] else -np.inf
        self.best_epoch = 0
        self.best_model_state_dict_path = Path(self.ckpt_path, "best_model_state_dict.pth")
        self.si_snr_loss = SISNRLoss(reduction="mean")
        self.num_mel_subbands = 25
        self.loud_loss_eps = 1e-9
        self._prepare_loud_loss_params()

        if type(self.window) not in [np.ndarray, torch.Tensor]:
            # If window is a string or a tuple, pass to librosa.filters.get_window
            # instead of trying to match it to one of the window functions
            # implemented in torch
            self.window = torch.Tensor(librosa.filters.get_window(self.window, Nx=self.frame_length))
        # And move window to device
        self.window = self.window.to(self.device)
        
        # Change dataloader collate function
        if self.batching_strat == "trim":
            self.train_data.collate_fn = self._trim_collate
        elif self.batching_strat == "pad":
            self.train_data.collate_fn = self._zero_pad_collate

        if self.activation_regularization:
            self._attach_regularization_hooks(layer_names=self.act_reg_layer_names,
                                              layer_types=self.act_reg_layer_types)
        if self.penalty_type == "l1":
            self.ord = 1
        elif self.penalty_type == "l2":
            self.ord = 2
        else:
            raise ValueError(f"penalty_type must be one of 'l1', 'l2', was {self.penalty_type}")
    
    def _run_train_epoch(self, epoch): 
        print(f"========= EPOCH {epoch + 1} : training ============")
        epoch_loss = 0
        self.model.train()
        for batch in tqdm(self.train_data):
            batch_loss = self._run_train_batch(batch)
            epoch_loss += batch_loss
        # Normalize by n° of batches
        # Note that this means we're displaying mean of means which is a bit different from 
        # mean of samples.
        epoch_loss = epoch_loss / len(self.train_data)
        print(f"========= EPOCH {epoch + 1} training loss : {epoch_loss} ============")
        self.metrics_array[epoch][0] = epoch_loss

    def _run_train_batch(self, batch):
        self.optimizer.zero_grad()

        # Clear regularization hooks
        if self.activation_regularization:
            for h in self.hooks:
                h.clear()

        if self.batching_strat == "pad":
            noisy_frames, clean_signal, sequence_lengths = batch
            noisy_frames, clean_signal = noisy_frames.to(self.device), clean_signal.to(self.device)
        else:
            noisy_frames, clean_signal = batch
            noisy_frames, clean_signal = noisy_frames.to(self.device), clean_signal.to(self.device)
            sequence_lengths = None

        # Convert noisy complex spectrogram to magnitude spectrogram
        noisy_frames_mag = torch.abs(noisy_frames)
        pred_weighted_mask = self.model(noisy_frames_mag)

        if self.batching_strat == "pad":
            # If batching with zero-padding, apply a loss mask to the model output
            # so that we don't compute the loss on pad frames.
            loss_mask = self._loss_mask(noisy_frames.shape, sequence_lengths=sequence_lengths)
            loss_mask = loss_mask.to(self.device)
            masked_pred_weighted_mask = pred_weighted_mask * loss_mask
            pred_frames = noisy_frames * masked_pred_weighted_mask

        else:
            pred_frames = noisy_frames * pred_weighted_mask

        batch_size = pred_frames.shape[0]
        seq_len = pred_frames.shape[-1]
        time_mask, seq_lengths_tensor = self._build_time_mask(batch_size, seq_len, sequence_lengths)

        pred_mag = torch.abs(pred_frames)
        clean_mag = torch.abs(clean_signal)
        loss = self._compute_loud_loss(pred_mag, clean_mag, time_mask, seq_lengths_tensor)

        if self.loss == "loud_loss__si_snr":
            si_loss = self._compute_si_snr(pred_frames, clean_signal, seq_lengths_tensor)
            loss = self.loud_loss_weight * loss + self.si_snr_loss_weight * si_loss
        else:
            loss = self.loud_loss_weight * loss

        if self.activation_regularization:
            reg_penalty = 0
            for h in self.hooks:
                for out in h:
                    if self.act_reg_threshold:
                        reg_mask = (torch.abs(out) >= self.act_reg_threshold)
                        out = out * reg_mask
                    reg_penalty += torch.norm(out, self.ord)
            reg_penalty *= self.activation_regularization
            loss += reg_penalty
        
        loss.backward()
        self.optimizer.step()
        # Clip weights after gradient update
        if self.weight_clipping_max:
            for p in self.model.parameters():
                p.data = p.data.clamp(min=-self.weight_clipping_max, max=self.weight_clipping_max)

        return loss

    def _run_validation_epoch(self, epoch):
        # We expect the batch size in the validation dataloader to be 1, 
        # so 1 batch corresponds to 1 noisy/clean pair, and we're not doing 
        # any padding or trimming for the sake of batching like during training
        print(f"========= EPOCH {epoch + 1} : validation ============")
        self.model.eval()
        with torch.no_grad():
            num_batches = len(self.valid_data)
            # Should put all this into a dict probably
            valid_metrics = {
                "train_loss": 0,
                "pesq": 0,
                "stoi": 0,
                "snr": 0,
                "si-snr": 0
            }
            for batch in tqdm(self.valid_data):
                batch_loss, batch_pesq, batch_stoi, batch_snr, batch_si_snr = self._run_validation_batch(batch)
                valid_metrics["train_loss"] += batch_loss
                valid_metrics["pesq"] += batch_pesq
                valid_metrics["stoi"] += batch_stoi
                valid_metrics["snr"] += batch_snr
                valid_metrics["si-snr"] += batch_si_snr

            valid_metrics["train_loss"] /= num_batches
            valid_metrics["pesq"] /= num_batches
            valid_metrics["stoi"] /= num_batches
            valid_metrics["snr"] /= num_batches
            valid_metrics["si-snr"] /= num_batches
        
        print(f"Validation MSE : {valid_metrics['train_loss']} \n"
              f"Validation PESQ : {valid_metrics['pesq']} \n"
              f"Validation STOI : {valid_metrics['stoi']} \n"
              f"Validation SNR : {valid_metrics['snr']} \n"
              f"Valid SI-SNR : {valid_metrics['si-snr']} \n")

        self.metrics_array[epoch][1:] = [valid_metrics["train_loss"], valid_metrics["pesq"], valid_metrics["stoi"], valid_metrics["snr"], valid_metrics["si-snr"]]
        
        # Early stopping stuff
        # Update the best model, best ref metric and best epoch
        self._update_best_model(value=valid_metrics[self.reference_metric], epoch=epoch)
        # If early stopping is enabled and patience is exceeded, return that we need to stop training
        return (self.early_stopping and (epoch - self.best_epoch > self.early_stopping_patience))

    def _run_validation_batch(self, batch):
        noisy_frames, clean_wave = batch
        noisy_frames = noisy_frames.to(self.device)

        # Convert noisy complex spectrogram to magnitude spectrogram
        noisy_frames_mag = torch.abs(noisy_frames)
        pred_weighted_mask = self.model(noisy_frames_mag)
        pred_frames = noisy_frames * pred_weighted_mask

        # Keep validation reconstruction aligned with ONNX evaluator:
        # use librosa.istft and the configured `center` setting.
        window = self.window
        if isinstance(window, torch.Tensor):
            window = window.detach().to("cpu").numpy()

        pred_frames_np = pred_frames.detach().to("cpu").numpy()
        pred_wave = librosa.istft(np.squeeze(pred_frames_np), n_fft=self.n_fft, hop_length=self.hop_length,
                                  win_length=self.frame_length, window=window, center=self.center)

        clean_source = clean_wave.detach().to("cpu").numpy().squeeze()
        denoised = np.squeeze(pred_wave)
        wave_len = min(denoised.shape[-1], clean_source.shape[-1])
        clean_source = clean_source[:wave_len]
        denoised = denoised[:wave_len]

        valid_loss = np.mean((denoised - clean_source) ** 2)

        valid_pesq = pesq(fs=self.sampling_rate,
                          ref=clean_source,
                          deg=denoised,
                          mode="wb")
        valid_stoi = stoi(x=clean_source,
                          y=denoised,
                          fs_sig=self.sampling_rate)
        valid_snr = snr(ref=clean_source, deg=denoised)
        valid_si_snr = si_snr(ref=clean_source,
                              deg=denoised)
        return (valid_loss, valid_pesq, valid_stoi, valid_snr, valid_si_snr)

    @staticmethod
    def _trim_collate(batch):
        # Intended as a replacement for the default collate_fn of a dataloader
        # Trims the end of sequences, so that the length of all sequences match the shortest in the batch.
        # Assumes batch is a (noisy_frames, clean_frames) tuple of numpy arrays
        # Assumes sequence length axis is the last axis in both arrays
        min_noisy_seq_len = min([elem[0].shape[-1] for elem in batch])
        min_clean_seq_len = min([elem[1].shape[-1] for elem in batch])
        for i, elem in enumerate(batch):
            # Convert to tensor if needed
            if isinstance(elem[0], np.ndarray):
                noisy = torch.as_tensor(elem[0])
            else:
                noisy = elem[0]
            if isinstance(elem[1], np.ndarray):
                clean = torch.as_tensor(elem[1])
            else:
                clean = elem[1]

            # Trim both noisy and clean sequences
            trimmed_noisy = torch.narrow(noisy, dim=-1, start=0, length=min_noisy_seq_len)
            trimmed_clean = torch.narrow(clean, dim=-1, start=0, length=min_clean_seq_len)

            batch[i] = (trimmed_noisy, trimmed_clean)
        # Give trimmed batch back to default dataloader collate function
        return default_collate(batch)
    
    @staticmethod
    def _zero_pad_collate(batch):
    # Intended as a replacement for the default collate_fn of a dataloader
    # Zero-pads sequences to the right, with the max length being the longest length of the batch.
    # Assumes batch is a (noisy_frames, clean_frames) tuple of numpy arrays
        max_noisy_seq_len = max([elem[0].shape[-1] for elem in batch])
        max_clean_seq_len = max([elem[1].shape[-1] for elem in batch])

        for i, elem in enumerate(batch):
            # Convert to tensor if needed
            if isinstance(elem[0], np.ndarray):
                noisy = torch.as_tensor(elem[0])
            else:
                noisy = elem[0]
            if isinstance(elem[1], np.ndarray):
                clean = torch.as_tensor(elem[1])
            else:
                clean = elem[1]

            # Pad both noisy and clean sequences
            noisy_seq_len = noisy.shape[-1]
            clean_seq_len = clean.shape[-1]

            noisy_pad_lengths = (0, max_noisy_seq_len - noisy_seq_len)
            clean_pad_lengths = (0, max_clean_seq_len - clean_seq_len)
            # Assuming noisy & clean sequences have same length
            # nn.functional.pad pads starting from the last axis
            padded_noisy = nn.functional.pad(noisy, noisy_pad_lengths, mode="constant")
            padded_clean = nn.functional.pad(clean, clean_pad_lengths, mode="constant")
            # Add original noisy sequence lengths in batch. We need this info to compute
            # the loss mask later.
            batch[i] = (padded_noisy, padded_clean, noisy_seq_len)
        # Give padded batch back to default dataloader collate function
        return default_collate(batch)
    
    @staticmethod
    # Loss mask : we don't want to compute loss on pad frames.
    def _loss_mask(batch_shape, sequence_lengths):
        mask = torch.zeros(batch_shape, requires_grad=False)
        # Manually setting loss mask to 1 on frames that are not pad frames
        for k, seq_len in enumerate(sequence_lengths):
            mask[k, :, :seq_len] += 1.0
        return mask

    def _build_time_mask(self, batch_size, seq_len, sequence_lengths=None):
        if sequence_lengths is None:
            sequence_lengths_tensor = torch.full((batch_size,), seq_len, dtype=torch.int64, device=self.device)
        else:
            sequence_lengths_tensor = torch.as_tensor(sequence_lengths, dtype=torch.int64, device=self.device)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=self.device)
        for idx in range(batch_size):
            length = int(sequence_lengths_tensor[idx].item())
            clipped_length = min(length, seq_len)
            if clipped_length > 0:
                mask[idx, :clipped_length] = 1.0
        return mask.unsqueeze(1), sequence_lengths_tensor

    def _prepare_loud_loss_params(self):
        self.spl_freqs = np.array(sorted(LOUD_SPL_TABLE.keys()))
        self.spl_values = np.array([LOUD_SPL_TABLE[freq] for freq in self.spl_freqs])
        self.spl_reference = self._lookup_spl(1000.0)
        mel_min = self._hz_to_mel(0.0)
        mel_max = self._hz_to_mel(self.sampling_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, self.num_mel_subbands + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_width = self.sampling_rate / self.n_fft
        k_c = np.floor(hz_points / bin_width).astype(int)
        k_c = np.clip(k_c, 0, self.n_fft // 2)
        k_c = np.maximum.accumulate(k_c)
        k_c[-1] = self.n_fft // 2
        band_infos = []
        for i in range(self.num_mel_subbands):
            start = int(k_c[i])
            end = int(k_c[i + 2])
            if end <= start:
                continue
            center_freq = float(hz_points[i + 1])
            weight = self._compute_band_weight(center_freq)
            band_infos.append({
                "start": start,
                "end": end,
                "weight": weight,
                "freq_bins": end - start
            })
        if not band_infos:
            raise ValueError("Loud-loss band split did not produce any valid sub-bands.")
        self.band_infos = band_infos

    @staticmethod
    def _hz_to_mel(freq):
        return 2595 * np.log10(1 + freq / 700)

    @staticmethod
    def _mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _lookup_spl(self, freq):
        idx = int(np.argmin(np.abs(self.spl_freqs - freq)))
        return float(self.spl_values[idx])

    def _compute_band_weight(self, freq):
        spl_value = self._lookup_spl(freq)
        return self.spl_reference / spl_value

    def _mag_to_log_power(self, magnitude):
        return 10.0 * torch.log10(magnitude.pow(2) + self.loud_loss_eps)

    def _compute_loud_loss(self, pred_mag, clean_mag, time_mask, sequence_lengths):
        pred_log = self._mag_to_log_power(pred_mag)
        clean_log = self._mag_to_log_power(clean_mag)
        mask = time_mask.to(pred_log.dtype)
        lengths = sequence_lengths.to(pred_log.dtype)
        total_loss = torch.tensor(0.0, dtype=pred_log.dtype, device=pred_log.device)
        for info in self.band_infos:
            freq_bins = info["freq_bins"]
            if freq_bins <= 0:
                continue
            diff = (pred_log[:, info["start"]:info["end"], :] - clean_log[:, info["start"]:info["end"], :]) ** 2
            masked_diff = diff * mask
            sum_sq = masked_diff.sum(dim=(1, 2))
            denominator = freq_bins * lengths + self.loud_loss_eps
            sample_loss = sum_sq / denominator
            weight_tensor = torch.tensor(info["weight"], dtype=pred_log.dtype, device=pred_log.device)
            total_loss = total_loss + weight_tensor * sample_loss.mean()
        return total_loss

    def _compute_si_snr(self, pred_frames, clean_frames, sequence_lengths):       
        
        # When center=False, ISTFT expects specific input/output relationships.
        # To avoid the RuntimeError: istft ... window overlap add min: 1, which happens when
        # center=False and padding is insufficient or reconstruction is tricky at edges,
        # we enforce center=True just for the loss calculation and handle shapes if needed,
        # OR we ensure the frames provided to istft are sufficient.
        
        # HACK: If we are training with center=False (streaming), using center=True here
        # is a safe approximation for calculating SI-SNR loss on the whole batch, 
        # as it just adds a bit of padding at edges which shouldn't affect the GLOBAL SNR too much
        # and avoids the crash. The model still learns to output causal frames.
        
        # However, for correctness let's try to stick to config. If it fails, we might need to pad frames manually.
        # The error "window overlap add min: 1" usually means we are trying to reconstruct a signal
        # where some parts don't have enough overlap (start/end) because center=False doesn't pad.
        
        # Fix strategy: Force center=True for reconstruction during loss calculation even if model is streaming.
        # This allows ISTFT to work without crashing on edge cases. 
        # The frames `pred_frames` are what the model outputted. Reconstructing them with center=True
        # might shift them slightly in time (half window), but SI-SNR is scale invariant, not shift invariant.
        # So we must apply the same logic to `clean_frames`.
        
        # Actually, the error might be due to `clean_frames` often being the STFT of the original signal
        # computed with center=False? If so, we need to be consistent.

        # Let's try forcing center=True just for this calculation to see if it stabilizes training.
        # Since we do it for both pred and clean, the shift is consistent.
        
        # REVERTING previous logic to just use center=True to avoid crash, assuming the time-shift cancels out
        # because we do it for both signals.
        
        pred_wave = torch.istft(pred_frames, n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.frame_length, window=self.window, center=True)
        clean_wave = torch.istft(clean_frames, n_fft=self.n_fft, hop_length=self.hop_length,
                                 win_length=self.frame_length, window=self.window, center=True)
                                     
        si_snr_loss = torch.tensor(0.0, dtype=pred_wave.dtype, device=pred_wave.device)
        valid_segments = 0
        for idx in range(pred_frames.shape[0]):
            frames = int(sequence_lengths[idx].item())
            if frames <= 0:
                continue
            wave_len = self.frame_length + max(frames - 1, 0) * self.hop_length
            wave_len = min(wave_len, pred_wave.shape[-1])
            if wave_len <= 0:
                continue
            pred_slice = pred_wave[idx, :wave_len]
            clean_slice = clean_wave[idx, :wave_len]
            if pred_slice.numel() == 0 or clean_slice.numel() == 0:
                continue
            si_snr_loss += self.si_snr_loss(pred_slice.unsqueeze(0), clean_slice.unsqueeze(0))
            valid_segments += 1
        if valid_segments == 0:
            return torch.tensor(0.0, dtype=si_snr_loss.dtype, device=si_snr_loss.device)
        return si_snr_loss / valid_segments

    def _attach_regularization_hooks(self, layer_names=None, layer_types=None):
        self.hooks = []
        named_modules = dict(self.model.named_modules())
        if layer_names is not None:
            for k in named_modules.keys():
                if k in layer_names:
                    hook = OutputHook()
                    named_modules[k].register_forward_hook(hook)
                    self.hooks.append(hook)
                    print(f"Attached output hook to layer {k}")

        elif layer_types is not None:
            for k in named_modules.keys():
                if type(named_modules[k]).__name__ in layer_types:
                    hook = OutputHook()
                    named_modules[k].register_forward_hook(hook)
                    self.hooks.append(hook)
                    print(f"Attached output hook to layer {k} of type {type(named_modules[k])}")

    def _update_best_model(self, value, epoch):
        '''Updates best metric and best model'''
        if self.reference_metric in ["train_loss", "wave_mse"]:
            if value <= self.best_metric:
                self.best_metric = value
                # We need to save/load state dict to untie self.model and self.best_model
                torch.save(self.model.state_dict(), self.best_model_state_dict_path)
                self.best_model.load_state_dict(torch.load(self.best_model_state_dict_path,
                                                           weights_only=True))
                self.best_epoch = epoch
        elif self.reference_metric in ["pesq", "stoi", "snr", "si-snr"]:
            if value >= self.best_metric:
                self.best_metric = value
                # We need to save/load state dict to untie self.model and self.best_model
                torch.save(self.model.state_dict(), self.best_model_state_dict_path)
                self.best_model.load_state_dict(torch.load(self.best_model_state_dict_path,
                                                           weights_only=True))
                self.best_epoch = epoch
        else:
            raise ValueError(f"reference metric must be in {self.header}")
