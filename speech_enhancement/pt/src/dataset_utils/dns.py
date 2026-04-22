import torch
import librosa
import torchaudio
import numpy as np
from pathlib import Path
import re
import random
from math import floor


class DNSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        set: str,
        clean_train_files_path: Path,
        noisy_train_files_path: Path,
        clean_valid_files_path: Path,
        noisy_valid_files_path: Path,
        clean_test_files_path: Path,
        noisy_test_files_path: Path,
        input_pipeline,
        sample_rate: int = 16000,
        target_pipeline=None,
        file_extension: str = ".wav",
        preproc_lib: str = "librosa",
        device: str = "cpu",
        n_clips=None,
        random_seed: int = 42,
    ):
        super().__init__()
        self.set = set
        self.file_extension = file_extension
        self.sample_rate = sample_rate
        self.input_pipeline = input_pipeline
        self.target_pipeline = target_pipeline
        self.preproc_lib = preproc_lib
        self.device = device
        self.n_clips = n_clips
        self.random_seed = random_seed

        if self.set not in ["train", "valid", "test"]:
            raise ValueError(f"set must be one of 'train', 'valid', 'test', was {self.set}")
        if self.preproc_lib not in ["librosa", "torchaudio"]:
            raise ValueError("preproc_lib must be one of ['librosa', 'torchaudio']")

        split_paths = {
            "train": (Path(clean_train_files_path), Path(noisy_train_files_path)),
            "valid": (Path(clean_valid_files_path), Path(noisy_valid_files_path)),
            "test": (Path(clean_test_files_path), Path(noisy_test_files_path)),
        }
        self.clean_files_path, self.noisy_files_path = split_paths[self.set]

        self.clean_file_list = sorted(self.clean_files_path.glob("*" + self.file_extension))
        self.noisy_file_list = sorted(self.noisy_files_path.glob("*" + self.file_extension))

        if len(self.clean_file_list) != len(self.noisy_file_list):
            raise ValueError(
                f"Different number of clean and noisy files in '{self.set}' split: "
                f"{len(self.clean_file_list)} clean vs {len(self.noisy_file_list)} noisy"
            )

        clean_by_id = {self._extract_file_id(p): p for p in self.clean_file_list}
        noisy_by_id = {self._extract_file_id(p): p for p in self.noisy_file_list}

        clean_ids = clean_by_id.keys()
        noisy_ids = noisy_by_id.keys()
        if clean_ids != noisy_ids:
            missing_in_noisy = len(clean_ids - noisy_ids)
            missing_in_clean = len(noisy_ids - clean_ids)
            raise ValueError(
                "Clean/noisy file IDs do not match in split "
                f"'{self.set}' (missing_in_noisy={missing_in_noisy}, missing_in_clean={missing_in_clean})"
            )

        # DNS clean/noisy filenames differ, but they share a trailing fileid_N token.
        self.file_pairs = [(noisy_by_id[file_id], clean_by_id[file_id]) for file_id in sorted(clean_ids)]
        total_clips = len(self.file_pairs)

        if self.n_clips is not None:
            self.file_pairs = self._sample_file_pairs(self.file_pairs)
            
        print(f"[INFO] DNS '{self.set}' set: {total_clips} total clips in folder, {len(self.file_pairs)} clips actually loaded")

    @staticmethod
    def _extract_file_id(path: Path) -> int:
        match = re.search(r"fileid_(\d+)$", path.stem)
        if not match:
            raise ValueError(f"Could not parse file ID from filename: {path.name}")
        return int(match.group(1))

    def _sample_file_pairs(self, file_pairs):
        if len(file_pairs) == 0:
            return file_pairs

        if isinstance(self.n_clips, int):
            n_selected = self.n_clips
        elif isinstance(self.n_clips, float):
            n_selected = floor(self.n_clips * len(file_pairs))
        else:
            raise TypeError(f"n_clips must be int, float, or None, was {type(self.n_clips)}")

        if n_selected <= 0:
            raise ValueError(f"n_clips selects 0 samples for set '{self.set}'. Received n_clips={self.n_clips}")
        if n_selected > len(file_pairs):
            raise ValueError(
                f"Requested n_clips={n_selected} exceeds available files in set '{self.set}' ({len(file_pairs)})"
            )

        rng = random.Random(self.random_seed)
        selected_indices = sorted(rng.sample(range(len(file_pairs)), n_selected))
        return [file_pairs[i] for i in selected_indices]

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.file_pairs[idx]

        if self.preproc_lib == "librosa":
            noisy_wave, _ = librosa.load(path=noisy_path, sr=self.sample_rate)
            clean_wave, _ = librosa.load(path=clean_path, sr=self.sample_rate)
        else:
            noisy_wave, noisy_sr = torchaudio.load(uri=noisy_path)
            clean_wave, clean_sr = torchaudio.load(uri=clean_path)
            if noisy_sr != self.sample_rate:
                noisy_wave = torchaudio.functional.resample(noisy_wave, noisy_sr, self.sample_rate)
            if clean_sr != self.sample_rate:
                clean_wave = torchaudio.functional.resample(clean_wave, clean_sr, self.sample_rate)
            if self.device != "cpu":
                noisy_wave = noisy_wave.to(self.device)
                clean_wave = clean_wave.to(self.device)

        if isinstance(self.input_pipeline, list):
            preproc_input = [pipe(noisy_wave) for pipe in self.input_pipeline]
        else:
            preproc_input = self.input_pipeline(noisy_wave)

        if self.target_pipeline is None:
            return preproc_input

        preproc_target = self.target_pipeline(clean_wave)
        try:
            preproc_target = np.copy(preproc_target)
        except Exception:
            pass

        return preproc_input, preproc_target
