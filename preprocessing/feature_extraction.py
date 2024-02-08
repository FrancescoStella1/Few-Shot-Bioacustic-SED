import glob
import os
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
import torchaudio
from torchaudio import transforms
from tqdm import tqdm

from config import CONFIG


class CustomPipeline(torch.nn.Module):
    def __init__(self, resample_freq: int, n_fft: int, n_mel: int, stretch_factor: int = 3, augment: bool = True,
                 logmel: bool = False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.spectrogram = transforms.Spectrogram(n_fft=n_fft, power=1, hop_length=n_fft//4)
        self.augment = augment
        if augment:
            self.spectrogram_aug = torch.nn.Sequential(
                # transforms.TimeStretch(stretch_factor, fixed_rate=True),
                transforms.FrequencyMasking(freq_mask_param=40),
                transforms.TimeMasking(time_mask_param=2)
            )
        self.mel_scale = transforms.MelScale(
            n_mels=n_mel,
            sample_rate=resample_freq,
            n_stft=(n_fft // 2) + 1
        )
        self.to_db = transforms.AmplitudeToDB(
            stype="amplitude",
            top_db=80
        )
        self.logmel = logmel

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        spectrogram = self.spectrogram(signal)
        if self.augment:
            spectrogram = self.spectrogram_aug(spectrogram)
        spectrogram = self.mel_scale(spectrogram)
        if CONFIG.get("PREPROCESSING", "features") == "logmel":
            spectrogram = self.to_db(spectrogram)
            return torch.Tensor(spectrogram)
        else:
            spectrogram = spectrogram.to("cpu").numpy()
            pcen = librosa.pcen(spectrogram * (2**31))
            return torch.Tensor(pcen)
        
        
class CustomDataset(Dataset):
    def __init__(self, split: str, n_samples: int = 22050, pipeline: CustomPipeline = None, include_neg_class: bool = False):
        self.split = split.lower()
        self.add_noise = True if CONFIG.get("PREPROCESSING", "add_gaussian_noise") == "true" else False
        self.annotations = None
        self.class_weights = {}
        self.dataset = None
        self.val_dataset = None
        self.n_samples = n_samples  #  // 2
        self.label_ids = {}
        self.pipeline = pipeline
        self.include_neg = include_neg_class
        self._get_annotations()
        self._generate_dataset()

    def __len__(self):
        return self.dataset[self.dataset.columns[0]].size

    def __getitem__(self, index: int):
        label, sr, time_signal = self.dataset.iloc[index]
        label_id = self.label_ids[label]
        # class_weight = self.get_class_weight(label).values[0]
        sr = torch.Tensor([sr])
        # copy numpy array to make it writable and supported by torch
        signal = time_signal[0].copy()
        signal = self._cut_signal_if_needed(signal)
        signal = self._pad_signal_if_needed(signal)
        if self.pipeline:
            signal = self.pipeline(signal)
        return signal, sr, label, label_id

    def _cut_signal_if_needed(self, signal: np.ndarray) -> torch.Tensor:
        signal = signal.copy()
        signal = np.expand_dims(signal, axis=0)
        if signal.shape[1] > self.n_samples:
            signal = signal[:, :self.n_samples]
        return torch.Tensor(signal)

    def _pad_signal_if_needed(self, signal: torch.Tensor) -> torch.Tensor:
        signal_len = signal.shape[1]
        if signal_len < self.n_samples:
            diff_len = self.n_samples - signal_len
            if self.add_noise:
                lpad = torch.zeros(diff_len//2)
                lpad = lpad + ((0.1**0.5) * torch.randn(diff_len//2))
                rpad = torch.zeros(diff_len - (diff_len//2))
                rpad = rpad + ((0.1**0.5) * torch.randn(diff_len-(diff_len//2)))
                signal = torch.concat((lpad, signal.squeeze()))
                signal = torch.concat((signal, rpad)).unsqueeze(0)
            else:
                padding = (diff_len//2, diff_len - (diff_len//2))  # (0, diff_len)
                signal = torch.nn.functional.pad(signal, padding)
        return signal

    def _get_annotations(self):
        if self.split in ["train", "training"]:
            self.split = "training"
            self.annotations = glob.glob("dataset/Training_Set/**/*.csv", recursive=True)
        elif self.split in ["val_train", "val_training"]:
            self.split = "val_training"
        elif self.split in ["val", "validation"]:
            self.split = "validation"
            self.annotations = glob.glob("dataset/Validation_Set/**/*.csv", recursive=True)

    def _generate_dataset(self):
        if self.split == "training":
            filepath = CONFIG.get("DATASETS", "training_feather")
            val_filepath = CONFIG.get("DATASETS", "val_train_feather")
        if self.split == "val_training":
            filepath = CONFIG.get("DATASETS", "val_train_feather")
        elif self.split == "validation":
            filepath = CONFIG.get("DATASETS", "validation_feather")
        elif self.split == "testing":
            filepath = CONFIG.get("DATASETS", "testing_feather")
        path1, path2 = filepath.split("/")[:2]
        datapath = os.path.join(path1, path2)
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        if os.path.exists(filepath):
            self.dataset = pd.read_feather(filepath)
            self._generate_class_ids_and_weights()
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
            if self.split == "training":
                self.val_dataset = pd.read_feather(val_filepath)
                self.val_dataset = self.val_dataset.sample(frac=1).reset_index(drop=True)
            return

        if self.split == "training":
            self.dataset = pd.DataFrame([], columns=["Class", "Sampling rate", "Time signal"])
            self.val_dataset = pd.DataFrame([], columns=["Class", "Sampling rate", "Time signal"])
            # self._generate_class_ids_and_weights()
        elif self.split == "val_training":
            self.dataset = pd.DataFrame([], columns=["Class", "Sampling rate", "Time signal"])
        else:
            self.dataset = pd.DataFrame([], columns=["Filepath", "Sampling rate", "Time signal"])

        # Define a threshold to create a training and a validation split from the dataset
        threshold = 0.7
        resampling_freq = CONFIG.getint("PREPROCESSING", "resampling_frequency")
        for ann in tqdm(self.annotations, desc="Generating datasets", total=len(self.annotations)):
            df = pd.read_csv(ann)
            audiofile = ann[:-4] + ".wav"
            num_channels = torchaudio.info(audiofile).num_channels
            y, sr = torchaudio.load(audiofile)
            if num_channels == 1:
                y = y[0]
            elif num_channels == 2:
                # compute mean values and convert to single-channel
                y = torch.mean(y, dim=0)
            
            # Normalize audiofile
            y = (y - y.mean()) / y.std()

            for c in df.columns[3:]:
                if self.split == "training" or self.split == "val_training":
                    pos_idxs = df[c].where(df[c] == "POS").dropna().index
                else:
                    pos_idxs = df[c].where(df[c] == "POS").dropna().head(5).index
                if pos_idxs.size == 0:
                    continue
                times = df.loc[pos_idxs.values][["Starttime", "Endtime"]]
                min_duration = times.apply(lambda x: x.iloc[1] - x.iloc[0], axis=1)
                min_duration = min_duration[min_duration > 1e-2].min()
                times *= sr
                min_duration *= sr
                first_col_value = c if self.split == "training" else audiofile

                for idx, row in enumerate(times.iterrows()):
                    start_frame = int(row[1].iloc[0])
                    end_frame = int(row[1].iloc[1])
                    time_signal = y[start_frame:end_frame]
                    if time_signal.shape[0] == 0:
                        continue
                    time_signal = torchaudio.functional.resample(time_signal, orig_freq=sr,
                                                                 new_freq=resampling_freq)
                    time_signal = time_signal.numpy()
                    tmp_df = pd.DataFrame([[first_col_value, resampling_freq, [time_signal]]],
                                          columns=self.dataset.columns)
                    if self.split == "training":
                        if idx <= times.shape[0] * threshold:
                            self.dataset = pd.concat([self.dataset, tmp_df])
                        else:
                            self.val_dataset = pd.concat([self.val_dataset, tmp_df])
                    else:
                        self.dataset = pd.concat([self.dataset, tmp_df])
                if self.include_neg:
                    # Add random negative images
                    # if self.split == "training":
                    end_times = []
                    for idx, row in df[df.columns[1:3]].iterrows():
                        if idx > 0:
                            if row.iloc[0] - endtime >= 2.5:
                                end_times.append(row.iloc[0])
                        endtime = row.iloc[1]
                    if len(end_times) < 3:
                        continue
                    end_times = np.random.choice(end_times, 3, replace=False)
                    # Add some margin before end_time
                    end_times -= .5
                    # Multiply by original sample rate
                    end_times *= sr
                    for idx, end_time in enumerate(end_times):
                        lb = min(sr, min_duration)
                        start_frame = int(end_time-lb)
                        end_frame = int(end_time)
                        time_signal = y[start_frame:end_frame]
                        assert time_signal.shape[0] > 0
                        time_signal = torchaudio.functional.resample(time_signal, orig_freq=sr,
                                                                    new_freq=resampling_freq)
                        time_signal = time_signal.numpy()
                        tmp_df = pd.DataFrame([["NEG", resampling_freq, [time_signal]]],
                                            columns=self.dataset.columns)
                        if idx <= 1:
                            self.dataset = pd.concat([self.dataset, tmp_df])
                        else:
                            self.val_dataset = pd.concat([self.val_dataset, tmp_df])

        self.dataset.reset_index(inplace=True, drop=True)
        self.dataset.to_feather(filepath)
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        if self.split == "training":
            self._generate_class_ids_and_weights()
            self.val_dataset.reset_index(inplace=True, drop=True)
            self.val_dataset.to_feather(val_filepath)
            self.val_dataset = self.val_dataset.sample(frac=1).reset_index(drop=True)

    def _generate_class_ids_and_weights(self):
        """
        Generates class IDs for the classes contained in the dataset and assigns weights
        in order to manage class imbalance problems.
        """
        class_weights_path = CONFIG.get("TRAINING", "class_weights_path")
        class_weights_file = os.path.join(class_weights_path, "class_weights.pt")
        classes = self.dataset["Class"].sort_values().unique()
        n_classes = self.dataset["Class"].nunique()
        tot_examples = self.dataset["Class"].count()
        encodings = one_hot(torch.arange(0, n_classes, step=1))
        if os.path.exists(class_weights_file):
            self.class_weights = torch.load(class_weights_file)
            for idx, c in enumerate(classes):
                class_id = encodings[idx].float()  # .type(torch.IntTensor)
                self.label_ids[c] = class_id
            return

        for idx, c in enumerate(classes):
            class_id = encodings[idx].float()  # .type(torch.IntTensor)
            self.label_ids[c] = class_id
            c_examples = self.dataset[self.dataset["Class"] == c]["Class"].count()
            self.class_weights[class_id] = (tot_examples - c_examples)/tot_examples
        class_weights_path = CONFIG.get("TRAINING", "class_weights_path")
        if not os.path.exists(class_weights_path):
            os.makedirs(class_weights_path)
            torch.save(self.class_weights, class_weights_file)

    def get_class(self, label_id: torch.Tensor) -> str:
        """
        Returns the class given its one-hot encoding.

        Parameters
        ----------
        label_id: torch.Tensor
            ID of the label associated to the class of interest.

        Returns
        -------
        class: str
            Class name corresponding to the encoding (or None).
        """
        for k, v in self.label_ids.items():
            if torch.equal(v, label_id):
                return k

    def get_class_weight(self, class_id: float) -> float:
        """
        Returns the weight of a class.

        Parameters
        ----------
        class_id: float
            ID of the class of interest.

        Returns
        -------
        weight: float
            Weight of the class.
        """
        try:
            return self.class_weights[class_id]

        except Exception:
            raise KeyError("Classname not present in the weights dictionary.")


def get_dataloader(split: str) -> DataLoader:
    """
    Creates the dataloader for the CustomDataset object, using an instance of the CustomPipeline class.
    
    Parameters
    ----------
    split: str
        Split of the dataset for which to create the dataloader.

    Returns
    -------
    dataloader: DataLoader
        Dataloader for the split of the dataset.
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # Fix some parameters for the audio features extraction
    N_FFT = CONFIG.getint("PREPROCESSING", "n_fft")
    N_MELS = CONFIG.getint("PREPROCESSING", "n_mels")
    RESAMPLE_FREQ = CONFIG.getint("PREPROCESSING", "resampling_frequency")
    logmel = False if CONFIG.get("PREPROCESSING", "features") == "pcen" else True

    # Define dataset, pipeline and dataloader
    batch_size = 32 if split == "train" else 16
    augment = True if split == "train" else False
    pipeline = CustomPipeline(resample_freq=RESAMPLE_FREQ, n_fft=N_FFT,
                              n_mel=N_MELS, augment=augment, logmel=logmel)
    dataset = CustomDataset(split=split, pipeline=pipeline)
    if device == "cuda":
        dataloader = DataLoader(dataset=dataset, pin_memory=True, batch_size=batch_size, pin_memory_device=device,
                                shuffle=True)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader
