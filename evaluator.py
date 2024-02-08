import glob
import os
import platform
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio

from config import CONFIG
from preprocessing.feature_extraction import CustomPipeline
from visualization import plotting

LOGMEL = False if CONFIG.get("PREPROCESSING", "features") == "pcen" else True


class Evaluator:
    def __init__(self, base_path: str, model: torch.nn.Module, n_samples: int = 22050,
                 generate_triplets: bool = False):
        """
        Instantiates an Evaluator object.

        Parameters
        ----------
        base_path: str
            Path to the root directory of the dataset.
        model: torch.nn.Module
            Model to evaluate.
        n_samples: int
            Number of samples to decide whether to cut or pad the time signal. By default, assuming that
            sampling rate is 22050, 1-second signal chunks are taken to compute PCEN (default: 22050).
        generate_triplets: bool
            Specifies whether to generate triplets during evaluation (default: False).
        """
        self.add_noise = True if CONFIG.get("PREPROCESSING", "add_gaussian_noise") == "true" else False
        self.annotations = pd.DataFrame([], columns=["Audiofilename", "Starttime", "Endtime"])
        self.audiofiles = glob.glob(f"{base_path}/**/*.csv", recursive=True)
        self.base_path = base_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.n_fft = CONFIG.getint("PREPROCESSING", "n_fft")
        self.n_mel = CONFIG.getint("PREPROCESSING", "n_mels")
        self.n_samples = n_samples
        self.support_set = None
        self.generate_triplets = generate_triplets
        try:
            # Support sets already contain only the first 5 shots of each class
            if "validation" in base_path.lower():
                self.support_set = pd.read_feather(CONFIG.get("DATASETS", "validation_feather"))
            elif "evaluation" in base_path.lower():
                self.support_set = pd.read_feather(CONFIG.get("DATASETS", "testing_feather"))
            if platform.system() == "Windows":
                self.support_set["Filepath"] = self.support_set["Filepath"].map(lambda f: f.replace("/", "\\"))
        except Exception as ex:
            raise FileNotFoundError("Dataset cannot be found. Please generate it first using the --generate_dataset "
                                    "option.")

    def _cut_signal_if_needed(self, signal: torch.Tensor) -> torch.Tensor:
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
    
    def __save_annotations(self):
        """
        Saves the annotations to the path specified in the configuration.
        """
        savepath = CONFIG.get("EVALUATION", "annotations_path")
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        split = "validation" if "validation" in self.base_path.lower() \
            else "evaluation"
        savepath = os.path.join(savepath, f"{split}_annotations.csv")
        self.annotations.to_csv(savepath, index=False, header=True)

    def evaluate_model(self, debugging=False):
        """
        Evaluates the model on the dataset specified during class instantiation.
        """
        folders = os.listdir(self.base_path)
        for idx, folder in tqdm(enumerate(folders), desc="Evaluating..", total=len(folders)):
            audiofiles = os.listdir(os.path.join(self.base_path, folder))
            for filename in audiofiles:
                if ".csv" in filename:
                    continue
                print(f"Processing: {filename}\n")
                img_generator = self.read_audiofile(folder=folder, filename=filename, hop_length=11025, debugging=debugging)
                pending_annotations = {
                    "starttime": None,
                    "endtime": None
                }
                for images, times in img_generator:
                    # if cnt >= 3:
                        # break
                    if self.generate_triplets:
                        prediction = self.model(images[0].unsqueeze(0).to(self.device),
                                                images[1].unsqueeze(0).to(self.device),
                                                images[2].unsqueeze(0).to(self.device))
                    else:
                        prediction = self.model(images[0].unsqueeze(0).to(self.device),
                                                images[1].unsqueeze(0).to(self.device))
                        if type(prediction) == tuple:
                            prediction = prediction[0]
                        if debugging:
                            print("Prediction: ", prediction.item(), "\n---\n\n")
                            plotting.visualize_pcen(image=images[1].squeeze().numpy(), title="Query PCEN")
                    
                    if prediction.item() >= .5:
                        if pending_annotations["starttime"] is not None:
                            pending_annotations["endtime"] = times[1]
                        else:
                            pending_annotations["starttime"] = times[0]
                            pending_annotations["endtime"] = times[1]
                    else:
                        if pending_annotations["starttime"] is not None:
                            self.annotations = pd.concat([self.annotations,
                                                        pd.DataFrame([[filename, pending_annotations["starttime"],
                                                                       pending_annotations["endtime"]]],
                                                                       columns=self.annotations.columns)])
                            pending_annotations["starttime"] = None
                            pending_annotations["endtime"] = None
                    # cnt += 1
        print("Annotations: \n", self.annotations["Audiofilename"].count())
        print(self.annotations)
        self.__save_annotations()

    def get_negative_image(self, folder: str, filename: str):
        """
        Returns a negative example to be used during evaluation of the model.

        Parameters
        ----------
        folder: str
            Name of the folder containing the audiofile.
        filename: str
            Name of the audiofile.

        Returns
        -------
        image: torch.Tensor
            Negative image.
        """
        time_signal = self.support_set[self.support_set["Filepath"] == "NEG"]["Time signal"].values[0].copy()[0]
        time_signal = self._cut_signal_if_needed(time_signal)
        time_signal = self._pad_signal_if_needed(time_signal)
        pipeline = CustomPipeline(resample_freq=22050, n_fft=self.n_fft, n_mel=self.n_mel, logmel=LOGMEL,
                                  augment=False)
        time_signal = torch.Tensor(time_signal)
        img = pipeline.forward(time_signal)
        return img  # .squeeze()

    def get_support_image(self, folder: str, filename: str, get_all: bool = False) -> Tuple:
        """
        Returns a random or all support images for the chosen audiofile.

        Parameters
        ----------
        folder: str
            Name of the folder containing the audiofile of interest.
        filename: str
            Name of the audiofile.
        get_all: bool
            Specifies whether to return all 5 shots of the support set (default: False).

        Returns
        -------
        image, duration: Tuple[torch.Tensor, float]
            Tuple containing a random or all anchor images (if get_all is True) and the minimum duration in frames of the events.
        """
        resampling_freq = CONFIG.getint("PREPROCESSING", "resampling_frequency")
        filepath = os.path.join(self.base_path, folder, filename)
        images = None
        pipeline = CustomPipeline(resample_freq=resampling_freq, n_fft=self.n_fft, n_mel=self.n_mel,
                                  logmel=LOGMEL, augment=False)

        if get_all:
            time_signals = self.support_set[self.support_set["Filepath"] == filepath]["Time signal"].values.copy()
            min_duration = np.inf
            for time_signal in time_signals:
                time_signal = time_signal[0]
                if time_signal.shape[0] < min_duration:
                    min_duration = time_signal.shape[0]
                time_signal = self._cut_signal_if_needed(time_signal)
                time_signal = self._pad_signal_if_needed(time_signal)
                time_signal = torch.Tensor(time_signal)
                img = pipeline.forward(time_signal)  # .squeeze()
                images = img if images is None else torch.row_stack((images, img))
            return images, min_duration
                
        else:
            time_signal = np.random.choice(self.support_set[self.support_set["Filepath"] == filepath]["Time signal"]
                                        .values)[0].copy()
            duration = time_signal.shape[0]
            time_signal = self._cut_signal_if_needed(time_signal)
            time_signal = self._pad_signal_if_needed(time_signal)
            time_signal = torch.Tensor(time_signal)  # .to(self.device)
            img = pipeline.forward(time_signal)
            return img, duration

    def read_audiofile(self, folder: str,  filename: str, hop_length: int = 5512,
                       n_fft: int = 1024, n_mels: int = 128, debugging: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Reads an audiofile in chunks, starting after the fifth shot of the class of interest.

        Parameters
        ----------
        folder: str
            Name of the folder containing the file to read.
        filename: str
            Name of the audiofile to read.
        hop_length: int
            Number of samples defining the sliding of the window (default: 5512).
        n_fft: int
            Specifies the number of bins in the STFT used to obtain the PCEN (default: 1024).
        n_mels: int
            Number of mel filterbanks (default: 128).
        debugging: bool
            Specifies whether debugging info must be printed on screen (default: False).

        Returns
        -------
        images, times: Tuple[torch.Tensor, float]
            Tuples representing extracted features and start/end times of the chunk.
        """
        filepath = os.path.join(self.base_path, folder, filename)
        annotations = filepath[:-4] + ".csv"
        annotations = pd.read_csv(annotations).sort_values("Starttime")
        shots = annotations[annotations["Q"] == "POS"].head(5)
        num_channels = torchaudio.info(filepath).num_channels
        offset = shots.iloc[-1]["Endtime"]
        y, sr = torchaudio.load(filepath)
        
        resampling_freq = CONFIG.getint("PREPROCESSING", "resampling_frequency")
        offset = round(offset * resampling_freq)

        if num_channels == 1:
            y = y[0]
        elif num_channels == 2:
            # compute mean values and convert to single-channel
            y = torch.mean(y, dim=0)

        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=resampling_freq)
        win_lb = offset

        pipeline = CustomPipeline(resample_freq=resampling_freq, logmel=LOGMEL,
                                  n_fft=n_fft, n_mel=n_mels, augment=False)
        get_all = False
        support_image, duration = self.get_support_image(folder=folder, filename=filename, get_all=get_all)
        if debugging:
            print(f"Minimum duration: {duration} frames = {duration/resampling_freq} seconds")
            for s_img in support_image:
                if LOGMEL:
                    plotting.visualize_logmel(image=s_img.numpy(), title="Support log-Mel spectrogram")
                else:
                    plotting.visualize_pcen(image=s_img.numpy(), title="Support PCEN")
        
        if get_all:
            # Take the mean of the support images
            support_image = torch.mean(support_image, dim=0).unsqueeze(0)

        if debugging:
            if LOGMEL:
                plotting.visualize_logmel(image=support_image.squeeze().numpy(), title="Mean Support log-Mel spectrogram")
            else:
                plotting.visualize_pcen(image=support_image.squeeze().numpy(), title="Mean Support PCEN")

        if self.generate_triplets:
            negative_image = self.get_negative_image(folder=folder, filename=filename)
        # print("Length of audiofile: ", y.shape[0])
            
        while win_lb < y.shape[0]:
            win_ub = win_lb + duration
            signal = self._cut_signal_if_needed(y[win_lb:win_ub])
            signal = self._pad_signal_if_needed(signal)
            img = pipeline.forward(signal)
            if debugging:
                print(f"[DEBUGGING INFO] - start time: {win_lb/resampling_freq}, end time: {win_ub/resampling_freq}, chunk_length: {win_ub-win_lb}")
            if self.generate_triplets:
                yield (support_image, img, negative_image), (win_lb/resampling_freq, win_ub/resampling_freq)
            else:
                yield (support_image, img), (win_lb/resampling_freq, win_ub/resampling_freq)
            win_lb += hop_length
