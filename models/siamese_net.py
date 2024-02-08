import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, RocCurveDisplay
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryPrecision
from tqdm import tqdm

from config import CONFIG
from models.triplet_net import get_embedding_model
from preprocessing.feature_extraction import CustomPipeline


activations = {}
np.random.seed(42)


class SiameseNet(nn.Module):
    def __init__(self, embedding_model: nn.Module, out_features: int = 64):
        """
        Initializes a Siamese Network.
        
        Parameters
        ----------
        embedding_model: nn.Module
            Embedding model of the network.
        out_features: int
            Number of output features for the first fully connected layer (default: 256).
        """
        super(SiameseNet, self).__init__()
        self.embedding_model = embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss = nn.BCELoss()
        self.fc = nn.Sequential(
            nn.LazyLinear(out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_features, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=16, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, y1: torch.Tensor, y2: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Performs inference.
        
        Parameters
        ----------
        y1: torch.Tensor
            Batched anchor inputs.
        y2: torch.Tensor
            Batched query inputs.
        labels: torch.Tensor
            Ground truth value, 1 if y1 and y2 have same label, 1 otherwise. It is None if model is in eval() mode (default: None).
        """
        y1 = self.embedding_model(y1)
        y2 = self.embedding_model(y2)
        concatenated = torch.cat((y1, y2), dim=1)
        y = self.fc(concatenated)
        y = self.sigmoid(y)
        bce_loss = None
        if labels is not None:
            bce_loss = self.loss(y.squeeze(), labels)
        return y, bce_loss
    

class SiameseDataset(Dataset):
    def __init__(self, filepath: str, pipeline: CustomPipeline = None):
        """
        Creates a datasets providing triplets of images to train triplet net.

        Parameters
        ----------
        filepath: str
            Path to the serialized dataset containing time-domain signals.
        pipeline: CustomPipeline
            Pipeline to extract features from time-domain signals (default: None).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("Cannot find serialized training set. Please generate it first.")
        self.add_noise = True if CONFIG.get("PREPROCESSING", "add_gaussian_noise") == "true" else False
        self.dataset = pd.read_feather(filepath)
        self.classes = self.dataset["Class"].unique() if "Class" in self.dataset.columns \
            else self.dataset["Filepath"].unique()
        self.class_ids = {}
        self.n_samples = 22050      # to use as a threshold for cutting a signal
        self.pipeline = pipeline
        self.__generate_class_ids()

    def __getitem__(self, indexes):
        anchor = torch.Tensor([])
        query = torch.Tensor([])
        a_label_ids = []
        q_label_ids = []
        a_labels = []
        q_labels = []
        for idxs_ in indexes:
            anchor_example, query_example = self.dataset.iloc[idxs_].values
            a_label, a_sr, a_signal = anchor_example
            q_label, q_sr, q_signal = query_example

            a_signal = self._cut_signal_if_needed(a_signal[0])
            q_signal = self._cut_signal_if_needed(q_signal[0])
            a_signal = self._pad_signal_if_needed(a_signal)
            q_signal = self._pad_signal_if_needed(q_signal)

            if self.pipeline:
                a_img = self.pipeline.forward(a_signal).unsqueeze(0)
                anchor = a_img if anchor.size(0) == 0 \
                    else torch.row_stack((anchor, a_img))
                
                q_img = self.pipeline.forward(q_signal).unsqueeze(0)
                query = q_img if query.size(0) == 0 \
                    else torch.row_stack((query, q_img))
                
            else:
                anchor = a_signal if anchor.size(0) == 0 \
                    else torch.row_stack((anchor, a_signal))
                query = q_signal if query.size(0) == 0 \
                    else torch.row_stack((query, q_signal))

            a_label_ids.append(self.class_ids[a_label])
            q_label_ids.append(self.class_ids[q_label])

            a_labels.append(a_label)
            q_labels.append(q_label)
        
        return (anchor, query), (a_label_ids, q_label_ids), (a_labels, q_labels)

    def __len__(self):
        return self.dataset["Class"].size if "Class" in self.dataset.columns \
            else self.dataset["Filepath"].size

    def __generate_class_ids(self):
        for cid, c in enumerate(np.sort(self.classes)):
            self.class_ids[c] = cid

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
                padding = (diff_len//2, diff_len - (diff_len//2))
                signal = torch.nn.functional.pad(signal, padding)
        return signal

class BalancedSampler(BatchSampler):
    def __init__(self, labels: pd.Series, n_classes: int = 16, n_samples: int = 4, debug: bool = False):
        """
        Initializes a Balanced BatchSampler object.

        Parameters
        ----------
        labels: pd.Series
            Series of the labels in the dataset.
        n_classes: int
            Number of anchor classes to sample for generating the batch (default: 32).
        n_samples: int
            Number of samples per class to consider for generating the batch (default: 4).
        """
        self.batch_size = n_classes * n_samples
        self.counter = 0
        self.class_ids = {}
        self.class_weights = {}
        self.labels = labels
        self.labels_set = set(labels.values)
        self.n_classes = n_classes
        self.n_dataset = labels.count()
        self.n_samples = n_samples
        self.__init_class_weights()
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __iter__(self):
        while self.counter + self.batch_size < self.n_dataset:
            # sample n_classes at random without replacement
            sample_classes = np.random.choice(list(self.labels_set.difference(["NEG"])), self.n_classes, replace=False)
            if self.debug:
                print("Sampled classes: ", sample_classes)
            n_pos = int(np.ceil(self.n_samples/2))
            pairs =  torch.Tensor([])

            for c in sample_classes:
                if self.debug:
                    print("\nClass: ", c)
                # for each sampled class, get the corresponding indexes in the dataset
                c_idxs = self.labels[self.labels == c].dropna().index.values
                # sample n_samples indexes
                if c_idxs.shape[0] >= self.n_samples:
                    c_idxs_sample = np.random.choice(c_idxs, self.n_samples, replace=False)
                else:
                    c_idxs_sample = np.random.choice(c_idxs, self.n_samples, replace=True)
                    # if a class has less than n_pos examples, then reduce number of positive samples in batch pairs
                    n_pos = min(n_pos, c_idxs.shape[0])
                
                q_pos_idxs = np.random.choice(c_idxs_sample, n_pos, replace=False)
                q_neg_idxs = np.random.choice(self.labels[self.labels != c].dropna().index.values, (self.n_samples - n_pos - 1), replace=False)
                q_neg_idxs = np.concatenate((q_neg_idxs, np.random.choice(self.labels[self.labels == "NEG"].dropna().index.values, 1)))
                q_idxs = np.concatenate((q_pos_idxs, q_neg_idxs))
                '''
                if self.debug:
                    print("c_idxs_sample: ", c_idxs_sample)
                    print("n_pos: ", n_pos)
                    print("Pos idxs:\n", q_pos_idxs, "\nNeg idxs:\n", q_neg_idxs)
                    print(f"c_idxs_sample shape: {c_idxs_sample.shape} \t q_idxs shape: {q_idxs.shape}")
                    print("q_idxs: ", q_idxs)
                '''
                idxs = np.column_stack((c_idxs_sample, q_idxs))
                if self.debug:
                    # print(f"Pair indexes for class {c}:\n", idxs)
                    print("n_pos: ", n_pos)
                    pair_labels = []
                    for idx in idxs:
                        pair_labels.append([self.labels.iloc[idx[0]],
                                            self.labels.iloc[idx[1]]])
                    print("Pair labels:\n", pair_labels)
                        
                idxs = torch.Tensor(idxs)
                pairs = idxs if pairs.size(0) == 0 else torch.concat((pairs, idxs))
            yield pairs
            self.counter += self.batch_size
        self.counter = 0

    def __len__(self):
        return self.n_dataset // self.batch_size

    def __init_class_weights(self):
        for c in self.labels_set:
            count = self.labels.where(self.labels == c).dropna().count()
            self.class_weights[c] = 1/count


def get_dataloader(split: str) -> DataLoader:
    """
    Returns the dataloader for the desired dataset split.
    
    Parameters
    ----------
    split: str
        Split of the dataset for which to get the dataloader.

    Returns
    -------
    dataloader: DataLoader
        DataLoader object.

    Raises
    ------
    ValueError: if the split is incorrect.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fix some parameters for the audio features extraction
    N_FFT = CONFIG.getint("PREPROCESSING", "n_fft")
    N_MELS = CONFIG.getint("PREPROCESSING", "n_mels")
    RESAMPLE_FREQ = CONFIG.getint("PREPROCESSING", "resampling_frequency")
    n_classes = 40
    n_samples = 6
    
    if split.lower() in ["train", "training"]:
        augment = True
        filepath = CONFIG.get("DATASETS", "training_feather")

    elif split.lower() in ["val", "val_train"]:
        augment = False
        filepath = CONFIG.get("DATASETS", "val_train_feather")

    elif split.lower() in ["val", "validation"]:
        augment = False
        filepath = CONFIG.get("DATASETS", "validation_feather")
        n_classes = 5
        n_samples = 5

    else:
        raise ValueError("Incorrect split. Please see documentation for the get_dataloader() function.")
    
    # Define pipeline, dataset and dataloader
    logmel = False if CONFIG.get("PREPROCESSING", "features") == "pcen" else True
    pipeline = CustomPipeline(resample_freq=RESAMPLE_FREQ, n_fft=N_FFT,
                              n_mel=N_MELS, augment=augment, logmel=logmel)
    dataset = SiameseDataset(filepath=filepath, pipeline=pipeline)
    labels = dataset.dataset["Class"] if "Class" in dataset.dataset.columns \
        else dataset.dataset["Filepath"]
    sampler = BalancedSampler(labels=labels, n_classes=n_classes, n_samples=n_samples, debug=False)
    if device == "cuda":
        dataloader = DataLoader(dataset=dataset, sampler=sampler, pin_memory=True, pin_memory_device=device, num_workers=2, prefetch_factor=2)
    else:
        dataloader = DataLoader(dataset=dataset, sampler=sampler, num_workers=4, prefetch_factor=3)
    
    return dataloader


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def hook_fn(m, i, o):
    """
    Registers an hook function.
    """
    global activations
    activations[m] = o


def load_params_from_trial() -> dict:
    """
    Loads parameters learned during model selection using Triplet Margin Loss.
    
    Returns
    -------
    params: dict
        Parameters learned during model selection.
    """
    try:
        fpath = os.path.join("tnet_trials", CONFIG.get("OPTIMIZATION", "tnet_trial"), "params.json")
        with open(fpath, "r") as f:
            params = json.load(f)  # ["config"]
        return params
    except Exception as ex:
        print(ex)

def register_hooks(model: SiameseNet):
    """
    Register hooks for each layer of the network.
    
    Parameters
    ----------
    model: SiameseNet
        Model for which to retrieve the layers.
    
    """
    for name, layer in model._modules.items():
        if isinstance(layer, nn.Sequential):
            register_hooks(layer)
        else:
            layer.register_forward_hook(hook_fn)


def load_model(load_finetuned: bool = False) -> SiameseNet:
    """
    Loads the model from the disk.

    Parameters
    ----------
    load_finetuned: bool
        Specifies whether to load finetuned model (default: False).

    Returns
    -------
    model: SiameseNet
        Trained SiameseNet object.

    Raises
    ------
    FileNotFoundError: whenever the saved model cannot be found.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "siamese_network_finetuned.pt" if load_finetuned else "siamese_network.pt"
    model_path = os.path.join(CONFIG.get("MODELS", "save_path"), model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Cannot find saved model.")
    model = torch.load(model_path, map_location=torch.device(device))
    return model


def save_model(epoch: int, model: SiameseNet, optimizer: torch.optim.Optimizer, filename: str):
    """
    Saves the model on disk.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, filename)


def finetune_snet(split: str) -> SiameseNet:
    """
    Performs finetuning on the given dataset split ('validation' or 'evaluation').
    
    Parameters
    ----------
    split: str
        Split of the dataset on which to finetune the model ('validation' or 'evaluation').
    
    Returns
    -------
    results: pd.DataFrame
        Dataframe containing the metrics of the finetuning.
    
    Raises
    ------
    ValueError: if the split is not correct.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if split not in ["validation", "evaluation"]:
        raise ValueError(f"Split '{split}' is not available. Please choose between 'validation' and 'evaluation'.")
    embedding_model = get_embedding_model()
    if CONFIG.get("MODELS", "freeze_embedding_module") == "true":
        print("Freezing embedding module parameters...")
        for _, param in embedding_model.named_parameters():
            param.requires_grad = False
    model = SiameseNet(embedding_model=embedding_model)
    checkpoint = load_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    weight_decay = load_params_from_trial()["weight_decay"]
    best_loss = None
    model_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.NAdam(params=model_params, lr=1e-5, weight_decay=weight_decay)
    # Parameters to define early stopping
    delta_loss = CONFIG.getfloat("TRAINING", "delta_loss")
    original_patience = 10
    patience = original_patience
    dataloader = get_dataloader(split=split)
    epochs = 20

    save_path = CONFIG.get("MODELS", "save_path")
    hist = pd.DataFrame([], columns=["Epoch", "Finetuning loss", "Finetuning accuracy",
                                    "Finetuning F1 score", "Finetuning precision"])
    model = model.train()
    for epoch in tqdm(range(epochs), desc=f"Finetuning on {split}",
                      total=epochs, leave=False):
        avg_loss = 0
        acc = BinaryAccuracy()
        f1 = BinaryF1Score()
        prec = BinaryPrecision()
        for images, label_ids, _ in tqdm(iter(dataloader), desc="Finetuning",
                                              total=len(dataloader), leave=False):
            optimizer.zero_grad()
            anchor, query = images
            anchor = anchor[0].to(device)
            query = query[0].to(device)
            anchor_labels, query_labels = label_ids
            targets = torch.zeros(len(anchor_labels))
            idxs = torch.where(torch.Tensor(anchor_labels) == torch.Tensor(query_labels))
            targets[idxs] = 1.
            targets = targets.to(device)
            predictions, loss = model(anchor, query, targets)
            loss.backward()
            optimizer.step()
            acc.update(predictions.squeeze(), targets)
            f1.update(predictions.squeeze(), targets)
            prec.update(predictions.squeeze(), targets)
            avg_loss += loss.item()
        avg_loss /= len(dataloader)
        print(f"\nFinetuning stats for epoch {epoch+1}:\n")
        print(f"\tAverage loss: {avg_loss}\n" \
              f"\tAccuracy: {acc.compute().item()}\n" \
              f"\tF1 score: {f1.compute().item()}\n" \
              f"\tPrecision: {prec.compute().item()}\n\n")
        if best_loss is None or avg_loss <= (best_loss - delta_loss):
            best_loss = avg_loss
            patience = original_patience
            save_model(epoch=epoch, model=model, optimizer=optimizer,
                       filename=os.path.join(save_path, "siamese_network_finetuned.pt"))
        else:
            patience -= 1
        hist = pd.concat([hist, pd.DataFrame([[epoch, avg_loss, acc.compute().item(),
                                               f1.compute().item(),
                                               prec.compute().item()]],
                                               columns=hist.columns)])
    hist.to_csv(os.path.join(save_path, "snet_finetuning_history.csv"), index=False,
                             columns=hist.columns)
    

def eval_snet():
    """
    Evaluates Siamese Network on Validation set.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = get_embedding_model()
    model = SiameseNet(embedding_model=embedding_model).to(device)
    checkpoint = load_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    val_dataloader = get_dataloader(split="val_train")
    val_acc = BinaryAccuracy()
    val_f1 = BinaryF1Score()
    val_prec = BinaryPrecision()
    avg_val_loss = 0
    model = model.eval()
    with torch.no_grad():
        for images, label_ids, _ in tqdm(iter(val_dataloader), desc="Processing validation batches",
                                            total=len(val_dataloader), leave=False):
            anchor, query = images
            anchor = anchor[0].to(device)
            query = query[0].to(device)
            anchor_labels, query_labels = label_ids
            targets = torch.zeros(len(anchor_labels))
            idxs = torch.where(torch.Tensor(anchor_labels) == torch.Tensor(query_labels))
            targets[idxs] = 1.
            targets = targets.to(device)
            predictions, loss = model(anchor, query, targets)
            val_acc.update(predictions.squeeze(), targets)
            val_f1.update(predictions.squeeze(), targets)
            val_prec.update(predictions.squeeze(), targets)
            avg_val_loss += loss.item()
        avg_val_loss /= len(val_dataloader)
        display = RocCurveDisplay.from_predictions(targets.to("cpu").numpy(), predictions.to("cpu").numpy())
        # display.plot()
        plt.title("ROC curve for Siamese Network")
        plt.show()
        print(f"\tAverage loss: {avg_val_loss}\n" \
                f"\tAccuracy: {val_acc.compute().item()}\n" \
                f"\tF1 score: {val_f1.compute().item()}\n" \
                f"\tPrecision: {val_prec.compute().item()}\n\n")


def train_siamese_net():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = get_embedding_model()
    if CONFIG.get("MODELS", "freeze_embedding_module") == "true":
        print("Freezing embedding module parameters...")
        for _, param in embedding_model.named_parameters():
            param.requires_grad = False
            
    model = SiameseNet(embedding_model=embedding_model).to(device)
    weight_decay = load_params_from_trial()["weight_decay"]
    best_loss = None
    model_params = [param for param in model.parameters() if param.requires_grad == True]
    # optimizer = torch.optim.SGD(params=model_params, lr=1e-3, weight_decay=weight_decay,
                                # nesterov=True, momentum=0.9, dampening=0)
    optimizer = torch.optim.NAdam(params=model_params, lr=1e-3, weight_decay=weight_decay)
    
    epochs = 100  # CONFIG.getint("TRAINING", "epochs")

    # Parameters to define early stopping
    delta_loss = CONFIG.getfloat("TRAINING", "delta_loss")
    original_patience = 30  # CONFIG.getint("TRAINING", "patience")
    patience = original_patience

    # savepath and history
    save_path = CONFIG.get("MODELS", "save_path")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hist = pd.DataFrame([], columns=["Epoch", "Training loss", "Validation loss",
                                     "Training accuracy", "Validation accuracy",
                                     "Training F1 score", "Validation F1 score",
                                     "Training precision", "Validation precision"])
    
    # define dataloader
    train_dataloader = get_dataloader(split="train")
    val_dataloader = get_dataloader(split="val_train")

    for epoch in tqdm(range(epochs), desc="Starting epoch\n", total=epochs):
        if patience == 0:
            print("\nEarly stopping..")
            break
        avg_train_loss = 0
        avg_val_loss = 0
        
        # Training part
        train_acc = BinaryAccuracy()
        train_f1 = BinaryF1Score()
        train_prec = BinaryPrecision()
        model = model.train()
        for images, label_ids, _ in tqdm(iter(train_dataloader), desc="Processing batches",
                                              total=len(train_dataloader), leave=False):
            optimizer.zero_grad()
            anchor, query = images
            anchor = anchor[0].to(device)
            query = query[0].to(device)
            anchor_labels, query_labels = label_ids
            targets = torch.zeros(len(anchor_labels))
            idxs = torch.where(torch.Tensor(anchor_labels) == torch.Tensor(query_labels))
            targets[idxs] = 1.
            targets = targets.to(device)
            predictions, loss = model(anchor, query, targets)
            loss.backward()
            optimizer.step()
            train_acc.update(predictions.squeeze(), targets)
            train_f1.update(predictions.squeeze(), targets)
            train_prec.update(predictions.squeeze(), targets)
            avg_train_loss += loss.item()
        avg_train_loss /= len(train_dataloader)
        print(f"\nTraining stats for epoch {epoch+1}:\n")
        print(f"\tAverage loss: {avg_train_loss}\n" \
              f"\tAccuracy: {train_acc.compute().item()}\n" \
              f"\tF1 score: {train_f1.compute().item()}\n" \
              f"\tPrecision: {train_prec.compute().item()}\n\n")

        # Validation part
        val_acc = BinaryAccuracy()
        val_f1 = BinaryF1Score()
        val_prec = BinaryPrecision()
        model = model.eval()
        with torch.no_grad():
            for images, label_ids, _ in tqdm(iter(val_dataloader), desc="Processing validation batches",
                                             total=len(val_dataloader), leave=False):
                anchor, query = images
                anchor = anchor[0].to(device)
                query = query[0].to(device)
                anchor_labels, query_labels = label_ids
                targets = torch.zeros(len(anchor_labels))
                idxs = torch.where(torch.Tensor(anchor_labels) == torch.Tensor(query_labels))
                targets[idxs] = 1.
                targets = targets.to(device)
                predictions, loss = model(anchor, query, targets)
                val_acc.update(predictions.squeeze(), targets)
                val_f1.update(predictions.squeeze(), targets)
                val_prec.update(predictions.squeeze(), targets)
                avg_val_loss += loss.item()
        avg_val_loss /= len(val_dataloader)
        # print(f"Average validation loss for epoch {epoch+1}: {avg_val_loss}\n---\n")
        print(f"\nValidation stats for epoch {epoch+1}:\n")
        print(f"\tAverage loss: {avg_val_loss}\n" \
              f"\tAccuracy: {val_acc.compute().item()}\n" \
              f"\tF1 score: {val_f1.compute().item()}\n" \
              f"\tPrecision: {val_prec.compute().item()}\n\n")
        if best_loss is None or avg_val_loss <= (best_loss - delta_loss):
            best_loss = avg_val_loss
            patience = original_patience
            save_model(epoch=epoch, model=model, optimizer=optimizer,
                       filename=os.path.join(save_path, "siamese_network.pt"))
        else:
            patience -= 1

        hist = pd.concat([hist, pd.DataFrame([[epoch, avg_train_loss, avg_val_loss,
                                               train_acc.compute().item(), val_acc.compute().item(),
                                               train_f1.compute().item(), val_f1.compute().item(),
                                               train_prec.compute().item(), train_prec.compute().item()]], columns=hist.columns)])

    hist.to_csv(os.path.join(save_path, "snet_training_history.csv"), index=False,
                            columns=hist.columns)
    return hist
    