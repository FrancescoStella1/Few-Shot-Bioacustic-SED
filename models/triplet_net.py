import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from pytorch_metric_learning import distances, losses, miners, reducers
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from preprocessing.feature_extraction import CustomDataset, CustomPipeline


activations = {}
np.random.seed(42)


class Cnn(nn.Module):
    """CNN model intended for performing model selection and for later use in other DNNs."""
    def __init__(self, out_channels_1: int, out_channels_2: int, out_channels_3: int,
                 kernel_size_1: int, kernel_size_2: int, kernel_size_3: int, bn_momentum: float,
                 dropout_1: float, dropout_2: float, dropout_3: float):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels_1,
                kernel_size=kernel_size_1
            ),
            nn.BatchNorm2d(out_channels_1, momentum=bn_momentum),
            nn.PReLU(),
            nn.Dropout(dropout_1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_1,
                out_channels=out_channels_2,
                kernel_size=kernel_size_2
            ),
            nn.BatchNorm2d(out_channels_2, momentum=bn_momentum),
            nn.PReLU(),
            nn.Dropout(dropout_2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_2,
                out_channels=out_channels_3,
                kernel_size=kernel_size_3
            ),
            nn.BatchNorm2d(out_channels_3, momentum=bn_momentum),
            nn.PReLU(),
            nn.Dropout(dropout_3)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        # x = nn.functional.normalize(x, p=2, dim=1)
        return x
    

class TripletNet(nn.Module):
    def __init__(self, embedding_model: nn.Module):
        super(TripletNet, self).__init__()
        self.embedding_cnn = embedding_model
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        return self.embedding_cnn(x)


def get_embedding_model(flatten: bool = True) -> torch.nn.Module:
    """
    Returns the trained embedding model.
    
    Parameters
    ----------
    flatten: bool
        Whether to flatten the output embeddings (default: True).

    Returns
    -------
    model: torch.nn.Module
        Embedding model.
    """
    model, _ = create_model_from_trial()
    checkpoint = load_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model._modules["embedding_cnn"]
    model = nn.Sequential(*list(model.children()))
    if not flatten:
        model = model[:3]
    return model


def get_dataloader(split: str, batch_size: int = None) -> DataLoader:
    """
    Returns the dataloader for the desired dataset split.
    
    Parameters
    ----------
    split: str
        Split of the dataset for which to get the dataloader.
    batch_size: int
        Number of examples in a batch. If None, the value is retrieved from the config file (default: None).

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

    if split.lower() in ["train", "training"]:
        augment = True
        # filepath = filepath=CONFIG.get("DATASETS", "training_feather")
        # n_classes = CONFIG.getint("TRAINING", "n_classes")
        # n_samples = CONFIG.getint("TRAINING", "n_samples")

    elif split.lower() in ["val", "val_train", "validation"]:
        augment = False
        # filepath = filepath=CONFIG.get("DATASETS", "val_train_feather")
        # n_classes = CONFIG.getint("TRAINING", "val_n_classes")
        # n_samples = CONFIG.getint("TRAINING", "val_n_samples")

    else:
        raise ValueError("Incorrect split. Please see documentation for the get_dataloader() function.")
    
    if batch_size is None:
        batch_size = CONFIG.getint("TRAINING", "batch_size")
    # define pipeline, dataset and dataloader
    logmel = False if CONFIG.get("PREPROCESSING", "features") == "pcen" else True
    pipeline = CustomPipeline(resample_freq=RESAMPLE_FREQ, n_fft=N_FFT,
                              n_mel=N_MELS, augment=augment, logmel=logmel)
    dataset = CustomDataset(split=split, pipeline=pipeline)
    if device == "cuda":
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, pin_memory_device=device, num_workers=2, prefetch_factor=4)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=3)
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


def register_hooks(model: TripletNet):
    """
    Register hooks for each layer of the network.
    
    Parameters
    ----------
    model: TripletNet
        Model for which to retrieve the layers.
    
    """
    for name, layer in model._modules.items():
        if isinstance(layer, nn.Sequential):
            register_hooks(layer)
        else:
            layer.register_forward_hook(hook_fn)


def create_model_from_trial() -> Tuple[TripletNet, float]:
    """
    Instantiates a model from a trial in order to be able to perform training.
    
    Returns
    -------
    model, weight_decay: Tuple[TripletNet, float]
        Model to train, instantiated using model selection trial results, and weight_decay.
    """
    try:
        fpath = os.path.join("tnet_trials", CONFIG.get("OPTIMIZATION", "tnet_trial"), "params.json")
        with open(fpath, "r") as f:
            params = json.load(f) # ["config"]
        cnn = Cnn(out_channels_1=params["out_channels_1"], out_channels_2=params["out_channels_2"], out_channels_3=params["out_channels_3"],
                  kernel_size_1=params["kernel_size_1"], kernel_size_2=params["kernel_size_2"], kernel_size_3=params["kernel_size_3"],
                  bn_momentum=params["bn_momentum"], dropout_1=params["dropout_1"], dropout_2=params["dropout_2"], dropout_3=params["dropout_3"])
        model = TripletNet(embedding_model=cnn)
        return model, params["weight_decay"]
    
    except Exception as ex:
        print(ex)


def load_model() -> TripletNet:
    """
    Loads the model from the disk.

    Returns
    -------
    model: TripletNet
        Trained TripletNet object.

    Raises
    ------
    FileNotFoundError: whenever the saved model cannot be found.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(CONFIG.get("MODELS", "save_path"),
                              "triplet_network.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Cannot find saved model.")
    model = torch.load(model_path, map_location=torch.device(device))
    return model


def save_model(epoch: int, model: TripletNet, optimizer: torch.optim.Optimizer, loss: torch.nn.Module,
               filename: str):
    """
    Saves the model on disk.
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, filename)


def eval_tnet(model: TripletNet) -> float:
    """
    Evaluates the model trained with triplet margin loss for model selection purposes.

    Parameters
    ----------
    model: TripletNet
        TripletNet instance.

    Returns
    -------
    avg_val_loss: float
        Average validation loss on validation split.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # define dataloader
    val_dataloader = get_dataloader(split="val_train", batch_size=CONFIG.getint("TRAINING", "batch_size"))

    # define triplet miner and loss
    distance=distances.LpDistance(power=2)
    margin=CONFIG.getfloat("MODELS", "triplet_net_margin")
    miner = miners.TripletMarginMiner(margin=margin, distance=distance, type_of_triplets=CONFIG.get("MODELS", "triplets_type"))  # "semihard")
    loss_fn = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducers.MeanReducer())    
    avg_val_loss = 0
    model = model.eval()
    with torch.no_grad():
        for images, _, labels, label_ids in iter(val_dataloader):
            images = images.to(device)
            # batch_labels = torch.tensor(label_ids, device=device)
            batch_labels = label_ids.clone().detach().to(device)
            batch_labels = torch.argmax(batch_labels, dim=1)
            embeddings = model(images)
            triplets_idxs = miner(embeddings, batch_labels)
            loss = loss_fn(embeddings, batch_labels, triplets_idxs)
            avg_val_loss += loss.item()
        avg_val_loss /= len(val_dataloader)
    return avg_val_loss


def train_tnet(model: TripletNet, optimizer: torch.optim.Optimizer, epochs: int) -> TripletNet:
    """
    Performs the training with triplet margin loss for model selection purposes.
    
    Parameters
    ----------
    model: TripletNet
        TripletNet instance.
    optimizer: torch.optim.Optimizer
        Optimizer object.
    epochs: int
        Number of epochs.
    
    Returns
    -------
    model: TripletNet
        Trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # define dataloader
    train_dataloader = get_dataloader(split="train", batch_size=CONFIG.getint("TRAINING", "batch_size"))

    # define triplet miner and loss
    distance=distances.LpDistance(power=2)
    margin=CONFIG.getfloat("MODELS", "triplet_net_margin")
    miner = miners.TripletMarginMiner(margin=margin, distance=distance, type_of_triplets=CONFIG.get("MODELS", "triplets_type"))  # "semihard")
    loss_fn = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducers.MeanReducer())
    model = model.train()
    for epoch in range(epochs):
        avg_train_loss = 0.
        for images, _, labels, label_ids in iter(train_dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            # batch_labels = torch.tensor(label_ids, device=device)
            batch_labels = label_ids.clone().detach().to(device)
            batch_labels = torch.argmax(batch_labels, dim=1)
            embeddings = model(images)
            triplets_idxs = miner(embeddings, batch_labels)
            loss = loss_fn(embeddings, batch_labels, triplets_idxs)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        avg_train_loss /= len(train_dataloader)
        print(f"Average training loss for epoch {epoch}: ", avg_train_loss)
    return model


def train_triplet_net():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build the model and transfer to device
    model, weight_decay = create_model_from_trial()
    model.to(device)

    # Set the optimizer and some variables for data gathering
    best_loss = None
    # weight_decay = load_params_from_trial(trial_name=CONFIG.get("OPTIMIZATION", "trial_name"))["config"]["weight_decay"]
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=weight_decay,
      #                           nesterov=True, momentum=0.9, dampening=0)
    optimizer = torch.optim.NAdam(params=model.parameters(), lr=1e-4, weight_decay=weight_decay)
    epochs = CONFIG.getint("TRAINING", "epochs")

    # Parameters to define early stopping
    delta_loss = CONFIG.getfloat("TRAINING", "delta_loss")
    patience = CONFIG.getint("TRAINING", "patience")

    # savepath and history
    save_path = CONFIG.get("MODELS", "save_path")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hist = pd.DataFrame([], columns=["Epoch", "Training loss", "Validation loss"])
    
    # define dataloader
    train_dataloader = get_dataloader(split="train")
    val_dataloader = get_dataloader(split="val_train")

    # define triplet miner and loss
    distance=distances.LpDistance(power=2)  # p=2 and normalize_embeddings=True by default
    margin=CONFIG.getfloat("MODELS", "triplet_net_margin")
    miner = miners.TripletMarginMiner(margin=margin, distance=distance, type_of_triplets=CONFIG.get("MODELS", "triplets_type"))  # "hard")
    loss_fn = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducers.MeanReducer())  #reducer=reducers.AvgNonZeroReducer())
    # miner = miners.PairMarginMiner(pos_margin=0.15, neg_margin=0.85, distance=distance)
    # loss_fn = losses.CircleLoss(m=0.25, gamma=256)

    for epoch in tqdm(range(epochs), desc="Starting epoch\n", total=epochs):
        if patience == 0:
            print("\nEarly stopping..")
            break
        avg_train_loss = 0
        avg_val_loss = 0
        
        # Training part
        model = model.train()
        for images, _, labels, label_ids in tqdm(iter(train_dataloader), desc="Processing batches",
                                                 total=len(train_dataloader), leave=False):
            optimizer.zero_grad()
            images = images.to(device)
            batch_labels = label_ids.clone().detach().to(device)
            batch_labels = torch.argmax(batch_labels, dim=1)
            embeddings = model(images)
            triplets_idxs = miner(embeddings, batch_labels)
            loss = loss_fn(embeddings, batch_labels, triplets_idxs)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        avg_train_loss /= len(train_dataloader)

        # Validation part
        model.eval()
        with torch.no_grad():
            for images, _, labels, label_ids in tqdm(iter(val_dataloader), desc="Processing validation batches",
                                                     total=len(val_dataloader), leave=False):
                images = images.to(device)
                batch_labels = label_ids.clone().detach().to(device)
                batch_labels = torch.argmax(batch_labels, dim=1)
                embeddings = model(images)
                triplets_idxs = miner(embeddings, batch_labels)
                loss = loss_fn(embeddings, batch_labels, triplets_idxs)
                avg_val_loss += loss.item()
            avg_val_loss /= len(val_dataloader)

        # Update history, best loss and patience
        hist = pd.concat([hist, pd.DataFrame([
            [epoch, avg_train_loss, avg_val_loss]], columns=hist.columns)
        ])

        if best_loss is None or avg_val_loss <= (best_loss - delta_loss):
            best_loss = avg_val_loss
            patience = CONFIG.getint("TRAINING", "patience")
            save_model(epoch, model, optimizer, avg_val_loss, os.path.join(save_path, f"triplet_network.pt"))
        else:
            patience -= 1

    hist.to_csv(os.path.join(save_path, "tnet_training_history.csv"), index=False,
                columns=hist.columns)
            