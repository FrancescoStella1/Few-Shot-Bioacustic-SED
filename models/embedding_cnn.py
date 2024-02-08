import json
import os
import platform
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torcheval.metrics.functional import multiclass_f1_score

from config import CONFIG
from utils import load_class_weights
from preprocessing.feature_extraction import get_dataloader


class Cnn(nn.Module):
    """CNN model intended for performing model selection and for later use in other DNNs."""
    def __init__(self, input_dim: tuple, n_classes: int, class_weights: dict,
                 out_channels_1: int, out_channels_2: int, out_channels_3: int, out_channels_4: int,
                 kernel_size_1: int, kernel_size_2: int, kernel_size_3: int, kernel_size_4: int,
                 padding: str, bn_momentum: float, dropout_1: float, dropout_2: float, dropout_3: float, dropout_4: float,
                 dropout_5: float, out_features_1: int, out_features_2: int):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels_1,
                kernel_size=kernel_size_1,
                padding=padding
            ),
            nn.BatchNorm2d(out_channels_1, momentum=bn_momentum),
            nn.PReLU(),
            nn.Dropout(dropout_1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_1,
                out_channels=out_channels_2,
                kernel_size=kernel_size_2,
                padding=0
            ),
            nn.BatchNorm2d(out_channels_2, momentum=bn_momentum),
            nn.PReLU(),
            nn.Dropout(dropout_2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_2,
                out_channels=out_channels_3,
                kernel_size=kernel_size_3,
                padding=0
            ),
            nn.BatchNorm2d(out_channels_3, momentum=bn_momentum),
            nn.PReLU(),
            nn.Dropout(dropout_3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_3,
                out_channels=out_channels_4,
                kernel_size=kernel_size_4,
                padding=0
            ),
            nn.BatchNorm2d(out_channels_4, momentum=bn_momentum),
            nn.PReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Sequential(
            nn.LazyLinear(out_features=out_features_1),
            nn.BatchNorm1d(num_features=out_features_1),
            nn.PReLU(),
            nn.Dropout(dropout_4)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=out_features_1, out_features=out_features_2),
            nn.BatchNorm1d(num_features=out_features_2),
            nn.PReLU(),
            nn.Dropout(dropout_5)
        )
        self.fc_3 = nn.Linear(in_features=out_features_2, out_features=n_classes)
        c_weights_tensor = torch.Tensor(list(class_weights.values()))
        self.loss = nn.CrossEntropyLoss(weight=c_weights_tensor, reduction="mean")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x


def eval_cnn(model: Cnn) -> Tuple[float, np.ndarray]:
    """
    Evaluates CNN on the validation split of the training dataset (intended for model selection).

    Parameters
    ----------
    model: Cnn
        Trained instance of the Cnn model.
    
    Returns
    -------
    val_loss, f1: Tuple[float, np.ndarray]
        Tuple containing validation loss and f1 score.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dataloader = get_dataloader(split="val_train")
    model.to(device)
    model.eval()
    val_loss = 0
    n_batches = len(val_dataloader)
    epoch_predictions = None
    epoch_targets = None

    with (torch.no_grad()):
        for images, sr, labels, label_ids in iter(val_dataloader):
            # Compute validation loss
            targets = torch.argmax(label_ids.to(device), dim=1)
            # label_ids = label_ids.to(device)
            predictions = model(images.to(device))
            
            # Update val_loss
            val_batch_loss = model.loss(predictions, targets.to(device))
            val_loss += val_batch_loss.item()

            # collect data for F1 score
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)
            epoch_predictions = predictions if epoch_predictions is None else \
                torch.cat((epoch_predictions, predictions))
            epoch_targets = targets if epoch_targets is None else \
                torch.cat((epoch_targets, targets))
            
        val_loss /= n_batches
        f1 = multiclass_f1_score(epoch_predictions.to("cpu"), epoch_targets.to("cpu"),
                                 num_classes=CONFIG.getint("TRAINING", "training_classes"), average="weighted")
    return val_loss, f1.numpy()


def get_embedding_model(truncate: bool = True, flatten: bool = True) -> nn.Module:
    """
    Returns the embedding model created starting from the BOHB chosen trial.

    Parameters
    ----------
    truncate: bool
        Specifies whether to truncate the last layers of the network (default: True).
    flatten: bool
        Specifies whether to flatten the output (default: True).

    Returns
    -------
    emb_model: nn.Module
        Embedding model.
    """
    try:
        trial_name = CONFIG.get("OPTIMIZATION", "trial_name")
        params = load_params_from_trial(trial_name=trial_name)
        class_weights = load_class_weights()
        params = params["config"]
        cnn = Cnn(input_dim=(128, 87), n_classes=46, class_weights=class_weights,
                  out_channels_1=params["out_channels_1"], out_channels_2=params["out_channels_2"],
                  out_channels_3=params["out_channels_3"], out_channels_4=params["out_channels_4"],
                  kernel_size_1=params["kernel_size_1"], kernel_size_2=params["kernel_size_2"],
                  kernel_size_3=params["kernel_size_3"], kernel_size_4=params["kernel_size_4"],
                  dropout_1=params["dropout_1"], dropout_2=params["dropout_2"], dropout_3=params["dropout_3"],
                  dropout_4=params["dropout_4"], dropout_5=params["dropout_5"], padding=params["padding"],
                  bn_momentum=params["bn_momentum"], out_features_1=params["out_features_1"], out_features_2=params["out_features_2"])
        if not flatten:
            emb_model = nn.Sequential(*(list(cnn.children())[:4]))
        elif truncate:
            emb_model = nn.Sequential(*(list(cnn.children())[:5]))
        else:
            emb_model = nn.Sequential(*(list(cnn.children())))
        return emb_model
    
    except Exception as ex:
        print(ex)
    

def load_params_from_trial(trial_name: str) -> dict:
    try:
        if platform.system() == "Linux":
            fpath = f"/home/francesco/Documenti/git/DCASE_task5/trials/{trial_name}/result.json"
        else:
            fpath = os.path.join("trials", trial_name, "result.json")
        with open(fpath, "r") as f:
            params = json.load(f)
        return params

    except Exception as ex:
        print(ex)


def train_cnn(model: Cnn, optimizer: torch.optim.Optimizer, epochs: int) -> Cnn:
    """
    Trains the CNN model (intended for the model selection task).
    
    Parameters
    ----------
    model: Cnn
        Model instance.
    optimizer: torch.optim.Optimizer
        Optimizer object.
    epochs: int
        Number of epochs for model training.
    
    Returns
    -------
    model: Cnn
        Trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(epochs):
        train_dataloader = get_dataloader(split="train")
        model.train()
        avg_loss = 0
        n_batches = len(train_dataloader)
        for images, sr, labels, label_ids in iter(train_dataloader):
            optimizer.zero_grad()
            targets = torch.argmax(label_ids, dim=1)
            predictions = model(images.to(device))
            batch_loss = model.loss(predictions, targets.to(device))
            batch_loss.backward()
            optimizer.step()
            avg_loss += batch_loss.item()
        avg_loss /= n_batches
        print(f"Average training loss for epoch {epoch}: ", avg_loss)

    return model


def train_cnn_main(model: nn.Module, optimizer: Optimizer, epochs: int, patience: int = 5) -> pd.DataFrame:
    """
    Performs CNN embeddings model training. It is meant to be used in the main (not for model selection).

    Parameters
    ----------
    model: nn.Module
        CNN model.
    optimizer: Optimizer
        Optimizer object.
    epochs: int
        Number of epochs.
    patience: int
        Number of epochs to wait until early stopping if no improvement is seen (default: 5).

    Returns
    -------
    training_hist: pd.DataFrame
        Training history.
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    filename = f"cnn_{CONFIG.get('OPTIMIZATION', 'trial_name')}.pt"
    filepath = os.path.join(CONFIG.get("MODELS", "save_path"), filename)

    hist = pd.DataFrame([], columns=["Epoch", "Training loss", "Validation loss", "Training f1 score",
                                     "Validation f1 score"])
    loop_patience = patience
    best_val_loss = np.Inf

    train_dataloader = get_dataloader(split="train")
    val_dataloader = get_dataloader(split="val_train")

    # Training
    for epoch in range(epochs):
        if loop_patience < 0:
            break
        model.train()
        avg_train_loss = 0
        avg_val_loss = 0
        epoch_predictions = None
        epoch_targets = None
        train_n_batches = len(train_dataloader)
        val_n_batches = len(val_dataloader)

        for images, sr, labels, label_ids in iter(train_dataloader):
            optimizer.zero_grad()
            # perform inference
            predictions = model(images.to(device))
            targets = torch.argmax(label_ids, dim=1)

            # propagate error and update weights
            batch_loss = model.loss(predictions, targets.to(device))
            batch_loss.backward()
            optimizer.step()

            # update loss
            avg_train_loss += batch_loss.item()

            # collect data for F1 score
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)
            epoch_predictions = predictions if epoch_predictions is None else \
                torch.cat((epoch_predictions, predictions))
            epoch_targets = targets if epoch_targets is None else \
                torch.cat((epoch_targets, targets))
            
        avg_train_loss /= train_n_batches
        train_f1_score = multiclass_f1_score(epoch_predictions.to("cpu"), epoch_targets.to("cpu"),
                                             num_classes=CONFIG.getint("TRAINING", "training_classes"), average="weighted")

        # Validation
        epoch_predictions = None
        epoch_targets = None
        model.eval()
        with torch.no_grad():
            for images, sr, labels, label_ids in iter(val_dataloader):
                # perform inference
                predictions = model(images.to(device))
                targets = torch.argmax(label_ids, dim=1)

                # estimate loss and update validation average loss
                batch_loss = model.loss(predictions, targets.to(device))
                avg_val_loss += batch_loss.item()

                # collect data for f1 score
                predictions = torch.nn.functional.softmax(predictions, dim=1)
                predictions = torch.argmax(predictions, dim=1)
                epoch_predictions = predictions if epoch_predictions is None else \
                    torch.cat((epoch_predictions, predictions))
                epoch_targets = targets if epoch_targets is None else \
                    torch.cat((epoch_targets, targets))

            avg_val_loss /= val_n_batches
            val_f1_score = multiclass_f1_score(epoch_predictions.to("cpu"), epoch_targets.to("cpu"),
                                               num_classes=CONFIG.getint("TRAINING", "training_classes"), average="weighted")

            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss
                loop_patience = patience
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                    "val_f1_score": val_f1_score
                }, filepath)
            else:
                loop_patience -= 1
        tmp_hist = pd.DataFrame([[epoch, avg_train_loss, avg_val_loss, train_f1_score.item(),
                                  val_f1_score.item()]], columns=hist.columns)
        hist = pd.concat([hist, tmp_hist])
        print(f"Epoch {epoch}:\n", hist, flush=True)
    hist.reset_index(inplace=True, drop=True)
    hist.to_csv(f"serialized/models/{filename[:-3]}_hist.csv")
    print("Done.")
    return hist
