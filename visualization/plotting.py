import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import nn, Tensor
from torchview import draw_graph

from config import CONFIG


def plot_architecture(model: nn.Module, model_name: str):
    """
    Saves an image representing the model's architecture.
    
    Parameters
    ----------
    model: nn.Module
        Model for which to plot the architecture.
    model_name: str
        Name of the model.
    """
    batch_size = CONFIG.getint("TRAINING", "batch_size")
    if model_name == "rnet":
        input_size = ((240, 1, 128, 44), (240, 1, 128, 44))
    elif model_name == "snet":
        input_size = ((240, 1, 128, 44), (240, 1, 128, 44))
    elif model_name == "tnet":
        input_size = ((batch_size, 1, 128, 44))
    model_graph = draw_graph(model, input_size=input_size, expand_nested=True)
    model_graph.resize_graph(scale=5.0)
    save_dir = CONFIG.get("MODELS", "save_path")
    model_graph.visual_graph.render(format="svg", directory=save_dir,
                                    filename=model_name)
    print(f"Model architecture saved in directory: {save_dir}.")


def plot_dataset_stats(filepath: str, sampling_rate: int = 22050):
    """
    Visualizes some stats for the training dataset.

    Parameters
    ----------
    filepath: str
        Path to the serialized training set (feather extension).
    sampling_rate: int
        Sampling rate of the signals (default: 22050).
    """
    avglen = 0
    maxlen = 0
    maxlen_class = None
    minlen = 10
    minlen_class = None
    class_samples = {}
    class_durations = {}
    train_data = pd.read_feather(filepath)
    for idx, row in train_data.iterrows():
        signal_len = row["Time signal"][0].shape[0] / sampling_rate
        # avglen += signal_len
        if signal_len > maxlen:
            maxlen = signal_len
            maxlen_class = row["Class"]
        if signal_len < minlen:
            minlen = signal_len
            minlen_class = row["Class"]
        if row["Class"] in class_samples.keys():
            class_samples[row["Class"]] += 1
            class_durations[row["Class"]] += signal_len
        else:
            class_samples[row["Class"]] = 1
            class_durations[row["Class"]] = signal_len
    # avglen /= train_data["Class"].count()
    for k, v in class_durations.items():
        avglen += (v/class_samples[k])
    avglen /= len(list(class_samples.keys()))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(list(class_samples.keys()), list(class_samples.values()), align="center")
    ax.set_ylabel("# of samples")
    ax.set_xlabel("Class")
    ax.set_xticklabels(list(class_samples.keys()), rotation=45)

    print(f"max length: {maxlen} - class: {maxlen_class}")
    print(f"weighted average length: {avglen}")
    print(f"min length: {minlen} - class: {minlen_class}")

    plt.title("Training Set statistics")
    plt.show()


def plot_rms(signal: np.ndarray):
    """
    Plots Root Mean Square Energy of the given signal.
    
    Parameters
    ----------
    signal: np.ndarray
        Time-domain signal.
    """
    S = librosa.stft(y=signal, n_fft=1024, hop_length=256)
    rms, phase = librosa.magphase(S)
    fig = plt.figure()
    plt.tight_layout()
    times = librosa.times_like(rms)
    ax = fig.add_subplot()
    ax.semilogy(times, rms.ravel(), label="RMS Energy")
    ax.set(xticks=[])
    ax.legend()
    ax.label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, red=np.max), y_axis="log", x_axis="time", ax=ax)
    plt.suptitle("Root Mean Square Energy")
    plt.show()


def plot_zcr(signal: np.ndarray):
    """
    Plots Zero-Crossing Rate of the given signal.
    
    Parameters
    ----------
    signal: np.ndarray
        Time-domain signal.
    """
    zcr = librosa.feature.zero_crossing_rate(signal)
    fig = plt.figure()
    plt.tight_layout()
    ax = fig.add_subplot()
    ax.plot(np.arange(0, zcr.shape[1]), zcr.ravel(), color="red", alpha=0.6)
    fig.suptitle("Zero-Crossing Rate")
    plt.show()


def visualize_feature_map(feature_map: np.ndarray, title: str):
    """
    Visualizes feature map of a specific layer of the network.
    
    Parameters
    ----------
    feature_map: np.ndarray
        Output of the chosen layer of the network.
    title: str
        Title of the figure.
    """
    n_channels = feature_map.shape[0]
    n_cols = min(n_channels//4, 4)
    n_rows = 2  # int(n_channels/n_cols)
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.suptitle(title)
    for idx in range(8):
        row = idx // n_cols  # 4
        col = idx % n_cols  # 4
        ax[row][col].imshow(feature_map[idx], cmap="inferno")
        ax[row][col].tick_params(left=False, right=False, labelleft=False, labelright=False, labelbottom=False, bottom=False)
        ax[row][col].set_title(f"Channel {idx+1}")
    plt.show()


def visualize_embeddings_kmeans(embeddings: np.ndarray, labels: np.ndarray, three_dim: bool = False, standardize: bool = True):
    """
    Performs k-means algorithm to visualize the learned embeddings of the chosen model.

    Parameters
    ----------
    model_name: str
        Name of the model for which to visualize the embeddings.
    embeddings: np.ndarray
        Embeddings to visualize.
    labels: np.ndarray
        Array containing the labels of the embeddings.
    three_dim: bool
        Specifies whether to visualize a 3d plot (default: False).
    standardize: bool
        Specifies whether to perform standardization (default: True).

    Raises: FileNotFoundError
        Whenever the saved model file cannot be found.
    """
    n_clusters = np.unique(labels).shape[0]
    n_components = 3 if three_dim else 2
    pca = PCA(n_components=n_components, random_state=42)
    if standardize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    y = pca.fit_transform(embeddings)
    kms = KMeans(n_clusters=n_clusters, tol=1e-8, max_iter=1000, verbose=1, random_state=42)
    kms.fit(y)
    y = np.column_stack((y, labels))
    if n_components == 3:
        df = pd.DataFrame(y, columns=["x", "y", "z", "class"])
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        df["z"] = df["z"].astype(float)
        df["class"] = df["class"].astype(str)
        figure = px.scatter_3d(df, x="x", y="y", z="z", color="class", symbol="class")
        marker_size = 5
    else:
        df = pd.DataFrame(y, columns=["x", "y", "class"])
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        df["class"] = df["class"].astype(str)
        figure = px.scatter(df, x="x", y="y", color="class", symbol="class")
        marker_size = 10

    figure.update_traces(marker_size=marker_size)
    figure.update_layout({
        "plot_bgcolor": "rgba(245, 245, 245, 0.3)"
    })
    figure.show()


def visualize_embeddings_tsne(embeddings: np.ndarray, labels: np.ndarray, three_dim: bool = False, standardize: bool = True):
    """
    Applies t-SNE algorithm to visualize the learned embeddings of the chosen model.

    Parameters
    ----------
    model_name: str
        Name of the model for which to visualize the embeddings.
    embeddings: np.ndarray
        Embeddings to visualize.
    labels: np.ndarray
        Array containing the labels of the embeddings.
    three_dim: bool
        Specifies whether to visualize a 3d plot (default: False).
    standardize: bool
        Specifies whether to perform standardization (default: True).

    Raises: FileNotFoundError
        Whenever the saved model file cannot be found.
    """
    if type(labels) != np.ndarray:
        labels = np.array(labels)
    # n_clusters = np.unique(labels).shape[0]
    n_components = 3 if three_dim else 2
    if standardize:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    pca_components = 64
    pca = PCA(n_components=pca_components, random_state=42)
    y = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=500)
    y = tsne.fit_transform(y)
    y = np.column_stack((y, labels))
    if n_components == 3:
        df = pd.DataFrame(y, columns=["x", "y", "z", "class"])
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        df["z"] = df["z"].astype(float)
        df["class"] = df["class"].astype(str)
        figure = px.scatter_3d(df, x="x", y="y", z="z", color="class", symbol="class")
        marker_size = 5
    else:
        df = pd.DataFrame(y, columns=["x", "y", "class"])
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        df["class"] = df["class"].astype(str)
        figure = px.scatter(df, x="x", y="y", color="class", symbol="class")
        marker_size = 10

    figure.update_traces(marker_size=marker_size)
    figure.update_layout({
        "plot_bgcolor": "rgba(245, 245, 245, 0.3)"
    })
    figure.show()


def visualize_training_loss(model: str, title: str):
    """
    Visualizes the training loss.

    Parameters
    ----------
    model: str
        Name of the model for which to plot the training loss.
    title: str
        Title of the plot.

    Raises: FileNotFoundError
        Whenever training and/or validation history .csv files cannot be found.
    """
    if model.lower() in ["tnet", "triplet_net", "triplet_network"]:
        model = "tnet"
    try:
        filepath = CONFIG.get("MODELS", "save_path")
        filepath = os.path.join(filepath, f"{model}_training_history.csv")
        hist = pd.read_csv(filepath)
        fig, ax = plt.subplots()
        ax.plot(hist["Epoch"], hist["Training loss"], color="black", label="Training")
        ax.plot(hist["Epoch"], hist["Validation loss"], color="orange", label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        print(hist)
        print(f"Best training loss: {hist['Training loss'].min()}\nBest validation loss: {hist['Validation loss'].min()}")
        plt.legend()
        plt.show()

    except Exception as ex:
        raise FileNotFoundError(f"Cannot find training and/or validation history for {model}."
                                f"Please check the serialized/models directory.")


def visualize_spectrogram(feature: Tensor, title: str, xlabel: str, ylabel: str):
    _, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.imshow(feature, origin="lower", aspect="auto", interpolation="nearest")  # librosa.power_to_db(feature.numpy())
    plt.show(block=False)
    plt.waitforbuttonpress()


def visualize_logmel(image: np.ndarray, title: str, x_axis: str = "time", y_axis: str = "mel"):
    fig, ax = plt.subplots(nrows=1)
    image = librosa.display.specshow(image, x_axis=x_axis, y_axis=y_axis, cmap="viridis")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    plt.show()

def visualize_pcen(image: np.ndarray, title: str, x_axis: str = "time", y_axis: str = "mel"):
    fig, ax = plt.subplots(nrows=1)
    ax.set_yscale("symlog")
    img = librosa.display.specshow(image, x_axis=x_axis, y_axis=y_axis, ax=ax, cmap="viridis")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.1f dB")
    plt.show()
