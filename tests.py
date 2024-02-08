from argparse import ArgumentParser
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import embedding_cnn
from models.embedding_cnn import Cnn
import models.relation_net as rnet
import models.siamese_net as snet
import models.triplet_net as tnet
from preprocessing.feature_extraction import CustomDataset
from preprocessing.feature_extraction import CustomPipeline
from preprocessing.feature_extraction import get_dataloader
from evaluator import Evaluator
from visualization.plotting import visualize_logmel, visualize_pcen


from config import CONFIG

N_FFT = CONFIG.getint("PREPROCESSING", "n_fft")
N_MELS = CONFIG.getint("PREPROCESSING", "n_mels")
RESAMPLE_FREQ = CONFIG.getint("PREPROCESSING", "resampling_frequency")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dataloader", action="count", help="Tests the standard Dataloader for the CNN"
                                                                  "embedding module.")
    parser.add_argument("--test_cnn", action="count", help="Tests the inference through the embedding CNN.")
    parser.add_argument("--test_evaluator", action="count", help="Tests the evaluator class.")
    parser.add_argument("--test_rnet", action="count", help="Tests Relation Network model.")
    parser.add_argument("--test_rnet_dataloader", action="count", help="Tests Relation Network dataloader.")
    parser.add_argument("--test_siamese_dataloader", action="count", help="Tests the Siamese Network dataloader.")
    parser.add_argument("--test_tnet_dataloader", action="count", help="Tests the Triplet Network dataloader.")
    parser.add_argument("--train_cnn", action="count", help="Tests the training loop of the embedding CNN.")
    parser.add_argument("--visualize_features", type=str, nargs=1, help="Visualizes extracted features for the specified class.")
    parser.add_argument("--visualize_neg_features", action="count", help="Visualizes some NEG features extracted from dataset.")
    args = parser.parse_args()

    if args.test_dataloader:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logmel = False if CONFIG.get("PREPROCESSING", "features") == "pcen" else True
        pipeline = CustomPipeline(resample_freq=RESAMPLE_FREQ, n_fft=N_FFT, n_mel=N_MELS, augment=True, logmel=logmel)
        db = CustomDataset(split="val_train", pipeline=pipeline)
        if device == "cuda":
            dataloader = DataLoader(dataset=db, batch_size=32, shuffle=True, pin_memory=True, pin_memory_device=device)
        else:
            dataloader = DataLoader(dataset=db, batch_size=32, shuffle=True)
        for images, sr, labels, label_ids in iter(dataloader):
            # images = torch.nn.functional.normalize(images, p=1, dim=2)
            img = images[0].squeeze(dim=0).numpy()
            if logmel:
                visualize_logmel(image=img, title=f"Log-Melspec of {labels[0]}")
            else:
                visualize_pcen(image=img, title=f"{labels[0]} PCEN")
            break

    if args.test_siamese_dataloader:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logmel = False
        dataloader = snet.get_dataloader(split="val_train")
        cnt = 0
        for images, label_ids, labels in iter(dataloader):
            if cnt >= 3:
                break
            anchor, query = images
            a_img = anchor[0, 0].squeeze().numpy()
            q_img = query[0, 0].squeeze().numpy()
            if logmel:
                visualize_logmel(image=a_img, title=f"Anchor log-Mel spectrogram of {labels[0][0]}")
                visualize_logmel(image=q_img, title=f"Query log-Mel spectrogram of {labels[1][0]}")
            else:
                visualize_pcen(image=a_img, title=f"Anchor PCEN of {labels[0][0]}")
                visualize_pcen(image=q_img, title=f"Query PCEN of {labels[1][0]}")
            cnt += 1
    
    if args.test_tnet_dataloader:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dataloader = tnet.get_dataloader(split="train", batch_size=CONFIG.getint("TRAINING", "batch_size"))
        results = {}
        for images, _, labels, label_ids in iter(dataloader):
            batch_labels = label_ids.clone().detach().to(device)
            batch_labels = torch.argmax(batch_labels, dim=1)
            for idx, label in enumerate(labels):
                if label in results.keys():
                    assert results[label] == batch_labels[idx].item()
                else:
                    results[label] = batch_labels[idx].item()
        assert len(list(results.values())) == len(set(results.values()))
        print("OK")
        print("\n", results)

    if args.test_cnn:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = CustomPipeline(resample_freq=RESAMPLE_FREQ, n_fft=N_FFT, n_mel=N_MELS)
        db = CustomDataset(split="train", pipeline=pipeline)
        class_weights = db.get_class_weights()
        if device == "cuda":
            dataloader = DataLoader(dataset=db, batch_size=32, shuffle=True, pin_memory=True, pin_memory_device=device)
        else:
            dataloader = DataLoader(dataset=db, batch_size=32, shuffle=True)
        cnn = Cnn(input_dim=(128, 87), n_classes=46, class_weights=class_weights, out_channels_1=16, out_channels_2=32,
                  out_channels_3=64, out_channels_4=128, kernel_size_1=3, kernel_size_2=5, kernel_size_3=5,
                  kernel_size_4=7, padding="same", out_features_1=1024, out_features_2=128).to(device)
        for examples, sr, labels, label_ids in iter(dataloader):
            examples = examples.to(device)
            label_ids = label_ids.to(device)
            pred = cnn(examples, label_ids)
            print("Predictions shape: ", pred.shape)
            avg_loss, f1_score = embedding_cnn.eval_cnn(cnn)
            print(f"Validation CE loss and F1 score:\n\tCE loss: {avg_loss}\n\tF1 score: {f1_score}")
            break

    if args.test_evaluator:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        folder = "ME"
        filename = "ME2.wav"
        cnn = tnet.get_embedding_model(flatten=False)
        checkpoint = rnet.load_model(load_finetuned=False)
        # model = snet.SiameseNet(embedding_model=cnn)
        model = rnet.create_model(embedding_model=cnn)
        # checkpoint = tnet.load_model()
        # model = tnet.TripletNet(embedding_model=cnn)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model = model.eval()
        evaluator = Evaluator(base_path=os.path.join("dataset", "Validation_Set"), model=model, generate_triplets=False)
        # evaluator.evaluate_model(debugging=True)
        print(f"Processing: {filename}\n")
        img_generator = evaluator.read_audiofile(folder=folder, filename=filename, hop_length=5512, debugging=True)
        logmel = False if CONFIG.get("PREPROCESSING", "features") == "pcen" else True
        for images, times in img_generator:
            prediction = model(images[0].unsqueeze(0).to(device),
                               images[1].unsqueeze(0).to(device))
            if type(prediction) == tuple:
                prediction = prediction[0]
            if prediction.item() >= .5:
                print(f"Prediction: {prediction.item()}")
                print(f"Times: {times[0]}, {times[1]}")
                if logmel:
                    visualize_logmel(image=images[1].squeeze().numpy(), title="Query log-Mel spectrogram")
                else:
                    visualize_pcen(image=images[1].squeeze().numpy(), title="Query PCEN")
        
    if args.test_rnet:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        dataloader = rnet.get_dataloader(split="train")
        cnn = tnet.get_embedding_model(flatten=False)
        model = rnet.create_model(embedding_model=cnn)
        model = model.eval()
        model.to(device)
        (anchor_images, query_images), (a_label_ids, q_label_ids), (a_labels, q_labels) = next(iter(dataloader))
        inputs = torch.Tensor([]).to(device)
        anchor_images = anchor_images.squeeze()
        print("Batch shape: ", anchor_images.shape)
        for idx in range(anchor_images.shape[0]):
            img = anchor_images[idx].to(device)
            print("PCEN shape: ", img.shape)
            inputs = img if inputs.shape[0] == 0 else torch.row_stack((inputs, img))
            break
        inputs = inputs.unsqueeze(0)
        targets = torch.argmax(a_label_ids, dim=1).to(device)
        print(inputs.shape, targets.shape)
        predictions = model(inputs)
        print("Predictions: ", predictions)

    if args.test_rnet_dataloader:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_data = pd.read_feather("serialized/dataset/training.feather")
        n_classes = CONFIG.getint("TRAINING", "n_classes")
        n_samples = CONFIG.getint("TRAINING", "n_samples")
        dataloader = rnet.get_dataloader(split="training")

    if args.train_cnn:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_weights_path = CONFIG.get("TRAINING", "class_weights_linux_abspath")
        class_weights_path = os.path.join(class_weights_path, "class_weights.pt")
        class_weights = torch.load(class_weights_path)
        cnn = Cnn(input_dim=(128, 87), n_classes=46, class_weights=class_weights, out_channels_1=16, out_channels_2=32,
                  out_channels_3=32, out_channels_4=64, kernel_size_1=3, kernel_size_2=5, kernel_size_3=5,
                  kernel_size_4=7, padding="same", out_features_1=256, out_features_2=64).to(device)
        
        optimizer = torch.optim.NAdam(params=cnn.parameters(), lr=1e-3, momentum_decay=0.02, weight_decay=0.1)
        embedding_cnn.train_cnn(cnn, optimizer, 5)

    if args.visualize_features:
        label = sys.argv[2]
        pipeline = CustomPipeline(resample_freq=RESAMPLE_FREQ, n_fft=N_FFT, n_mel=N_MELS, augment=False, logmel=False)
        dataset = "validation_feather"
        column = "Filepath" if dataset == "validation_feather" else "Class"
        data = pd.read_feather(CONFIG.get("DATASETS", dataset))
        examples = data[data[column] == label]
        n_samples = 22050
        cnt = 0
        for idx, row in examples.iterrows():
            # if cnt >= 5:
                # break
            signal = row.iloc[2][0].copy()
            signal = np.expand_dims(signal, axis=0)
            
            if signal.shape[1] > n_samples:
                signal = signal[:, :n_samples]
            
            signal = torch.Tensor(signal)
            if signal.shape[1] < n_samples:
                diff_len = n_samples - signal.shape[1]
                padding = (0, diff_len)
                signal = torch.nn.functional.pad(signal, padding)

            img = pipeline.forward(signal)
            visualize_pcen(image=img.squeeze().numpy(), title=f"PCEN of {row.iloc[0]}")
            cnt += 1

    if args.visualize_neg_features:
        pipeline = CustomPipeline(resample_freq=RESAMPLE_FREQ, n_fft=N_FFT, n_mel=N_MELS, augment=False, logmel=False)
        dataset = "validation_feather"
        column = "Filepath" if dataset == "validation_feather" else "Class"
        data = pd.read_feather(CONFIG.get("DATASETS", dataset))
        examples = data[data[column] == "NEG"].sample(5)
        n_samples = 22050
        for idx, row in examples.iterrows():
            signal = row.iloc[2][0].copy()
            signal = np.expand_dims(signal, axis=0)
            
            if signal.shape[1] > n_samples:
                signal = signal[:, :n_samples]
            
            signal = torch.Tensor(signal)
            if signal.shape[1] < n_samples:
                diff_len = n_samples - signal.shape[1]
                padding = (0, diff_len)
                signal = torch.nn.functional.pad(signal, padding)

            img = pipeline.forward(signal)
            visualize_pcen(image=img.squeeze().numpy(), title=f"PCEN of {row.iloc[0]}")
