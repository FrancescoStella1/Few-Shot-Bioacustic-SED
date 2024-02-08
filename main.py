from argparse import ArgumentParser
import sys

import torch

from config import CONFIG
from evaluator import Evaluator
from models import relation_net as rnet
from models import siamese_net as snet
from models import triplet_net as tnet
from models.embedding_cnn import Cnn, get_embedding_model, load_params_from_trial, train_cnn_main
from optimization.optimizer import optimize
from preprocessing.feature_extraction import CustomDataset
import utils
import visualization.plotting as plt

N_FFT = CONFIG.getint("PREPROCESSING", "n_fft")
N_MELS = CONFIG.getint("PREPROCESSING", "n_mels")
RESAMPLE_FREQ = CONFIG.getint("PREPROCESSING", "resampling_frequency")


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generate_dataset", nargs=1, type=str,
                        help="Generates dataset containing time signals.")
    parser.add_argument("--perform_finetuning", nargs=1, type=str, help="Performs finetuning on the specified model.")
    parser.add_argument("--plot_architecture", nargs=1, type=str, help="Plots the architecture for the specified model.")
    parser.add_argument("--plot_dataset_stats", action="count", help="Plots some statistics about the training set.")
    parser.add_argument("--plot_training_stats", nargs=1, type=str,
                        help="Plots training stats for the given model.")
    parser.add_argument("--plot_validation_stats", nargs=1, type=str, help="Plots validation stats for the given model")
    parser.add_argument("--run_model_selection", nargs=1, type=str, help="Performs model selection on the given architecture.")
    parser.add_argument("--train_cnn", action="count", help="Trains CNN defined through specific trial parameters.")
    parser.add_argument("--train_rnet", action="count", help="Trains the Relation Network model.")
    parser.add_argument("--train_snet", action="count", help="Trains Siamese Network.")
    parser.add_argument("--train_tnet", action="count", help="Trains Triplet Network.")
    parser.add_argument("--validate_model", nargs=1, type=str,
                        help="Validates the given model on the validation dataset.")
    parser.add_argument("--visualize_embeddings", nargs=1, type=str,
                        help="Visualizes the embeddings of the given model after performing KMeans.")
    parser.add_argument("--visualize_feature_map", nargs=1, type=str,
                        help="Visualizes the feature map of selected layers from the chosen model.")
    args = parser.parse_args()


    if args.generate_dataset:
        split = sys.argv[2]
        if split in ["train", "training"]:
            db = CustomDataset(split="training", include_neg_class=False)
        elif split in ["val", "validation"]:
            db = CustomDataset(split="validation", include_neg_class=True)
        elif split in ["test", "testing", "eval", "evaluation"]:
            db = CustomDataset(split="testing")
        print("Done.")

    if args.perform_finetuning:
        model_name = sys.argv[2].lower()
        if model_name in ["snet", "siamese_net", "siamese_network"]:
            snet.finetune_snet(split="validation")
        elif model_name in ["rnet", "relation_net", "relation_network"]:
            rnet.finetune_rnet(split="validation")

    if args.plot_architecture:
        model_name = sys.argv[2].lower()
        if model_name == "cnn":
            cnn = get_embedding_model(truncate=False, flatten=False)
            plt.plot_architecture(model=cnn, model_name=model_name)

        elif model_name in ["tnet", "triplet_net", "triplet_network"]:
            cnn = tnet.get_embedding_model()
            model = tnet.TripletNet(embedding_model=cnn)
            plt.plot_architecture(model=model, model_name="tnet")

        elif model_name in ["rnet", "relation_net", "relation_network"]:
            cnn = tnet.get_embedding_model(flatten=False)
            model = rnet.create_model(embedding_model=cnn)
            plt.plot_architecture(model=model, model_name="rnet")

        elif model_name in ["snet", "siamese_net", "siamese_network"]:
            cnn = tnet.get_embedding_model()
            model = snet.SiameseNet(embedding_model=cnn)
            plt.plot_architecture(model=model, model_name="snet")

        else:
            print("Please specify a model between 'cnn', 'rnet' or 'tnet'.")

    if args.plot_dataset_stats:
        plt.plot_dataset_stats(filepath="serialized/dataset/training.feather")
        
    if args.plot_training_stats:
        model_name = sys.argv[2].lower()
        plt.visualize_training_loss(model_name, "Training/Validation loss")

    if args.plot_validation_stats:
        model_name = sys.argv[2].lower()
        if model_name in ["rnet", "relation_net", "relation_network"]:
            rnet.eval_rnet()
        elif model_name in ["snet", "siamese_net", "siamese_network"]:
            snet.eval_snet()

    if args.run_model_selection:
        model_name = sys.argv[2].lower()
        optimize(model_name=model_name)
        print("Done.")

    if args.train_cnn:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trial_name = CONFIG.get("OPTIMIZATION", "trial_name")
        config = load_params_from_trial(trial_name=trial_name)["config"]
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        momentum_decay = config["momentum_decay"]
        class_weights = utils.load_class_weights()
        model = Cnn(input_dim=(87, 128), n_classes=46, class_weights=class_weights, out_channels_1=config["out_channels_1"],
                    out_channels_2=config["out_channels_2"], out_channels_3=config["out_channels_3"],
                    out_channels_4=config["out_channels_4"], kernel_size_1=config["kernel_size_1"],
                    kernel_size_2=config["kernel_size_2"], kernel_size_3=config["kernel_size_3"],
                    kernel_size_4=config["kernel_size_4"], padding=config["padding"], bn_momentum=0.1,
                    out_features_1=config["out_features_1"], out_features_2=config["out_features_2"],
                    dropout_1=config["dropout_1"], dropout_2=config["dropout_2"], dropout_3=config["dropout_3"],
                    dropout_4=config["dropout_4"], dropout_5=config["dropout_5"]).to(device)
        opt = torch.optim.NAdam(params=model.parameters(), lr=lr,
                                  weight_decay=weight_decay,
                                  momentum_decay=momentum_decay)
        hist = train_cnn_main(model=model, optimizer=opt, epochs=30)
    
    if args.train_rnet:
        rnet.train_rnet_main()

    if args.train_snet:
        snet.train_siamese_net()

    if args.train_tnet:
        tnet.train_triplet_net()

    if args.visualize_embeddings:
        model_name = sys.argv[2].lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"
     
        if model_name in ["triplet_net", "tnet", "triplet_network"]:
            activations = torch.Tensor([]).to(device)
            dataloader = tnet.get_dataloader(split="train")
            checkpoint = tnet.load_model()
            labels = []
            model, _ = tnet.create_model_from_trial()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            standardize = False

            with torch.no_grad():
                n_batches = 768 // CONFIG.getint("TRAINING", "batch_size")
                for images, _, batch_labels, label_ids in iter(dataloader):
                    if n_batches <= 0:
                        break
                    embeddings = model.forward(images.to(device))
                    activations = embeddings if activations.size()[0] == 0 \
                        else torch.row_stack((activations, embeddings))
                    labels += batch_labels
                    n_batches -= 1

        elif model_name == "rnet":
            dataloader = rnet.get_dataloader(split="train")
            checkpoint = rnet.load_model()
            cnn = tnet.get_embedding_model(flatten=False)
            model = rnet.create_model(embedding_model=cnn)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model = model.eval()
            rnet.register_hooks(model)
            standardize = False
            layer = model._modules["g"]._modules["5"]  # ["15"]
            layer.register_forward_hook(rnet.get_activation("fc"))
            activations = torch.Tensor([]).to(device)
            labels = []
            with torch.no_grad():
                n_batches = 6
                for (support_images, query_images), _, (support_labels, query_labels) in iter(dataloader):
                    if n_batches <= 0:
                        break
                    predictions = model.forward(support_images[0].to(device), query_images[0].to(device))
                    activation = torch.flatten(rnet.activations["fc"], start_dim=1)
                    activations = activation if activations.size()[0] == 0 \
                        else torch.row_stack((activations, activation))
                    labels += query_labels
                    n_batches -= 1
        
        elif model_name == "snet":
            dataloader = snet.get_dataloader(split="val_train")
            cnn = tnet.get_embedding_model()
            model = snet.SiameseNet(embedding_model=cnn).to(device)
            checkpoint = snet.load_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            # snet.register_hooks(model)
            model = model.eval()
            standardize = False
            activations = torch.Tensor([]).to(device)
            flatten_layer = model.embedding_model._modules["3"]
            flatten_layer.register_forward_hook(snet.get_activation("flatten"))
            labels = []
            with torch.no_grad():
                n_batches = 6
                for images, label_ids, batch_labels in iter(dataloader):
                    if n_batches <= 0:
                        break
                    anchor, query = images
                    anchor = anchor[0].to(device)
                    query = query[0].to(device)
                    anchor_labels, query_labels = batch_labels
                    predictions, _ = model.forward(anchor, query)
                    activation = snet.activations["flatten"]
                    activations = activation if activations.size(0) == 0 \
                        else torch.row_stack((activations, activation))
                    labels += query_labels
                    n_batches -= 1
            
        plt.visualize_embeddings_tsne(embeddings=activations.to("cpu"), labels=labels, three_dim=False, standardize=standardize)
        # plt.visualize_embeddings_kmeans(embeddings=activations.to("cpu"), labels=labels, three_dim=False, standardize=standardize)

    if args.visualize_feature_map:
        model_name = sys.argv[2].lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name == "tnet":
            dataloader = tnet.get_dataloader(split="val_train")
            model, _ = tnet.create_model_from_trial()
            model.to(device)
            model = model.eval()
            # tnet.register_hooks(model)
            first_cnn = model.embedding_cnn.conv1._modules["0"]
            second_cnn = model.embedding_cnn.conv2._modules["0"]
            third_cnn = model.embedding_cnn.conv3._modules["0"]
            first_cnn.register_forward_hook(tnet.get_activation("conv1"))
            second_cnn.register_forward_hook(tnet.get_activation("conv2"))
            third_cnn.register_forward_hook(tnet.get_activation("conv3"))
            with torch.no_grad():
                images, _, labels, label_ids = next(iter(dataloader))
                predictions = model.forward(images[0].unsqueeze(0).to(device))
                plt.visualize_feature_map(tnet.activations["conv1"].detach().squeeze().cpu().numpy(), f"Activation of first CNN layer for {labels[0]}")
                plt.visualize_feature_map(tnet.activations["conv2"].detach().squeeze().cpu().numpy(), f"Activation of second CNN layer for {labels[0]}")
                plt.visualize_feature_map(tnet.activations["conv3"].detach().squeeze().cpu().numpy(), f"Activation of third CNN layer for {labels[0]}")
                sys.exit(1)

        elif model_name == "snet":
            pass

        elif model_name == "rnet":
            dataloader = rnet.get_dataloader(split="val_train")
            checkpoint = rnet.load_model()
            cnn = get_embedding_model(flatten=False)
            model = rnet.create_model_from_trial(embedding_model=cnn)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            with torch.no_grad():
                (support_images, query_images), _, (support_labels, query_labels) = next(iter(dataloader))
                predictions = model.forward(support_images[0].to(device), query_images[0].to(device))
                sys.exit(1)

    if args.validate_model:
        model_name = sys.argv[2].lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name in ["rnet", "relation_net", "relation_network"]:
            cnn = tnet.get_embedding_model(flatten=False)
            model = rnet.create_model(embedding_model=cnn)
            checkpoint = rnet.load_model(load_finetuned=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model = model.eval()
            evaluator = Evaluator(base_path="dataset/Validation_Set", model=model)
            evaluator.evaluate_model(debugging=False)
        elif model_name in ["snet", "siamese_net", "siamese_network"]:
            cnn = tnet.get_embedding_model()
            model = snet.SiameseNet(embedding_model=cnn)
            checkpoint = snet.load_model(load_finetuned=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model = model.eval()
            evaluator = Evaluator(base_path="dataset/Validation_Set", model=model)
            evaluator.evaluate_model(debugging=False)