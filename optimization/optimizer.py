import os
import tempfile

from ray import train, tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search import ConcurrencyLimiter
import torch

from config import CONFIG
from models.embedding_cnn import Cnn
from models.embedding_cnn import eval_cnn, train_cnn
from models.relation_net import RelationNet
from models.relation_net import eval_rnet, train_rnet
import models.triplet_net as tnet
from utils import load_class_weights


def _get_config_space(model_name: str) -> dict:
    if model_name == "cnn":
        return _get_cnn_config_space()
    elif model_name in ["rnet", "relation_net", "relation_network"]:
        return _get_rnet_config_space()
    elif model_name in ["tnet", "triplet_net", "triplet_network"]:
        return _get_tnet_config_space()
    return None


def _get_cnn_config_space() -> dict:
    config_space = {
        "activation": tune.choice(["relu", "prelu"]),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        # "momentum_decay": tune.loguniform(2e-4, 2e-1),
        "padding": tune.choice(["same", "valid"]),
        "bn_momentum": tune.uniform(0.1, 0.8),
        "kernel_size_1": tune.choice([2, 3, 5, 7, 10]),
        "kernel_size_2": tune.choice([2, 3, 5, 7, 10]),
        "kernel_size_3": tune.choice([2, 3, 5, 7, 10]),
        "kernel_size_4": tune.choice([2, 3, 5, 7, 10]),
        "out_channels_1": tune.choice([16, 32]),
        "out_channels_2": tune.choice([16, 32, 64]),
        "out_channels_3": tune.choice([16, 32, 64]),
        "out_channels_4": tune.choice([16, 32, 64]),
        "dropout_1": tune.uniform(0.1, 0.3),
        "dropout_2": tune.uniform(0.1, 0.7),
        "dropout_3": tune.uniform(0.1, 0.7),
        "dropout_4": tune.uniform(0.1, 0.7),
        "dropout_5": tune.uniform(0.1, 0.7),
        "out_features_1": tune.choice([256, 512, 1024]),
        "out_features_2": tune.choice([64, 128, 256])
    }
    return config_space


def _get_rnet_config_space() -> dict:
    config_space = {
        "bn_momentum": tune.uniform(0.1, 0.8),
        "dropout_1": tune.uniform(0.2, 0.8),
        "dropout_2": tune.uniform(0.05, 0.4),
        "dropout_3": tune.uniform(0.05, 0.4),
        "dropout_4": tune.uniform(0.01, 0.3),
        "kernel_size_1": tune.choice([2, 3, 5, 7]),
        "kernel_size_2": tune.choice([2, 3, 5, 7]),
        "kernel_size_3": tune.choice([2, 3, 5, 7]),
        "kernel_size_4": tune.choice([2, 3, 5, 7]),
        # "momentum_decay": tune.loguniform(2e-4, 2e-1),
        "out_channels_1": tune.choice([32, 64]),
        "out_channels_2": tune.choice([32, 64]),
        "out_features_1": tune.choice([256, 512, 1024]),
        "out_features_2": tune.choice([16, 32, 64, 128]),
        "weight_decay": tune.loguniform(1e-4, 1e-1)
    }
    return config_space


def _get_tnet_config_space() -> dict:
    config_space = {
        "bn_momentum": tune.uniform(0.1, 1.0),
        "dropout_1": tune.uniform(0.2, 0.8),
        "dropout_2": tune.uniform(0.05, 0.5),
        "dropout_3": tune.uniform(0.05, 0.5),
        "kernel_size_1": tune.choice([2, 3, 5, 7, 10]),
        "kernel_size_2": tune.choice([2, 3, 5, 7, 10]),
        "kernel_size_3": tune.choice([2, 3, 5, 7, 10]),
        "momentum": tune.uniform(0.1, 1.0),
        "out_channels_1": tune.choice([16, 32, 64]),
        "out_channels_2": tune.choice([32, 64]),
        "out_channels_3": tune.choice([32, 64, 128]),
        "weight_decay": tune.loguniform(1e-4, 1e-1)
    }
    return config_space


def _get_search_algorithm(model_name: str) -> ConcurrencyLimiter:
    """
    Returns a Bayesian Hyperparameter Optimization (BHO) algorithm
    on a predefined hyperparameter space.

    Returns
    -------
    hyperopt: ConcurrencyLimiter
        Wrapper object for the BOHB search algorithm.
    """
    config_space = _get_config_space(model_name=model_name)
    algorithm = TuneBOHB(
        bohb_config=config_space,
        metric="val_f1_score",
        mode="max",
        seed=42
    )
    algorithm = ConcurrencyLimiter(algorithm, max_concurrent=4)
    return algorithm


def _cnn_objective(config) -> None:
    """
    This function wraps the embedding model training function and collects the metrics to perform
    model selection.

    Parameters
    ----------
    config: CS.ConfigurationSpace
        Configuration space object containing parameters to optimize.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_decay = config["weight_decay"]
    # momentum_decay = config["momentum_decay"]
    class_weights = load_class_weights()
    model = Cnn(input_dim=(87, 128), n_classes=46, class_weights=class_weights, out_channels_1=config["out_channels_1"],
                out_channels_2=config["out_channels_2"], out_channels_3=config["out_channels_3"],
                out_channels_4=config["out_channels_4"], kernel_size_1=config["kernel_size_1"],
                kernel_size_2=config["kernel_size_2"], kernel_size_3=config["kernel_size_3"],
                kernel_size_4=config["kernel_size_4"], padding=config["padding"], bn_momentum=config["bn_momentum"],
                dropout_1=config["dropout_1"], dropout_2=config["dropout_2"], dropout_3=config["dropout_3"],
                dropout_4=config["dropout_4"], dropout_5=config["dropout_5"], out_features_1=config["out_features_1"],
                out_features_2=config["out_features_2"]).to(device)
    '''
    optim = torch.optim.NAdam(params=model.parameters(), lr=1e-3,
                              weight_decay=weight_decay,
                              momentum_decay=momentum_decay)
    '''
    optim = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=weight_decay,
                            nesterov=True, momentum=0.9, dampening=0)
    
    model = train_cnn(model, optim, 6)
    val_loss, val_f1_score = eval_cnn(model)
    train.report({
        "val_loss": val_loss,
        "val_f1_score": val_f1_score
    })


def _relation_net_objective(config):
    """
    This function wraps the Relation Network training function and collects the metrics to perform
    model selection.

    Parameters
    ----------
    config: CS.ConfigurationSpace
        Configuration space object containing parameters to optimize.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = tnet.get_embedding_model(flatten=False)
    for _, param in cnn.named_parameters():
        param.requires_grad = False
    model = RelationNet(embedding_model=cnn, bn_momentum=config["bn_momentum"], dropout_1=config["dropout_1"], dropout_2=config["dropout_2"],
                        dropout_3=config["dropout_3"], dropout_4=config["dropout_4"], kernel_size_1=config["kernel_size_1"], 
                        kernel_size_2=config["kernel_size_2"], kernel_size_3=config["kernel_size_3"], kernel_size_4=config["kernel_size_4"],
                        out_channels_1=config["out_channels_1"], out_channels_2=config["out_channels_2"], out_features_1=config["out_features_1"],
                        out_features_2=config["out_features_2"]).to(device)
    model_params = [param for param in model.parameters() if param.requires_grad == True]
    optim = torch.optim.NAdam(params=model_params, lr=1e-3, weight_decay=config["weight_decay"],
                              momentum_decay=config["momentum_decay"])
    
    # optim = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=config["weight_decay"],
                            # nesterov=True, momentum=0.9, dampening=0)
    model = train_rnet(model=model, optimizer=optim, epochs=6)
    val_loss, val_f1_score = eval_rnet(model)
    train.report({
        "val_loss": val_loss,
        "val_f1_score": val_f1_score
    })


def _triplet_net_objective(config):
    """
    This function wraps the Triplet Network training function and collects the metrics to
    perform model selection.

    Parameters
    ----------
    config: dict
        Configuration space object containing parameters to optimize.
    """
    cnn = tnet.Cnn(bn_momentum=config["bn_momentum"], dropout_1=config["dropout_1"], dropout_2=config["dropout_2"],
                   dropout_3=config["dropout_3"], out_channels_1=config["out_channels_1"], out_channels_2=config["out_channels_2"],
                   out_channels_3=config["out_channels_3"], kernel_size_1=config["kernel_size_1"],
                   kernel_size_2=config["kernel_size_2"], kernel_size_3=config["kernel_size_3"])
    model = tnet.TripletNet(embedding_model=cnn)
    optim = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=config["weight_decay"],
                            nesterov=True, momentum=config["momentum"], dampening=0)
    if train.get_checkpoint():
        checkpoint = train.get_checkpoint()
        with checkpoint.as_directory() as checkpoint_dir:
            model_state, optim_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optim.load_state_dict(optim_state)
    model = tnet.train_tnet(model=model, optimizer=optim, epochs=6)
    val_loss = tnet.eval_tnet(model)
    with tempfile.TemporaryDirectory() as tmp_checkpoint_dir:
        path = os.path.join(tmp_checkpoint_dir, "checkpoint.pt")
        torch.save(
            (model.state_dict(), optim.state_dict()), path
        )
        checkpoint = train.Checkpoint.from_directory(tmp_checkpoint_dir)
        train.report({
            "val_loss": val_loss
        }, checkpoint=checkpoint)


def optimize(model_name: str):
    print("Starting optimization...")
    max_iter = 1
    num_samples = 20
    algo = _get_search_algorithm(model_name)
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=max_iter,
        reduction_factor=2,
        stop_last_trials=False
    )
    if model_name.lower() == "cnn":
        obj_func = _cnn_objective
        gpu = 0.3
        metric = "val_f1_score"
        mode = "max"
    elif model_name.lower() in ["rnet", "relation_net", "relation_network"]:
        obj_func = _relation_net_objective
        gpu = 1
        metric = "val_f1_score"
        mode = "max"
    elif model_name.lower() in ["tnet", "triplet_net", "triplet_network"]:
        obj_func = _triplet_net_objective
        gpu = 0.5
        metric = "val_loss"
        mode = "min"
    else:
        return
    
    os.environ["RAY_DEDUP_LOGS"] = "0"
    trials_path = CONFIG.get("OPTIMIZATION", "trials_dir")
    trials_path = os.path.join(os.path.abspath(os.curdir), trials_path)
    if not os.path.exists(trials_path):
        os.mkdir(trials_path)

    tuner = tune.Tuner(
        obj_func,
        run_config=train.RunConfig(
            name="bohb",
            stop={"training_iteration": max_iter},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True
            )
        ),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            search_alg=algo,
            scheduler=scheduler,
            metric=metric,  # "val_f1_score",
            mode=mode
        )
    )
    if not torch.cuda.is_available():
        results = tuner.fit()
    else:
        config_space = _get_config_space(model_name=model_name)
        results = tune.run(
            obj_func,
            num_samples=num_samples,
            config=config_space,
            resources_per_trial={"cpu": 2, "gpu": gpu},
            metric=metric,  # "val_f1_score",
            mode=mode,
            max_concurrent_trials=6,
            local_dir=trials_path,
            storage_path=trials_path
        )
    # best_hp = results.get_best_result().config  # ("val-f1-score", mode="max").path
    best_hp = results.best_result
    print("Best hyperparameters: ", best_hp)
    # state_dict = torch.load(os.path.join(logdir, "model.pt"))
