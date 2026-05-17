import os
import json
import datetime
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
os.environ["PYTHONHASHSEED"]          = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

from args import get_parser
from utils import (
    get_data, get_target_dims, SlidingWindowDataset, create_data_loaders,
    plot_losses, clean_training_data,
)
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer


def run_single_seed(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    run_id = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")

    dataset         = args.dataset
    window_size     = args.lookback
    normalize       = args.normalize
    n_epochs        = args.epochs
    batch_size      = args.bs
    init_lr         = args.init_lr
    val_split       = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda        = args.use_cuda
    print_every     = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index     = args.group[0]
    index           = args.group[2:]

    # Data loading
    if dataset == "SMD":
        output_path = f"output/SMD/{args.group}"
        (x_train, _), (x_test, y_test) = get_data(
            f"machine-{group_index}-{index}", normalize=normalize
        )
    elif dataset in ["MSL", "SMAP"]:
        output_path = f"output/{dataset}"
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset == "CUSTOM":
        output_path = "output/CUSTOM"
        if args.custom_csv is not None:
            from utils import prepare_custom_dataset
            prepare_custom_dataset(
                csv_path    = args.custom_csv,
                label_path  = args.custom_label_csv,
                train_ratio = args.train_ratio,
                normalize   = args.normalize,
                save_dir    = "datasets/custom",
            )
            (x_train, _), (x_test, y_test) = get_data("CUSTOM", normalize=False)
        else:
            (x_train, _), (x_test, y_test) = get_data("CUSTOM", normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    save_path = f"{output_path}/{run_id}"
    log_dir   = f"{output_path}/logs"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    x_train    = torch.from_numpy(x_train).float()
    x_test     = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    if args.normalize and args.use_sr_cleaning:
        x_train_np = x_train.numpy()
        x_train_np = clean_training_data(x_train_np, threshold=3.0)
        x_train    = torch.from_numpy(x_train_np).float()

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset  = SlidingWindowDataset(x_test,  window_size, target_dims)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset,
        test_dataset=test_dataset,
    )

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        recon_out_dim      = n_features,
        kernel_size        = args.kernel_size,
        use_gatv2          = args.use_gatv2,
        feat_gat_embed_dim = args.feat_gat_embed_dim,
        time_gat_embed_dim = args.time_gat_embed_dim,
        gru_n_layers       = args.gru_n_layers,
        gru_hid_dim        = args.gru_hid_dim,
        forecast_n_layers  = args.fc_n_layers,
        forecast_hid_dim   = args.fc_hid_dim,
        recon_n_layers     = args.recon_n_layers,
        recon_hid_dim      = args.recon_hid_dim,
        dropout            = args.dropout,
        alpha              = args.alpha,
        use_vae            = args.use_vae,
    )

    optimizer          = torch.optim.Adam(model.parameters(), lr=init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion    = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        str(args.__dict__),
        use_vae   = args.use_vae,
        kl_weight = args.kl_weight,
    )

    trainer.fit(train_loader, val_loader)

    losses_serializable = {
        k: [float(v) for v in vals]
        for k, vals in trainer.losses.items()
    }
    with open(f"{save_path}/losses.json", "w") as f:
        json.dump(losses_serializable, f, indent=2)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss:       {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss:          {test_loss[2]:.5f}")

    # Hyperparameter suggestions
    level_q_dict = {
        "SMAP":   (0.90, 0.005),
        "MSL":    (0.90, 0.001),
        "CUSTOM": (0.90, 0.005),
        "SMD-1":  (0.9950, 0.001),
        "SMD-2":  (0.9925, 0.001),
        "SMD-3":  (0.9999, 0.001),
    }
    key   = ("SMD-" + args.group[0]) if dataset == "SMD" else dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q



    trainer.load(f"{save_path}/model.pt")

    prediction_args = {
        "dataset":     dataset,
        "target_dims": target_dims,
        "scale_scores": args.scale_scores,
        "level":       level,
        "q":           q,
        "dynamic_pot": args.dynamic_pot,
        "use_mov_av":  args.use_mov_av,
        "gamma":       args.gamma,
        "save_path":   save_path,
        "use_vae":     args.use_vae,
    }

    predictor = Predictor(trainer.model, window_size, n_features, prediction_args)

    # Label alignment
    if y_test is None:
        label = None
    elif dataset in ["MSL", "SMAP", "SMD"]:
        label = y_test[window_size:]
    else:
        expected_len = len(x_test) - window_size
        if len(y_test) == expected_len:
            label = y_test
        elif len(y_test) == len(x_test):
            label = y_test[window_size:]
        elif len(y_test) > expected_len:
            label = y_test[-expected_len:]
        else:
            label = np.zeros(expected_len, dtype="float32")

    summary = predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    with open(f"{save_path}/config.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Export scores for downstream pipeline
    import sys
    sys.path.append('/Work/Users/nmoussa/mtad-gat-pytorch')
    from export_scores import export_from_saved
    stream_name = args.stream_name if args.stream_name is not None else args.dataset.lower()
    export_from_saved(
        save_path   = save_path,
        stream_name = stream_name,
        handoff_dir = '/Work/Users/nmoussa/handoff',
    )

    return summary


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()
    print(args)

    seeds = [int(s) for s in args.seeds.split(",")][:args.n_seeds]

    all_results = []
    for seed in seeds:
        print(f"\n{'='*50}\nRUNNING SEED {seed}\n{'='*50}")
        result = run_single_seed(args, seed)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*50}\nAVERAGED RESULTS ACROSS {len(seeds)} SEEDS\n{'='*50}")
        for method in ["pot_result", "bf_result"]:
            f1s   = [r[method]["f1"]        for r in all_results]
            precs = [r[method]["precision"]  for r in all_results]
            recs  = [r[method]["recall"]     for r in all_results]
            print(f"\n{method}:")
            print(f"  F1:        {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
            print(f"  Precision: {np.mean(precs):.4f} +- {np.std(precs):.4f}")
            print(f"  Recall:    {np.mean(recs):.4f} +- {np.std(recs):.4f}")

        avg_summary = {
            method: {
                "f1_mean":        float(np.mean([r[method]["f1"]       for r in all_results])),
                "f1_std":         float(np.std( [r[method]["f1"]       for r in all_results])),
                "precision_mean": float(np.mean([r[method]["precision"] for r in all_results])),
                "recall_mean":    float(np.mean([r[method]["recall"]    for r in all_results])),
                "individual_runs": [r[method] for r in all_results],
            }
            for method in ["pot_result", "bf_result"]
        }
        with open(f"averaged_summary_{args.dataset}.json", "w") as f:
            json.dump(avg_summary, f, indent=2)
        print("\nAveraged summary saved.")
