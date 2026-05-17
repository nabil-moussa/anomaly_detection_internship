import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--dataset", type=str.upper, default="SMD")
    parser.add_argument("--group", type=str, default="1-1",
                        help="Required for SMD: <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--custom_csv", type=str, default=None)
    parser.add_argument("--custom_label_csv", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--stream_name", type=str, default=None)

    # Conv layer
    parser.add_argument("--kernel_size", type=int, default=7)

    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=False)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)

    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=300)

    # Forecasting head
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=300)

    # Reconstruction head
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=300)
    parser.add_argument("--use_vae", type=str2bool, default=True)
    parser.add_argument("--kl_weight", type=float, default=0.1)

    # Other model
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)
    parser.add_argument("--use_sr_cleaning", type=str2bool, default=True)

    # Multi-seed
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="42,123,456")

    # Predictor / thresholding
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--level", type=float, default=0.98)
    parser.add_argument("--q", type=float, default=1e-3)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    return parser
